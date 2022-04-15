

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import pickle
import torch
import vtk
from torch_geometric import utils
from torch_geometric.data import Data
from vtk.util.numpy_support import vtk_to_numpy

import setup
from data_augmentation import create_knn_data
from setup import *

GRAPH_FEATURES_ONEHOT = {
    'LAD': torch.tensor([1., 0., 0.]),
    'LCX': torch.tensor([0., 1., 0.]),
    'RCA': torch.tensor([0., 0., 1.])
}


def parse_data_file_name(file):
    """
    File name is assumed to be structured into "parts" separated by underscore `_`:
        <part1>_<part2>*.pt
    The first part is the patient id, the second part is the artery type.
    The next parts are optional and may describe data augmentation.
    If one of the following patterns *is in* any optional part (i.e. after part 1 and 2, if any):
        - rot
        - KNN
        - NOISE
    Matching is case-INsensitive.
    @param file: str, file name or file path
    @returns: is_patient: bool, patient_id: str, is_augmented: bool
              patient id and whether the file is original or augmented data
    """
    if Path(file).suffix != '.pt':
        return False, None, None

    # File name without (last) suffix
    fname = Path(file).stem
    # Collect parts
    patient_id, artery, *optional = fname.split('_')
    # Enforce backward compatibility, may probably be removed though
    if len(patient_id) != 6:
        return False, None, None
    # Find if augmented
    is_augmented = False
    for opt_part in optional:
        # TODO: this is not default deny => bad design
        if any(pattern in opt_part.lower() for pattern in ('knn', 'noise', 'rot', 'diffusion')):
            is_augmented = True
            break

    return True, patient_id, is_augmented


def read_labels(filename):
    """
    Creates list of culprit segments
    0 for non-culprit, 1 for culprit
    """
    logging.info(f'parsing labels ...')
    df = pd.read_excel(filename, engine='openpyxl')
    idx_culprit = df.index[df['FC'] == 1].tolist()
    culprits = df['Code'][idx_culprit].to_list()
    return culprits


def read_tsvi(filename):
    """Creates dictionary between segment and TSVI measurement"""
    logging.info('parsing tsvi ...')
    df = pd.read_excel(filename)
    seg_to_tsvi = dict()
    for index, row in df.iterrows():
        seg_to_tsvi[row['Code']] = row['TSVI']
    return seg_to_tsvi


def read_poly_data(filename):
    logging.debug(f'reading poly data from file {filename}')
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def get_edge_index(surface):
    """Returns list of edges of the mesh as pairs of nodes"""
    logging.debug('parsing edge indices')
    cells = vtk_to_numpy(surface.GetPolys().GetData())
    triangles = cells.reshape(-1, 4)[:, 1:4]
    T = np.concatenate(
        [np.expand_dims(np.concatenate([triangles[:, 0],
                                        triangles[:, 0],
                                        triangles[:, 1]],
                                       axis=0),
                        axis=1),
         np.expand_dims(np.concatenate([triangles[:, 1],
                                        triangles[:, 2],
                                        triangles[:, 2]],
                                       axis=0),
                        axis=1)],
        axis=1)
    E = np.concatenate([np.max(T,
                               axis=1,
                               keepdims=True),
                        np.min(T,
                               axis=1,
                               keepdims=True)],
                       axis=1)
    E = np.unique(E, axis=0)
    # Sanity check: no self loop
    assert (E[:, 0] == E[:, 1]).any() == False
    # We have undirected graph -> build reverse edges
    E_rev = E[:, [1, 0]]
    E = np.concatenate((E, E_rev))
    # Sanity check
    assert utils.is_undirected(torch.from_numpy(E.T))

    return E


def create_patient_dict(path, savepath):
    """
    Creates dictionary with patients as key and list of segments
    for each patient as item.

    :param path: path to data source
    :param savepath: where to save the dictionary
    """
    logging.info(f'creating patient dict with source -> dest: {path} -> {savepath}')
    files = []
    for file in os.listdir(path):
        if file.endswith(".vtp"):
            files.append(file)
    patient_seg = [file.split('_WSS')[0] for file in files]
    data_dict = {}
    for item in patient_seg:
        patient = item.split('_')[0]
        if patient in data_dict.keys():
            data_dict[patient].append(item)
        else:
            data_dict[patient] = []
            data_dict[patient].append(item)

    with open(savepath, 'w') as f:
        json.dump(data_dict, f, indent=4)


def load_patient_dict(path=None):
    if path is None:
        _, path = setup.get_data_paths(path_out_only=True)

    with open(path.joinpath('patient_dict.json'), 'rb') as f:
        patient_dict = json.load(f)

    return patient_dict


def create_mesh_data(surface, div=False, coord=False):
    """
        Creates data for the mesh where each node contains the 30 WSS
        magnitude and other features.

        :param div: whether to include the WSS divergence
        :param coord: where to include the 3D point coordinates
        :return data: data
    """
    points = vtk_to_numpy(surface.GetPoints().GetData())
    nb_features = 2 if div else 1
    data = np.empty((points.shape[0], nb_features, 30))
    for t in range(30):
        wssArray_t = vtk_to_numpy(
                         surface.GetPointData()
                                .GetArray('WSS_'+'{0:03}'.format(t + 1)))
        X = np.expand_dims(wssArray_t, axis=1)
        if div:
            div_t = vtk_to_numpy(
                         surface.GetPointData()
                                .GetArray('DIV_'+'{0:03}'.format(t + 1)))
            X = np.concatenate([X, np.expand_dims(div_t, axis=1)], axis=1)
        data[:, :, t] = X
    data = data.reshape(-1, nb_features * 30)

    if coord:
        pts_coord = vtk_to_numpy(surface.GetPoints().GetData())
        data = np.concatenate((data, pts_coord), axis=1)
    return data


def create_graph_features(artery_type: str):
    onehot = GRAPH_FEATURES_ONEHOT[artery_type]
    # WARNING: the reshaping is absolutely needed
    return onehot.reshape(1, -1)


@arg_logger
def create_data(name, k_neigh, rot_angles, path_input, path_label, path_write):
    """
    Create .pt files from .vtk files. Each .pt file is a serialized torch_geometric.data.Data object with attributes:
        x: node features, size n_nodes x n_features (n_features might be zero)
        coord: node coordinates, size n_nodes x 3
        g_x: graph features
        edge_index: adjacency list, size 2 x n_nodes
        y: output culprit/non culprit, size n_nodes
    """
    # Initialize writing path
    savepath = path_write.joinpath(name)
    if not savepath.exists():
        savepath.mkdir()

    # Create patient-segment dictionary
    path_patient_dic = path_write.joinpath('patient_dict.json')
    if not path_patient_dic.exists():
        create_patient_dict(path_input, path_patient_dic)
    logging.info('loading patient dictionary ...')
    patient_dict = load_patient_dict()

    # Parse labels
    culprits = read_labels(path_label)
    if name == 'TsviPlusCnc':
        seg_to_tsvi = read_tsvi(path_label)

    logging.info(f'number of keys in patient dict: {len(patient_dict)}')

    # Loop over patients: process data from input path and write to output path
    counter = 0
    for pt in patient_dict.keys():
        for pt_seg in patient_dict[pt]:
            logging.debug(f'processing item {pt_seg}')
            counter += 1
            culprit = 1 if pt_seg in culprits else 0
            if name == 'TsviPlusCnc':
                tsvi = seg_to_tsvi[pt_seg]
            # Graph features (LAD, LDX, RCA)
            graph_features = create_graph_features(pt_seg[-3:])
            # read data
            surface = read_poly_data(path_input + '/' + pt_seg + '_WSSMag.vtp')
            # get points of surface
            coord = torch.from_numpy(
                vtk_to_numpy(
                    surface.GetPoints()
                           .GetData()
                )
            ).type(torch.FloatTensor)

            num_nodes = len(coord)
            edges = get_edge_index(surface)
            edge_index = torch.from_numpy(
                edges.transpose()
            ).type(torch.LongTensor)

            if name == 'CoordToCnc':
                y = culprit
                # No node features in this case
                segment_data = torch.empty((num_nodes, 0))
                #segment_data = vtk_to_numpy(surface.GetPoints().GetData())
            elif name == 'WssToCnc':
                y = culprit
                segment_data = create_mesh_data(surface, div=False, coord=False)
            elif name == 'WssPlusCnc':
                wss_mag = np.mean(create_mesh_data(surface,
                                                   div=False,
                                                   coord=False),
                                  axis=1).reshape(-1, 1)
                one_culprits = np.ones((num_nodes, 1)) * culprit
                y = torch.from_numpy(
                    np.concatenate((wss_mag, one_culprits), axis=1)
                ).type(torch.FloatTensor)
                segment_data = vtk_to_numpy(surface.GetPoints().GetData())
            elif name == 'TsviPlusCnc':
                segment_data = vtk_to_numpy(surface.GetPoints().GetData())
                y = torch.tensor([culprit, tsvi], dtype=torch.float)
            else:
                raise ValueError

            if isinstance(segment_data, np.ndarray):
                segment_data = torch.from_numpy(segment_data).type(torch.FloatTensor)
            data = Data(x=segment_data,
                        coord=coord,
                        g_x=graph_features,
                        edge_index=edge_index,
                        y=y)
            path_file = savepath.joinpath(pt_seg + '.pt')
            logging.info(f'saving data in file {str(path_file)}')
            torch.save(data, path_file)

            # k_neighbours data aug:
            if k_neigh != 0:
                edges = create_knn_data(coord, k_neigh)
                data = Data(x=segment_data,
                            coord=coord,
                            g_x=graph_features,
                            edge_index=edges,
                            y=y)

                path_file = savepath.joinpath(f'{pt_seg}_KNN{k_neigh}.pt')
                logging.info(f'saving KNN-augmented data in file {str(path_file)}')
                torch.save(data, path_file)

            # rotation_data aug
            for angle in rot_angles:
                mat = torch.tensor([[1, 0, 0],
                                    [0, np.cos(math.radians(angle)),
                                     -np.sin(math.radians(angle))],
                                    [0, np.sin(math.radians(angle)),
                                     np.cos(math.radians(angle))]]).float()
                coord = torch.transpose(
                    mat@torch.transpose(coord, 0, 1), 0, 1
                )
                data = Data(x=segment_data,
                            coord=coord,
                            g_x=graph_features,
                            edge_index=edge_index,
                            y=y)
                path_file = savepath.joinpath(f'{pt_seg}_rot{angle:03d}.pt')
                logging.info(f'saving rotation-augmented data in file {str(path_file)}')
                torch.save(data, path_file)
    return


if __name__ == '__main__':
    logging.info('Running create_data.py ...')
    path_data_in, path_data_out = get_data_paths()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n', '--dataset_name',
        type=str,
        help='name of dataset to be created',
        default='WssToCnc',
        choices=['CoordToCnc', 'WssToCnc', 'WssPlusCnc', 'TsviPlusCnc']
    )
    parser.add_argument(
        '-k', '--augment_data',
        type=int,
        help='number of neighbours used for KNN',
        default=0  # used to be 5!
    )
    parser.add_argument(
        '-s', '--data_source',
        type=str,
        help='path to raw data',
        default=str(path_data_in.joinpath('CFD/ClinicalCFD/MagnitudeClinical'))
    )
    parser.add_argument(
        '-l', '--labels_source',
        type=str,
        help='path to data label',
        default=str(path_data_in.joinpath('CFD/labels/WSSdescriptors_AvgValues.xlsx'))
    )
    args = parser.parse_args()

    create_data(
        name=args.dataset_name,
        rot_angles=[-18, -9, 9, 18],
        k_neigh=args.augment_data,
        path_input=args.data_source,
        path_label=args.labels_source,
        path_write=path_data_out
    )
# options are ['CoordToCnc', 'WssToCnc', 'WssPlusCnc', 'TsviPlusCnc']
