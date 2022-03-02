import os
import argparse

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import pickle5 as pickle  # to work with python3.6.5
# import pickle
import pandas as pd
import torch
import math

# from datasets import MyData
from torch_geometric.data import Data
from data_augmentation import create_knn_data


NAME_TO_INT = {'LAD': 0, 'LCX': 1, 'RCA': 2}


def read_labels(filename):
    """
    Creates list of culprit segments
    0 for non-culprit, 1 for culprit
    """
    df = pd.read_excel(filename, engine='openpyxl')
    idx_culprit = df.index[df['FC'] == 1].tolist()
    culprits = df['Code'][idx_culprit].to_list()
    return culprits


def read_tsvi(filename):
    """Creates dictionary between segment and TSVI measurement"""
    df = pd.read_excel(filename)
    seg_to_tsvi = dict()
    for index, row in df.iterrows():
        seg_to_tsvi[row['Code']] = row['TSVI']
    return seg_to_tsvi


def read_poly_data(filename):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def get_edge_index(surface):
    """Returns list of edges of the mesh as pairs of nodes"""
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
    E_rev = np.concatenate([np.max(E, axis=1, keepdims=True),
                            np.min(E, axis=1, keepdims=True)],
                           axis=1)
    E = np.concatenate((E, E_rev))
    return E


def create_patient_dict(path, savepath):
    """
    Creates dictionary with patients as key and list of segments
    for each patient as item.

    :param path: path to data source
    :param savepath: where to save the dictionary
    """
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

    with open(os.path.join(savepath, 'patient_dict.pickle'), 'wb') as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


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


def create_data(name, k_neigh, rot_angles, path, label_path):
    savepath = os.path.join('data/', 'CoordToCncKNN200/')

    # Create patient-segment dictionary
    if not os.path.exists('data/patient_dict.pickle'):
        create_patient_dict(path, 'data/')
    with open('data/patient_dict.pickle', 'rb') as f:
        data_dict = pickle.load(f)

    culprits = read_labels(label_path)
    if name == 'TsviPlusCnc':
        seg_to_tsvi = read_tsvi(label_path)

    print("Nb of keys: ", len(data_dict.keys()))

    counter = 0
    for pt in data_dict.keys():
        for pt_seg in data_dict[pt]:
            print(counter)
            counter += 1
            culprit = 1 if pt_seg in culprits else 0
            if name == 'TsviPlusCnc':
                tsvi = seg_to_tsvi[pt_seg]
            seg = pt_seg[-3:]  # LAD, RCA or LCX
            cur_path = os.path.join(savepath, pt_seg + '.pt')  # savename
            # read data
            surface = read_poly_data(path + '/' + pt_seg + '_WSSMag.vtp')
            # get points of surface
            pos = torch.from_numpy(
                                vtk_to_numpy(
                                    surface.GetPoints()
                                           .GetData())).type(torch.FloatTensor)
            num_nodes = len(pos)
            edges = get_edge_index(surface)
            edge_index = torch.from_numpy(
                                     edges.transpose()
                                         ).type(torch.LongTensor)

            if name == 'CoordToCnc':
                y = culprit
                segment_data = vtk_to_numpy(surface.GetPoints().GetData())
            elif name == 'WssToCnc':
                y = culprit
                segment_data = create_mesh_data(surface, div=True, coord=False)
            elif name == 'WssPlusCnc':
                wss_mag = np.mean(create_mesh_data(surface,
                                                   div=False,
                                                   coord=False),
                                  axis=1).reshape(-1, 1)
                one_culprits = np.ones((num_nodes, 1)) * culprit
                y = torch.from_numpy(
                                np.concatenate((wss_mag, one_culprits),
                                               axis=1)).type(torch.FloatTensor)
                segment_data = vtk_to_numpy(surface.GetPoints().GetData())
            elif name == 'TsviPlusCnc':
                segment_data = vtk_to_numpy(surface.GetPoints().GetData())
                y = torch.tensor([culprit, tsvi], dtype=torch.float)
            segment_data = torch.from_numpy(
                                        segment_data).type(torch.FloatTensor)
            data = Data(x=segment_data,
                        edge_index=edge_index,
                        y=y,
                        pos=pos,  # TODO: replace by torch.tensor??
                        segment=NAME_TO_INT[seg])
            # torch.save(data, cur_path)

            # k_neighbours data aug:
            if k_neigh != 0:
                edges = create_knn_data(surface, k_neigh)
                data = Data(x=segment_data,
                            edge_index=edges,
                            y=y,
                            pos=pos,
                            segment=NAME_TO_INT[seg])
                torch.save(data, os.path.join(savepath, pt_seg + '.pt'))

            # rotation_data aug
            for angle in rot_angles:
                mat = torch.tensor([[1, 0, 0],
                                    [0, np.cos(math.radians(angle)),
                                     -np.sin(math.radians(angle))],
                                    [0, np.sin(math.radians(angle)),
                                     np.cos(math.radians(angle))]]).float()
                segment_data_rot = torch.transpose(
                                     mat@torch.transpose(segment_data, 0, 1),
                                     0, 1)
                data = Data(x=segment_data_rot,
                            edge_index=edge_index,
                            y=y,
                            pos=pos,
                            segment=NAME_TO_INT[seg])
                torch.save(data,
                           os.path.join(savepath,
                                        pt_seg+'_rot'+f'{angle:03d}'+'.pt'))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_name',
        type=str,
        help='name of dataset to be created',
        default='WssToCnc',
        choices=['CoordToCnc', 'WssToCnc', 'WssPlusCnc', 'TsviPlusCnc']
    )
    parser.add_argument(
        '--augment_data',
        type=int,
        help='number of neighbours used for KNN',
        default=0  # used to be 5!
    )
    parser.add_argument(
        '--data_source',
        type=str,
        help='path to raw data',
        # default = '/app/original_data/CFD/ClinicalCFD/MagnitudeClinical'

        default='/Volumes/cardio-project/cardio/CFD/ClinicalCFD/MagnitudeClinical'
        # default='/mnt/NAS/cardio/CFD/ClinicalCFD/MagnitudeClinical'
    )
    parser.add_argument(
        '--labels_source',
        type=str,
        help='path to data label',
        # default = '/app/original_data/CFD/labels/WSSdescriptors_AvgValues.xlsx'
        default='/Volumes/cardio-project/cardio/CFD/labels/WSSdescriptors_AvgValues.xlsx'
        # default='../data/WSSdescriptors_AvgValues.xlsx'
    )
    args = parser.parse_args()

    create_data(
        name='CoordToCnc',
        rot_angles=[],
        k_neigh=args.augment_data,
        path=args.data_source,
        label_path=args.labels_source
    )
# options are ['CoordToCnc', 'WssToCnc', 'WssPlusCnc', 'TsviPlusCnc']
