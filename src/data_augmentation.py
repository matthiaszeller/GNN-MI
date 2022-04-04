

import logging
from pathlib import Path

import numpy as np
import scipy.sparse
import torch
from scipy.spatial import distance_matrix
from torch_geometric.utils import is_undirected
from vtk.util.numpy_support import vtk_to_numpy

import setup


def transition_matrix(A: scipy.sparse.spmatrix, type='randomwalk'):
    if type != 'randomwalk':
        raise ValueError

    # T_rw = A D^-1, D is diagonal with D_ii = sum of row i of A
    # Compute Dinv
    D = A.sum(axis=1)
    # Remove unused dimension
    D = np.array(D).squeeze()
    Dinv = scipy.sparse.diags(1/D)

    T = A @ Dinv
    return T


def diffusion(T, param, n, type='pagerank'):
    if type == 'pagerank':
        coef_k = lambda k: param * (1 - param) ** k
    else:
        raise ValueError

    # Matrix storing powers of T
    # First power is identity
    m = T.shape[0]
    Tpowk = scipy.sparse.eye(m)
    # S will collect weighted sum of powers
    S = Tpowk * coef_k(0)

    for k in range(1, n+1):
        # Increase power
        Tpowk = Tpowk @ T
        S += coef_k(k) * Tpowk

    return S


def create_knn_data(surface, k_neighbours):
    """Implements data augmentation using KNN algorithm"""
    coordinates = vtk_to_numpy(surface.GetPoints().GetData())
    distance_mat = distance_matrix(coordinates, coordinates, 2)
    # Nearest neighbours: discard first element which is the node itself (having distance zero with itself)
    nearest_neigh = np.argsort(distance_mat, axis=1)[:, 1:k_neighbours+1]
    edges = np.array([[i, neighbour]
                     for i in range(len(nearest_neigh))
                     for neighbour in nearest_neigh[i]])
    sorted_edges = np.concatenate([np.max(edges, axis=1, keepdims=True),
                                   np.min(edges, axis=1, keepdims=True)],
                                  axis=1)
    sorted_edges = np.unique(sorted_edges, axis=0)
    reversed_edges = sorted_edges[:, [1, 0]]
    edges = np.concatenate((sorted_edges, reversed_edges))
    edges = torch.from_numpy(edges.transpose()).type(torch.LongTensor)
    # Sanity check: undirected graph, i.e. symmetric adjacency matrix
    assert is_undirected(edges)

    return edges


def attribute_masking(segment_data, alpha):
    """Randomly masks the alpha-portion of the node features."""
    mask = np.random.binomial(1, 1-alpha, segment_data.shape)
    return torch.from_numpy(segment_data.numpy() * mask).type(torch.FloatTensor)


@setup.arg_logger
def gaussian_noise(path_data, save_path, k=2, mean=0, std=0.1):
    """Augments the data by adding Gaussian noise to the coordinate data.
    Copies original files from path_data into save_path, and additionally create
    k files per original file with gaussian noise added to coordinates.
    save_path must not exist.
    """
    path_data, save_path = Path(path_data), Path(save_path)
    if save_path.exists():
        raise ValueError('save path must not exist and will be created')

    save_path.mkdir()
    for file_path in path_data.glob('*.pt'):
        data = torch.load(file_path)
        # Copy original file into destination
        file_name = file_path.name
        torch.save(data, save_path.joinpath(file_name))
        logging.info(f'copied {str(file_path)} to {save_path.joinpath(file_name)}')
        # Augment
        for i in range(1, k+1):
            noise = torch.normal(mean=torch.ones(data.coord.shape)*mean, std=std)
            new_data = data.clone()
            new_data.coord += noise
            file_name = f'{file_path.stem}_NOISE{i}{file_path.suffix}'
            torch.save(new_data, save_path.joinpath(file_name))
            logging.info(f'generated gaussian augmented data file {save_path.joinpath(file_name)}')


if __name__ == '__main__':
    from torch_geometric.utils import to_scipy_sparse_matrix

    path = setup.get_dataset_path('CoordToCnc_KNN5')

    for file in path.glob('*.pt'):
        data = torch.load(file)
        break

    A = to_scipy_sparse_matrix(data.edge_index)
    A = (A + A.T) / 2
    T = transition_matrix(A)
