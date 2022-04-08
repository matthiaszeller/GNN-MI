

import logging
from pathlib import Path

import numpy as np
import pygsp
import scipy.sparse
import torch
from graph_coarsening import coarsen
from scipy.spatial import distance_matrix
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected, contains_self_loops, from_scipy_sparse_matrix, to_scipy_sparse_matrix
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


def create_knn_data(coordinates, k_neighbours):
    """Implements data augmentation using KNN algorithm"""
    distance_mat = distance_matrix(coordinates, coordinates, 2)
    # We don't want self loops, but diagonal of distance_max is full of zeros -> put infinity
    np.fill_diagonal(distance_mat, np.inf)

    # Nearest neighbours: use partition instead of sorting (efficient++)
    # this is basically sorting but when we're only interested in the smallest `k_neighbours` values (unsorted)
    nearest_neigh = np.argpartition(distance_mat, k_neighbours, axis=1)
    # The first `k_neighbours` indices along second dimension are the nearest neighbours -> discard the rest
    nearest_neigh = nearest_neigh[:, :k_neighbours]

    # IMPORTANT NOTE: the relationship "belonging to a node's nearest neighbours" is not symmetric,
    # i.e. the following statements
    #   - node i is part of the k-nns of node j
    #   - node j is part of the k-nns of node i
    # do not imply each other
    n_nodes = nearest_neigh.shape[0]
    edge_list = np.array([
        [i, nn]
        for i in range(n_nodes)
        for nn in nearest_neigh[i, :]
    ])
    # Symmetrization: add reversed edges
    edge_list = np.concatenate(
        (edge_list, edge_list[:, [1, 0]])
    )
    # Because of symmetrization, we might have added duplicate edges -> remove them
    edge_list = np.unique(edge_list, axis=0)

    # Convert to torch_geometric edge representation
    edge_index = torch.from_numpy(edge_list.T)
    # Sanity check: undirected graph, i.e. symmetric adjacency matrix
    assert is_undirected(edge_index)
    # Sanity check: no self loop
    assert (not contains_self_loops(edge_index))

    return edge_index


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


# >>> Graph coarsening


def coarsen_graph(data: Data, r=0.7, k=5, method: str = 'variation_neighborhood'):
    if data.x.shape[1] > 0:
        raise NotImplementedError('cannot infer new node attributes')

    G = to_pygsp_graph(data)
    N1, Ne1 = G.N, G.Ne
    _, Gsparsed, *_ = coarsen(G, K=k, r=r, method=method)
    N2, Ne2 = Gsparsed.N, Gsparsed.Ne
    logging.info(f'coarsened graph: nodes {N1} -> {N2}, edges: {Ne1} -> {Ne2}')

    data_coarsened = pygsp_to_data(original=data, G=Gsparsed)
    return data_coarsened


def to_pygsp_graph(data: Data):
    A = to_scipy_sparse_matrix(data.edge_index)
    G = pygsp.graphs.Graph(A, coords=data.coord)
    return G


def pygsp_to_data(original: Data, G: pygsp.graphs.graph.Graph):
    """
    Create new torch sample based on a PyGSP graph itself based on a torch sample.
    G is a modified version of the original, but we first clone the original since it contains data not handled by PyGSP.
    More clearly, we take edge index and coordinates from G, the rest from the original
    """
    edge_index, edge_weight = from_scipy_sparse_matrix(G.A)
    data = original.clone()
    data.edge_index = edge_index
    data.coord = torch.from_numpy(G.coords).type(torch.FloatTensor)
    if data.coord.shape[0] != original.coord.shape[0]:
        data.x = torch.empty(data.coord.shape[0], 0, dtype=torch.float)

    return data


# <<<


if __name__ == '__main__':

    path = setup.get_dataset_path('CoordToCnc')

    for file in path.glob('*.pt'):
        newdata = coarsen_graph(torch.load(file), r=0.7)
        a=0
