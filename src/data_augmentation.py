

import logging
from math import factorial
from pathlib import Path
from typing import Callable, Union

import numpy as np
import pygsp
import scipy.sparse
import torch
from graph_coarsening import coarsen
from scipy.spatial import distance_matrix
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected, contains_self_loops, from_scipy_sparse_matrix, to_scipy_sparse_matrix
from vtk.util.numpy_support import vtk_to_numpy

from multiprocessing import cpu_count, Pool

import setup
from perimeter import compute_perimeters, parse_perimeter_data


def create_dataset_with_perimeter(input_dset: Union[str, Path],
                                  perimeter_dset: Union[str, Path], output_dset: Union[str, Path]):
    input_dset, output_dset = Path(input_dset), Path(output_dset)
    perimeter_dset = Path(perimeter_dset)
    if not output_dset.exists():
        output_dset.mkdir()

    logging.info('creating dataset with perimeter')
    files_dset = list(input_dset.glob('*.pt'))
    files_perim = list(perimeter_dset.glob('*.json'))
    patients = set(file.stem for file in files_perim)
    patients = patients.intersection(file.stem for file in files_perim)
    if len(patients) != len(files_dset) or len(patients) != len(files_perim):
        raise ValueError('intersection of patients do not match')

    for file in input_dset.glob('*.pt'):
        logging.info(f'parsing torch data and perimeter data for {file.stem}')
        # Load torch graph data
        sample = torch.load(file)
        # Parse perimeter data
        perim_data = parse_perimeter_data(perimeter_dset.joinpath(file.stem + '.json'))
        # Reshape
        perim_data = perim_data.reshape(-1, 1)
        # Concatenate node features along 2nd dimension
        sample.x = torch.concat((sample.x, perim_data), dim=1)

        output_file = output_dset.joinpath(file)
        logging.info(f'writing output in {output_file}')
        torch.save(sample, output_file)


def compute_dataset_perimeter(input_path: Union[str, Path], output_path: Union[str, Path]):
    """Compute perimeter at each node of all samples of a dataset in parallel."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir()

    data = {
        file: torch.load(file) for file in input_path.glob('*.pt')
    }

    args_iterator = (
        (sample, output_path.joinpath(file.stem + '.json'))
        for file, sample in data.items()
    )
    logging.info('starting perimeter computation')
    # pool = Pool(cpu_count() - 1)
    # pool.starmap_async(compute_perimeters, args_iterator)
    # pool.close()
    # pool.join()
    for sample, output_file in args_iterator:
        compute_perimeters(sample, output_file)
    logging.info('end of perimeter computation')


def transition_matrix(A: scipy.sparse.spmatrix, type='randomwalk') -> scipy.sparse.spmatrix:
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


def diffusion_matrix(T: scipy.sparse.spmatrix, param: float, n: int, kernel='pagerank') -> scipy.sparse.spmatrix:
    if kernel == 'pagerank':
        coef_k = lambda k: param * (1 - param) ** k
    elif kernel == 'heat':
        coef_k = lambda k: np.exp(-param) * param**k / factorial(k)
    else:
        raise ValueError

    # Matrix storing powers of T
    # First power is identity
    m = T.shape[0]
    Tpowk = scipy.sparse.eye(m, format='csr')
    # S will collect weighted sum of powers
    S = Tpowk * coef_k(0)
    # Convert T to csr format for efficient matrix-matrix mult
    T = T.tocsr()

    for k in range(1, n+1):
        # Increase power
        Tpowk = Tpowk @ T
        S += coef_k(k) * Tpowk

    return S


@setup.arg_logger
def create_diffusion_data(data: Data, threshold, param=0.1,
                          kernel='heat', n=15, symmetrize: bool = True, self_loops: bool = False) -> Data:
    out = data.clone()
    # Adjacency matrix
    A = to_scipy_sparse_matrix(out.edge_index)
    # Get transition matrix
    T = transition_matrix(A)
    # Compute diffusion matrix
    S = diffusion_matrix(T, param=param, n=n, kernel=kernel)
    # Symmetrization
    if symmetrize:
        S = (S + S.T) / 2
    # Efficient thresholding
    S.data = np.where(S.data < threshold, 0.0, S.data)
    S.eliminate_zeros()
    # Remove self loops
    if self_loops is False:
        n = S.shape[0]
        S[range(n), range(n)] = 0.0
        S.eliminate_zeros()

    # Build new sample
    out.edge_index, out.edge_weight = from_scipy_sparse_matrix(S)
    # Float64 -> float32
    out.edge_weight = out.edge_weight.float()

    return out


def create_random_diffusion(data: Data, n_connections: int = 3, diff_threshod: float = 5e-4, heat_param: float = 8.0):
    """
    Graph diffusion with randomization to add long-range connections.
    First get a diffused graph with create_diffusion_data, with parameters that favor long-range connections.
    Then for each node, randomly sample n_connections with prbability proportional to the weights obtained by diffusion.
    """
    def get_neigh(i, nnz_r, nnz_c):
        mask = nnz_r == i
        return nnz_c[mask]

    out = data.clone()
    # Diffusion data with "long range" connections
    diff = create_diffusion_data(data, threshold=diff_threshod, param=heat_param, n=20, kernel='heat')
    # Convenient data format
    A = to_scipy_sparse_matrix(diff.edge_index)
    nnz_r, nnz_c = A.nonzero()
    # Randomly add new edges
    new_connections = []
    for i in range(A.shape[0]):
        # Get neighbours of node i in diffusion data
        neigh = get_neigh(i, nnz_r, nnz_c)
        # Sampling probabilities
        w = np.array(diff.edge_weight[neigh])
        w /= w.sum()
        chosen = np.random.choice(neigh, size=n_connections, p=w, replace=False)
        new_connections.extend([
            (i, u) for u in chosen
        ])
    # Format new edges
    new_edges = torch.tensor(new_connections)
    new_edges = torch.concat((new_edges, new_edges[:, [1, 0]])).T
    # Add new edges
    out.edge_index = torch.concat((out.edge_index, new_edges), dim=1)

    return out


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


@setup.arg_logger
def augment_dataset(
        fun_process_sample: Callable,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        filename_suffix: str,
        copy_original: bool = True,
        n_sample: int = 1,
        *args,
        **kwargs
):
    """
    Generic function to augment a dataset.

    :param fun_process_sample: data augmentation function, takes as input a torch geometric data sample, returns the
    augmented version of the sample
    :param input_path: dataset input path, must exist
    :param output_path: augmented dataset path, must not exist
    :param filename_suffix: suffix of augmented-data files
    :param copy_original: whether to include original samples in the new dataset
    :param n_sample: how many times to call fun_process_sample, use > 1 only if randomized
    :param args, kwargs: passed to fun_process_sample
    """
    input_path, output_path = Path(input_path), Path(output_path)
    if not input_path.exists():
        raise ValueError
    if output_path.exists():
        raise ValueError
    output_path.mkdir()

    number_suffix = (lambda i: f'_{i}') if n_sample > 1 else (lambda i: '')

    for file in input_path.glob('*.pt'):
        original = torch.load(file)
        if copy_original:
            outfile = output_path.joinpath(file.name)
            logging.info(f'copy original file in {outfile}\n{original}')
            torch.save(original, outfile)

        for i in range(n_sample):
            # Get augmented sample
            augmented = fun_process_sample(original, *args, **kwargs)
            nsuffix = number_suffix(i)
            outfile = output_path.joinpath(f'{file.stem}_{filename_suffix}{nsuffix}{file.suffix}')
            logging.info(f'saving augmented data {outfile}\n{augmented}')
            torch.save(augmented, outfile)

# <<<


if __name__ == '__main__':

    path = setup.get_dataset_path('CoordToCnc')

    augment_dataset(
        create_diffusion_data,
        path,
        path.parent.joinpath('CoordToCnc_diffusion'),
        'diffusion',
        kernel='heat',
        threshold=0.1,
        param=1.0
    )
    #
    # for file in path.glob('*.pt'):
    #     S = create_diffusion_data(torch.load(file), threshold=1e-2)
    #     # Next line is for breakpoints
    #     a=1

    # for file in path.glob('*.pt'):
    #     newdata = coarsen_graph(torch.load(file), r=0.7)
    #     a=0
