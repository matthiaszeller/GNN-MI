import logging
import os
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import distance_matrix
from vtk.util.numpy_support import vtk_to_numpy

import setup


def adjacency_matrix_to_edge_list(A):
    """Translates adjacency matrix to edge list"""
    edges = np.array(np.nonzero(A))
    return torch.from_numpy(edges).type(torch.LongTensor)


def create_knn_data(surface, k_neighbours):
    """Implements data augmentation using KNN algorithm"""
    coordinates = vtk_to_numpy(surface.GetPoints().GetData())
    distance_mat = distance_matrix(coordinates, coordinates, 2)
    nearest_neigh = np.argsort(distance_mat, axis=1)[:, :k_neighbours]
    edges = np.array([[i, neighbour]
                     for i in range(len(nearest_neigh))
                     for neighbour in nearest_neigh[i]])
    sorted_edges = np.concatenate([np.max(edges, axis=1, keepdims=True),
                                   np.min(edges, axis=1, keepdims=True)],
                                  axis=1)
    sorted_edges = np.unique(sorted_edges, axis=0)
    reversed_edges = np.concatenate([np.max(sorted_edges, axis=1, keepdims=True), 
                                     np.min(sorted_edges, axis=1, keepdims=True)],
                                    axis=1)
    edges = np.concatenate((sorted_edges, reversed_edges))
    edges = torch.from_numpy(edges.transpose()).type(torch.LongTensor)
    return edges


def attribute_masking(segment_data, alpha):
    """Randomly masks the alpha-portion of the node features."""
    mask = np.random.binomial(1, 1-alpha, segment_data.shape)
    return torch.from_numpy(segment_data.numpy() * mask).type(torch.FloatTensor)


def edge_perturbation(A, alpha):
    """Randomly flips an alpha-portion of the edges"""
    mask = np.random.binomial(1, 1 - alpha, A.shape)
    A_perturbed = A * mask + (1 - A) * (1 - mask)
    return adjacency_matrix_to_edge_list(A_perturbed)


def edge_diffusion(A, alpha):
    """Implements the edge diffusion algorithm for GNN data augmentation"""
    num_nodes = len(A)
    Dinv = np.diag(1 / np.sqrt(np.sum(A, axis=1)))
    A_modified = alpha * np.linalg.inv(np.eye(num_nodes)-(1-alpha)*Dinv@A@Dinv)
    return adjacency_matrix_to_edge_list(A_modified)


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
            noise = torch.normal(mean=torch.ones(data.x.shape)*mean, std=std)
            new_data = data.clone()
            new_data += noise
            file_name = f'{file_path.stem}_NOISE{i}{file_path.suffix}'
            torch.save(new_data, save_path.joinpath(file_name))
            logging.info(f'generated gaussian augmented data file {save_path.joinpath(file_name)}')

