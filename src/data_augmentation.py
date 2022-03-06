import os
import numpy as np
import torch
from scipy.spatial import distance_matrix
from vtk.util.numpy_support import vtk_to_numpy


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


def gaussian_noise(path_data="data/CoordToCnc",
                   save_path="data/CoordToCncGaussian2", k=2,
                   mean=0, std=0.1):
    """Augments the data by adding Gaussian noise to the coordinate data.
    First fetches the data as torch tensors from path_data, augments it k
    times, and then saves it into save_path. For recommended usage: save_path
    is empty, and path_data contains only original Coord data."""
    for seg in os.listdir(path_data):
        data = torch.load(path_data+"/"+seg)
        torch.save(data, save_path+"/"+seg)
        for i in range(1, k+1):
            noise = torch.normal(mean=torch.ones(data.x.shape)*mean, std=0.1)
            data.x = data.x + noise
            torch.save(data, save_path+"/"+seg[:-3]+"NOISE"+str(i)+seg[-3:])
