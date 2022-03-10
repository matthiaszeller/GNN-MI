from typing import Union

import torch
import torch.nn.functional as F
import torch_geometric.nn as nn
from torch import Tensor
from torch.nn import Linear, ELU, ReLU, Dropout, BatchNorm1d, Sequential, Softmax

import setup
from egnn import E_GCL


class GINActivatedModule(nn.GINConv):
    def __init__(self, in_dim, h_dim, out_dim):
        MLP = Sequential(
            Linear(in_dim, h_dim),
            BatchNorm1d(h_dim),  # TODO: is this really necessary?
            ReLU(),
            Linear(h_dim, out_dim)
        )
        super(GINActivatedModule, self).__init__(MLP, eps=0.0, train_eps=False)

    def forward(self, x, edge_index, size=None):
        x = super(GINActivatedModule, self).forward(x, edge_index)
        return F.elu(x, alpha=0.1), edge_index


class GNNBase(torch.nn.Module):
    """
    Baseline model, contains common blocks of EquivNoPys and NoPhysicsGnn
    """

    def __init__(self, num_classes: int):
        super().__init__()

        self.gmp = nn.global_mean_pool
        self.gap = nn.global_max_pool

        self.equiv = None
        self.gin_layers = []

        self.classifier = Sequential(
            Dropout(p=0.5),
            Linear(2 * 16 + 1, 16),  # TODO: once one-hot-encoding for segment is done, should be +3 instead of +1
            ELU(alpha=0.1),
            Dropout(p=0.5),
            Linear(16, num_classes),
            # TODO: For the culprit/nonculprit classification, we could simply predict one probability instead of two, saving parameters.
            Softmax(dim=1)  # TODO: remove this or change loss function to one that does not recompute the softmax
        )

    def forward(self, h0, coord0, g0, edge_index, batch):
        """
        @param h0: node embeddings
        @param coord0: node coordinates
        @param g0: graph embedding
        @param edge_index: graph connectivity
        @param batch: 1D tensor where batch[i] ∈ {1, ..., batch_size}, see torch_geometric.loader.DataLoader
        """
        pass


class EGNN(GNNBase):
    """
    Equivariant Graph Neural Network composed of:
        Sequential(<E_GCL layers>) -> Sequential(<GIN layer> -> elu -> ...)
    The output of GIN layers are processed by the classifier of the parent class GNNBase.
    """

    def __init__(self, num_classes: int, num_node_features: int = 0, num_equiv: int = 2, num_gin: int = 2):
        """
        @param num_classes: see BaseGNN
        @param num_node_features: number of features per node
        @param num_equiv: number of E_GCL layers
        @param num_gin: number of gin layers
        """
        super().__init__(num_classes)

        self.num_gin = num_gin
        self.num_equiv = num_equiv
        assert self.num_equiv > 0

        self.equiv = nn.Sequential('h, edge_index, coord',
           [
               # First module handled separately because of input dimension
               (
                   E_GCL(num_node_features, 16, 16, tanh=False, residual=False),
                   'h, edge_index, coord -> h, coord, edge_attr')
           ] + [
               # Then add as many modules as necessary
               (
                   E_GCL(16, 16, 16, tanh=False),
                   'h, edge_index, coord -> h, coord, edge_attr'
               ) for _ in range(num_equiv - 1)
           ]
        )
        # self.equiv = E_GCL(0, 16, 16, tanh=False, residual=False)
        # self.equiv1 = E_GCL(16, 16, 16, tanh=False)
        # self.equiv2 = E_GCL(16, 16, 16, tanh=False)
        # self.equiv3 = E_GCL(16, 16, 16, tanh=False)
        # self.equiv4 = E_GCL(16, 16, 16, tanh=False)
        # self.equiv5 = E_GCL(16, 16, 16, tanh=False)
        # self.equiv6 = E_GCL(16, 16, 16, tanh=False)

        if self.num_gin > 0:
            self.gin_layers = nn.Sequential('x, edge_index', [
                (GINActivatedModule(16, 16, 16), 'x, edge_index -> x, edge_index')
                for _ in range(num_gin)
            ])
        # self.gin = self.get_GIN(16, 16, 16)
        # self.gin2 = self.get_GIN(16, 16, 16)
        # self.gin3 = self.get_GIN(16, 16, 16)

    def forward(self, h0, coord0, g0, edge_index, batch):
        """See GNNBase for arguments description."""
        # if self.num_equiv > 0:
        #     rep, coord, _ = self.equiv(None, edge_index, x, None)
        # if self.num_equiv > 1:
        #     rep, coord, _ = self.equiv2(rep, edge_index, coord, None)
        # if self.num_equiv > 2:
        #     rep, coord, _ = self.equiv3(rep, edge_index, coord, None)
        # if self.num_equiv > 3:
        #     rep, coord, _ = self.equiv4(rep, edge_index, coord, None)
        # if self.num_equiv > 4:
        #     rep, coord, _ = self.equiv5(rep, edge_index, coord, None)
        # if self.num_equiv > 5:
        #     rep, coord, _ = self.equiv6(rep, edge_index, coord, None)
        # TODO check node_attr vs h
        h, x, edge_attr = self.equiv(h0, edge_index, coord0)

        if self.num_gin > 0:
            h, _ = self.gin_layers(h, edge_index)
        # for gin_layer in self.gin_layers:
        #     h = gin_layer(h, edge_index)
        #     h = F.elu(h, alpha=0.1)
        # if self.num_gin>0:
        #     rep = self.gin(rep, edge_index)
        #     rep = F.elu(rep, alpha=0.1)
        #     if self.num_gin>1:
        #         rep = self.gin2(rep, edge_index)
        #         rep = F.elu(rep, alpha=0.1)
        #         if self.num_gin>2:
        #             rep = self.gin3(rep, edge_index)
        #             rep = F.elu(rep, alpha=0.1)
        #     rep, coord = rep, coord

        # Graph pooling operations
        x1 = self.gmp(h, batch)
        x2 = self.gap(h, batch)

        x = torch.cat((x1, x2, g0.view(-1, 1)), dim=1)  # TODO: Change this into a one-hot-encoding
        x = self.classifier(x)
        return x


class NoPhysicsGnn(GNNBase):
    """
    Model for WssToCnc and CoordToCnc.
    CoordToCnc is also called GIN in Jacob's report.
    """

    def __init__(self, dataset):
        super().__init__(dataset)
        self.gin_input = self.get_GIN(3, 16, 16)
        self.gin16 = self.get_GIN(16, 16, 16)

    def forward(self, x, edge_index, batch, segment):
        x = self.gin_input(x, edge_index)
        x = F.elu(x, alpha=0.1)

        x = self.gin16(x, edge_index)
        x = F.elu(x, alpha=0.1)

        x = self.gin16(x, edge_index)
        x = F.elu(x, alpha=0.1)
        x1 = self.gmp(x, batch)
        x2 = self.gap(x, batch)

        x = torch.cat([x1, x2], dim=1)
        x = torch.cat((x, segment.view(-1, 1)), dim=1)  # TODO: Change this into a one-hot-encoding
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    """
    Do a few tests for debugging / code inspection.
    """
    from torch_geometric.loader import DataLoader

    path = setup.get_project_root_path().joinpath('data/')

    sample = torch.load(path.joinpath('sample_WSS.pt'))
    num_node_features = sample.x.shape[1]

    model = EGNN(num_classes=2, num_node_features=num_node_features, num_equiv=2, num_gin=2)
    for data in DataLoader([sample], batch_size=1):
        output_wss = model(data.x,
                           data.coord,
                           data.segment,
                           data.edge_index,
                           data.batch)

    sample = torch.load(path.joinpath('sample_Coord.pt'))
    model = EGNN(num_classes=2, num_node_features=0, num_equiv=2, num_gin=2)
    for data in DataLoader([sample], batch_size=1):
        output_nowss = model(data.x,
                             data.coord,
                             data.segment,
                             data.edge_index,
                             data.batch)
