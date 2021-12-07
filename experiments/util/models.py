import torch
from torch.nn import Linear, ELU, ReLU, Dropout, BatchNorm1d, Sequential, Softmax
import torch_geometric.nn as nn
import torch.nn.functional as F

import time


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from util.egnn import EGNN

class EquivBaseline(torch.nn.Module):
    def __init__(self, dataset, n_layers=1):
        super().__init__()

        self.equiv = EGNN(1, 16, 16, n_layers=n_layers, device='cuda')
        #self.equiv2 = EGNN(16, 16, 16, n_layers=n_layers, device='cuda')

        self.gin = self.get_GIN(16, 16, 16)

        self.gmp = nn.global_mean_pool
        self.gap = nn.global_max_pool

        self.classifier = Sequential(
            Dropout(p=0.5),
            Linear(2 * 16 + 1, 16),
            ELU(alpha=0.1),
            Dropout(p=0.5),
            Linear(16, dataset.num_classes),
            Softmax(dim=1)
        )

    @staticmethod
    def get_GIN(in_dim, h_dim, out_dim):
        MLP = Sequential(
            Linear(in_dim, h_dim),
            BatchNorm1d(h_dim),
            ReLU(),
            Linear(h_dim, out_dim)
        )
        return nn.GINConv(MLP, eps=0.0, train_eps=False)

class EquivNoPhys(EquivBaseline):
    def __init__(self, dataset, n_layers=1):
        super().__init__(dataset, n_layers)

    def forward(self, x, edge_index, batch, segment):
        h = torch.zeros(x.shape[0], 1, device = 'cuda')
        #edge_attr = torch.zeros(edge_index.shape[1], 1, device='cuda')
        rep, pos = self.equiv(h, x, edge_index, None)
        #rep = F.elu(rep, alpha=0.1)

        #rep, pos = self.equiv2(rep, pos, edge_index, None)
        #rep = F.elu(rep, alpha=0.1)

        #rep, pos = self.equiv2(rep, pos, edge_index, None)
        #rep = F.elu(rep, alpha=0.1)
        
        x1 = self.gmp(rep, batch)
        x2 = self.gap(rep, batch)

        x = torch.cat([x1, x2], dim=1)
        x = torch.cat((x, segment.view(-1, 1)), dim=1)
        x = self.classifier(x)
        return x


class EquivMixed(EquivBaseline):
    def __init__(self, dataset):
        super().__init__(dataset)

    def forward(self, x, edge_index, batch, segment):
        h = torch.zeros(x.shape[0], 1, device = 'cuda')
        #edge_attr = torch.zeros(edge_index.shape[1], 1, device='cuda')
        rep, pos = self.equiv(h, x, edge_index, None)
        rep = F.elu(rep, alpha=0.1)

        rep = self.gin(rep, edge_index)
        rep = F.elu(rep, alpha=0.1)

        rep = self.gin(rep, edge_index)
        rep = F.elu(rep, alpha=0.1)
        
        #rep, pos = self.equiv2(rep, pos, edge_index, None)
        #rep = F.elu(rep, alpha=0.1)

        #rep, pos = self.equiv2(rep, pos, edge_index, None)
        #rep = F.elu(rep, alpha=0.1)

        x1 = self.gmp(rep, batch)
        x2 = self.gap(rep, batch)

        x = torch.cat([x1, x2], dim=1)
        x = torch.cat((x, segment.view(-1, 1)), dim=1)
        x = self.classifier(x)
        return x

class GnnBaseline(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()

        self.gin_input = self.get_GIN(dataset.num_node_features, 16, 16)
        self.gin16 = self.get_GIN(16, 16, 16)

        # self.pool = nn.EdgePooling(16) # this layer could not run on GPU for Jacob
        self.gmp = nn.global_mean_pool
        self.gap = nn.global_max_pool

        self.classifier = Sequential(
            Dropout(p=0.5),
            Linear(2 * 16 + 1, 16),
            ELU(alpha=0.1),
            Dropout(p=0.5),
            Linear(16, dataset.num_classes),
            Softmax(dim=1)
        )

    @staticmethod
    def get_GIN(in_dim, h_dim, out_dim):
        MLP = Sequential(
            Linear(in_dim, h_dim),
            BatchNorm1d(h_dim),
            ReLU(),
            Linear(h_dim, out_dim)
        )
        return nn.GINConv(MLP, eps=0.0, train_eps=False)


class NoPhysicsGnn(GnnBaseline):
    """Model for WssToCnc and CoordToCnc"""

    def __init__(self, dataset):
        super().__init__(dataset)

    def forward(self, x, edge_index, batch, segment):

        x = self.gin_input(x, edge_index)
        x = F.elu(x, alpha=0.1)
        # x, edge_index, batch, _ = self.pool(x, edge_index, batch)

        x = self.gin16(x, edge_index)
        x = F.elu(x, alpha=0.1)
        # x, edge_index, batch, _ = self.pool(x, edge_index, batch)

        x = self.gin16(x, edge_index)
        x = F.elu(x, alpha=0.1)
        x1 = self.gmp(x, batch)
        x2 = self.gap(x, batch)

        x = torch.cat([x1, x2], dim=1)
        x = torch.cat((x, segment.view(-1, 1)), dim=1)
        x = self.classifier(x)
        return x
