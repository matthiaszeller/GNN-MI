import torch
from torch.nn import Linear, ELU, ReLU, Dropout, BatchNorm1d, Sequential, Softmax
import torch_geometric.nn as nn
import torch.nn.functional as F

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from util.egnn import E_GCL

class GnnBaseline(torch.nn.Module):
    """
    Baseline model, contains common blocks of EquivNoPys and NoPhysicsGnn
    """
    def __init__(self, dataset):
        super().__init__()

        self.gmp = nn.global_mean_pool
        self.gap = nn.global_max_pool

        self.classifier = Sequential(
            Dropout(p=0.5),
            Linear(2 * 16 + 1, 16), # TODO: once one-hot-encoding for segment is done, should be +3 instead of +1
            ELU(alpha=0.1),
            Dropout(p=0.5),
            Linear(16, dataset.num_classes),
            Softmax(dim=1) #TODO: remove this or change loss function to one that does not recompute the softmax
        )

    @staticmethod
    def get_GIN(in_dim, h_dim, out_dim):
        MLP = Sequential(
            Linear(in_dim, h_dim),
            BatchNorm1d(h_dim), # TODO: is this really necessary?
            ReLU(),
            Linear(h_dim, out_dim)
        )
        return nn.GINConv(MLP, eps=0.0, train_eps=False)

class EquivNoPhys(GnnBaseline):
    def __init__(self, dataset, num_gin=2, num_equiv=2):
        super().__init__(dataset)

        self.num_gin=num_gin
        self.num_equiv=num_equiv

        self.equiv = E_GCL(0, 16, 16, tanh=False, residual=False)
        self.equiv1 = E_GCL(16, 16, 16, tanh=False)
        self.equiv2 = E_GCL(16, 16, 16, tanh=False)
        self.equiv3 = E_GCL(16, 16, 16, tanh=False)
        self.equiv4 = E_GCL(16, 16, 16, tanh=False)
        self.equiv5 = E_GCL(16, 16, 16, tanh=False)
        self.equiv6 = E_GCL(16,16,16, tanh=False)

        self.gin = self.get_GIN(16, 16, 16)
        self.gin2 = self.get_GIN(16, 16, 16)
        self.gin3 = self.get_GIN(16, 16, 16)

    def forward(self, x, edge_index, batch, segment):
        if self.num_equiv>0:
            rep, pos, _ = self.equiv(None, edge_index, x, None)
        if self.num_equiv>1:
            rep, pos, _ = self.equiv2(rep, edge_index, pos, None)
        if self.num_equiv>2:
            rep, pos, _ = self.equiv3(rep, edge_index, pos, None)
        if self.num_equiv>3:
            rep, pos, _ = self.equiv4(rep, edge_index, pos, None)
        if self.num_equiv>4:
            rep, pos, _ = self.equiv5(rep, edge_index, pos, None)
        if self.num_equiv>5:
            rep, pos, _ = self.equiv6(rep, edge_index, pos, None)

        if self.num_gin>0:
            rep = self.gin(rep, edge_index)
            rep = F.elu(rep, alpha=0.1)
            if self.num_gin>1:
                rep = self.gin2(rep, edge_index)
                rep = F.elu(rep, alpha=0.1)
                if self.num_gin>2:
                    rep = self.gin3(rep, edge_index)
                    rep = F.elu(rep, alpha=0.1)
            rep, pos = rep, pos

        x1 = self.gmp(rep, batch)
        x2 = self.gap(rep, batch)

        x = torch.cat([x1, x2], dim=1)
        x = torch.cat((x, segment.view(-1, 1)), dim=1) #TODO: Change this into a one-hot-encoding
        x = self.classifier(x)
        return x

class NoPhysicsGnn(GnnBaseline):
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
        x = torch.cat((x, segment.view(-1, 1)), dim=1)   #TODO: Change this into a one-hot-encoding
        x = self.classifier(x)
        return x
