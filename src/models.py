import logging
from typing import Union, Dict, OrderedDict

import torch
import torch.nn.functional as F
import torch_geometric.nn as nn
from torch import Tensor
from torch.nn import Linear, ELU, ReLU, Dropout, BatchNorm1d, Sequential, Softmax

import setup
from egnn import E_GCL


def checkpoint_model(path, model_dict: OrderedDict, optimizer_dict: Dict, epoch: int, metrics: Dict):
    logging.info(f'checkpointing model at epoch {epoch} into {path}')
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model_dict,
            'optim_state_dict': optimizer_dict,
            'metrics': metrics
        },
        path
    )


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

    def __init__(self, num_classes: int,
                 num_hidden_dim: int,
                 num_graph_features: int,
                 num_pooling_ops: int,
                 num_hidden_dim_classif: int = None):
        super().__init__()

        self.gmp = nn.global_mean_pool
        self.gap = nn.global_max_pool

        self.equiv = None
        self.gin_layers = []
        self.num_pooling_ops = num_pooling_ops
        if num_hidden_dim_classif is None:
            num_hidden_dim_classif = num_hidden_dim

        self.classifier = Sequential(
            #Dropout(p=0.5),
            Linear(self.num_pooling_ops * num_hidden_dim + num_graph_features, num_hidden_dim_classif),
            ELU(alpha=0.1),
            Dropout(p=0.5),
            Linear(num_hidden_dim_classif, num_classes),
            Softmax(dim=1)
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

    def __init__(self, num_classes: int, num_hidden_dim: int, num_graph_features: int,
                 num_node_features: int = 0, num_equiv: int = 2, num_gin: int = 2):
        """
        @param num_classes: see BaseGNN
        @param num_node_features: number of features per node
        @param num_equiv: number of E_GCL layers
        @param num_gin: number of gin layers
        """
        super().__init__(num_classes=num_classes,
                         num_hidden_dim=num_hidden_dim,
                         num_graph_features=num_graph_features,
                         num_pooling_ops=2)

        self.num_gin = num_gin
        self.num_equiv = num_equiv
        self.num_hidden_dim = num_hidden_dim
        self.num_node_features = num_node_features
        assert self.num_equiv > 0

        self.equiv = nn.Sequential('h, edge_index, coord',
           [
               # First module handled separately because of input dimension
               (
                   E_GCL(self.num_node_features, self.num_hidden_dim, self.num_hidden_dim, tanh=False, residual=False),
                   'h, edge_index, coord -> h, coord, edge_attr')
           ] + [
               # Then add as many modules as necessary
               (
                   E_GCL(self.num_hidden_dim, self.num_hidden_dim, self.num_hidden_dim, tanh=False),
                   'h, edge_index, coord -> h, coord, edge_attr'
               ) for _ in range(num_equiv - 1)
           ]
        )

        if self.num_gin > 0:
            self.gin_layers = nn.Sequential('x, edge_index', [
                (
                    GINActivatedModule(self.num_hidden_dim, self.num_hidden_dim, self.num_hidden_dim),
                    'x, edge_index -> x, edge_index'
                )
                for _ in range(num_gin)
            ])

    def forward(self, h0, coord0, g0, edge_index, batch):
        """See GNNBase for arguments description."""
        # TODO check node_attr vs h
        h, x, edge_attr = self.equiv(h0, edge_index, coord0)

        if self.num_gin > 0:
            h, _ = self.gin_layers(h, edge_index)

        # Graph pooling operations
        x1 = self.gmp(h, batch)
        x2 = self.gap(h, batch)

        x = torch.cat((x1, x2, g0), dim=1)  # TODO: Change this into a one-hot-encoding
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
    Test model saving / restoring. Prerequisites: copy a file from CoordToCnc and WSSToCnc in data.
    """
    from torch_geometric.loader import DataLoader

    path = setup.get_project_root_path().joinpath('data/')
    params_EGNN = {
        'num_classes': 2,
        'num_hidden_dim': 8,
        'num_graph_features': 3,
        'num_gin': 2,
    }

    # --- Pass sample of CoordToCnc dataset through model
    sample = torch.load(path.joinpath('sample_coordtocnc.pt'))
    sample.g_x = sample.g_x.reshape(1, -1)
    sample2 = sample.clone()
    sample2.y = 0
    sample2.g_x[0, 2] = 1.0
    params_EGNN['num_node_features'] = sample.x.shape[1]
    model = EGNN(**params_EGNN)
    model.eval()
    for data in DataLoader([sample, sample2], batch_size=2):
        output_wss = model(data.x, data.coord, data.g_x, data.edge_index, data.batch)

    # --- Pass sample of WssToCnc dataset through model
    sample = torch.load(path.joinpath('sample_wss.pt'))
    params_EGNN['num_node_features'] = sample.x.shape[1]
    model = EGNN(**params_EGNN)
    model.eval()
    for data in DataLoader([sample], batch_size=1):
        output_nowss = model(data.x, data.coord, data.g_x, data.edge_index, data.batch)

    # -- Checkpoint and restore a copy of the model
    checkpoint_model('test_checkpoint.pt', model, torch.optim.Adam(model.parameters()), 100, dict())
    checkpoint = torch.load('test_checkpoint.pt')
    restored = EGNN(**params_EGNN)
    restored.load_state_dict(checkpoint['model_state_dict'])
    restored.eval()
    # Check if this yields the same output
    for data in DataLoader([sample], batch_size=1):
        output_nowss_compare = model(data.x, data.coord, data.g_x, data.edge_index, data.batch)

    torch.testing.assert_allclose(output_nowss, output_nowss_compare)

