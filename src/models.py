import logging
from typing import Union, Dict, OrderedDict

import torch
import torch.nn.functional as F
import torch_geometric.nn as nn
from torch import Tensor
from torch.nn import Linear, ELU, ReLU, Dropout, BatchNorm1d, Sequential, Softmax
from torch_scatter import scatter_max, scatter_mean

import setup
from egnn import E_GCL
from utils import get_model_num_params


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


class GraphPooler(torch.nn.Module):
    def __init__(self, input_dim: int, mean_pooler: bool = True, max_pooler: bool = True,
                 min_pooler: bool = False, norm_pooler: bool = False):
        super(GraphPooler, self).__init__()
        self.pooling_ops = []

        self.output_dim = 0
        if mean_pooler:
            self.pooling_ops.append(nn.global_mean_pool)
            self.output_dim += input_dim
        if max_pooler:
            self.pooling_ops.append(nn.global_max_pool)
            self.output_dim += input_dim
        if min_pooler:
            self.pooling_ops.append(lambda x, b: nn.global_max_pool(-x, b))
            self.output_dim += input_dim
        if norm_pooler:
            def norm_pool(x: torch.Tensor, batch: torch.Tensor):
                n = self.compute_norms(x)
                out = torch.tensor([
                    scatter_mean(n, batch),
                    scatter_max(n, batch)
                ])
                return out

            self.pooling_ops.append(norm_pool)
            self.output_dim += 2

    def compute_norms(self, x: torch.Tensor):
        return x.norm(dim=1)

    def forward(self, x: torch.Tensor, batch: torch.Tensor):
        out = torch.concat([
            operation(x, batch) for operation in self.pooling_ops
        ], dim=-1)

        return out


class MasterNode(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_pooling_ops: int = 2):
        super(MasterNode, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_pooling_ops = n_pooling_ops

        if self.n_pooling_ops != 2:
            raise NotImplementedError

        self.nodes_to_master = Linear(self.input_dim, self.hidden_dim)

        #self.weights = torch.nn.Conv1d(in_channels=self.hidden_dim, out_channels=1, kernel_size=(self.n_pooling_ops,))

        self.weights = torch.nn.Conv1d(in_channels=self.n_pooling_ops, out_channels=1, kernel_size=(1,))

        #self.weights = torch.nn.Conv2d(in_channels=self.hidden_dim, out_channels=1, kernel_size=(self.n_pooling_ops,))

        #self.weights = Linear(self.n_pooling_ops, self.output_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor):
        # h is              N_nodes x D_hidden

        # Flow information from nodes to master node
        x = ELU()(self.nodes_to_master(h))

        # x is              N_nodes x D_hidden

        # Pool: each pooled vector is considered as a "channel" for convolution layers
        x = torch.stack((
            nn.global_max_pool(x, batch),
            nn.global_mean_pool(x, batch)
        ), dim=1)

        # x is              N_batch x N_pool x D_hidden, number of pooling ops N_pool = 2 (mean, max)

        # Weight the contribution of the pooled signal to the D_hidden components of h
        # TODO weights kernel size should be D_hidden
        h_perturbation = ELU()(self.weights(x))
        h_perturbation = h_perturbation.squeeze(dim=1)

        # There's a different perturbation for each graph, but the perturbation
        # is the same for all nodes within a given graph
        # h_pert is         N_batch x D_hidden

        # Flow information from master node to nodes
        # First, "tile" the perturbation network
        h_perturbation = torch.index_select(h_perturbation, dim=0, index=batch)
        h = h + h_perturbation

        return h, edge_index, batch


class Mastered_EGCL(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Mastered_EGCL, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.EGCL = E_GCL(input_nf=input_dim, output_nf=output_dim, hidden_nf=hidden_dim)
        self.batch_norm = BatchNorm1d(num_features=output_dim)
        self.master_node = MasterNode(input_dim=output_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    def forward(self, h, edge_index, coord, batch):
        h, edge_index, coord = self.EGCL(h, edge_index, coord)

        h = ELU()(h)

        h = self.batch_norm(h)

        # h, edge_index, batch -> only h changes
        h, _, _ = self.master_node(h, edge_index, batch)

        return h, edge_index, coord, batch


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

    def __init__(self, num_classes, num_hidden_dim, num_graph_features,
                 num_node_features, num_equiv, num_gin):
        super().__init__()

        self.num_classes = num_classes
        self.num_graph_features = num_graph_features
        self.num_hidden_dim = num_hidden_dim
        self.num_node_features = num_node_features
        self.num_gin = num_gin
        self.num_equiv = num_equiv

        # self.gmp = nn.global_mean_pool
        # self.gap = nn.global_max_pool
        #
        # self.equiv = None
        # self.gin_layers = []
        # self.num_pooling_ops = num_pooling_ops
        # if num_hidden_dim_classif is None:
        #     num_hidden_dim_classif = num_hidden_dim
        #
        # self.classifier = Sequential(
        #     #Dropout(p=0.5),
        #     Linear(self.num_pooling_ops * num_hidden_dim + num_graph_features, num_hidden_dim_classif),
        #     ELU(alpha=0.1),
        #     BatchNorm1d(num_hidden_dim_classif),
        #     #Dropout(p=0.5),
        #     Linear(num_hidden_dim_classif, num_classes),
        #     Softmax(dim=1)
        # )

    def forward(self, h0, coord0, g0, edge_index, batch):
        """
        @param h0: node embeddings
        @param coord0: node coordinates
        @param g0: graph embedding
        @param edge_index: graph connectivity
        @param batch: 1D tensor where batch[i] ∈ {1, ..., batch_size}, see torch_geometric.loader.DataLoader
        """
        pass


class Classifier(torch.nn.Module):
    def __init__(self, num_input_dim: int, num_hidden_dim: int, num_classes: int):
        super(Classifier, self).__init__()

        self.num_input_dim = num_input_dim
        self.num_hidden_dim = num_hidden_dim
        self.num_classes = num_classes

        self.classifier = Sequential(
            Linear(self.num_input_dim, self.num_hidden_dim),
            ELU(alpha=0.1),
            BatchNorm1d(self.num_hidden_dim),
            Linear(self.num_hidden_dim, self.num_classes),
            Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class Regressor(torch.nn.Module):
    def __init__(self, num_input_dim: int, num_hidden_dim: int):
        super(Regressor, self).__init__()
        self.num_input_dim = num_input_dim
        self.num_hidden_dim = num_hidden_dim
        self.regressor = Sequential(
            Linear(self.num_input_dim, self.num_hidden_dim),
            ELU(alpha=0.1),
            BatchNorm1d(self.num_hidden_dim),
            Linear(self.num_hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(x)


class EGNN(GNNBase):
    """
    Equivariant Graph Neural Network composed of:
        Sequential(<E_GCL layers>) -> Sequential(<GIN layer> -> elu -> ...)
    The output of GIN layers are processed by the classifier of the parent class GNNBase.
    """

    def __init__(self, num_classes: int, num_hidden_dim: int, num_graph_features: int,
                 num_node_features: int = 0, num_equiv: int = 2, num_gin: int = 2,
                 auxiliary_task: bool = False, auxiliary_nodewise: bool = False,
                 min_pooler: bool = False, norm_pooler: bool = False):
        """
        @param num_classes: see BaseGNN
        @param num_node_features: number of features per node
        @param num_equiv: number of E_GCL layers
        @param num_gin: number of gin layers
        """
        super(EGNN, self).__init__(num_classes, num_hidden_dim, num_graph_features,
                                   num_node_features, num_equiv, num_gin)

        assert self.num_equiv > 0
        self.auxiliary_task = auxiliary_task
        self.auxiliary_nodewise = auxiliary_nodewise

        self.equiv = nn.Sequential('h, edge_index, coord',
                                   [
                                       # First module handled separately because of input dimension
                                       (
                                           E_GCL(self.num_node_features, self.num_hidden_dim, self.num_hidden_dim,
                                                 tanh=False, residual=False),
                                           'h, edge_index, coord -> h, edge_index, coord')
                                   ] + [
                                       # Then add as many modules as necessary
                                       (
                                           E_GCL(self.num_hidden_dim, self.num_hidden_dim, self.num_hidden_dim,
                                                 tanh=False),
                                           'h, edge_index, coord -> h, edge_index, coord'
                                       ) for _ in range(num_equiv - 1)
                                   ]
                                   )

        if self.num_gin > 0:
            self.gin_layers = nn.Sequential('x, edge_index', [
                (
                    # +1 because we provide norm of delta coordinates
                    GINActivatedModule(self.num_hidden_dim + 1, self.num_hidden_dim, self.num_hidden_dim),
                    'x, edge_index -> x, edge_index'
                )
                for _ in range(num_gin)
            ])

        self.pooler = GraphPooler(self.num_hidden_dim, min_pooler=min_pooler, norm_pooler=norm_pooler)
        self.classifier = Classifier(self.pooler.output_dim + self.num_graph_features,
                                     self.num_hidden_dim, self.num_classes)
        if self.auxiliary_task:
            if self.auxiliary_nodewise:
                self.regressor = Regressor(self.num_hidden_dim, 1)
            else:
                self.regressor = Regressor(self.pooler.output_dim + self.num_graph_features, self.num_hidden_dim)

    def forward(self, h0, coord0, g0, edge_index, batch):
        """See GNNBase for arguments description."""
        # Coords will be modified in place by EGCLs, clone them for later usage
        coord_original = coord0.clone()

        # Equivariant layers
        h, _, coord = self.equiv(h0, edge_index, coord0)

        # GIN layers
        if self.num_gin > 0:
            # Create feature: norm of coordinate changes
            delta_coord = (coord - coord_original).norm(dim=-1).unsqueeze(dim=-1)
            h_gin = torch.concat((h, delta_coord), dim=-1)
            h, _ = self.gin_layers(h_gin, edge_index)

        if self.auxiliary_task and self.auxiliary_nodewise:
            x_node_aux = self.regressor(h)

        # Graph pooling operations
        p = self.pooler(h, batch)

        # Build classifier input: concatenated pooled vectors + graph features
        x = torch.cat((p, g0), dim=1)

        if self.auxiliary_task and not self.auxiliary_nodewise:
            x = self.classifier(x), self.regressor(x)
        elif self.auxiliary_task and self.auxiliary_nodewise:
            x = self.classifier(x), x_node_aux
        else:
            x = self.classifier(x)
        return x


class EGNNMastered(GNNBase):
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
        # TODO: need to refactor this class to match new structure
        super().__init__(num_classes=num_classes,
                         num_hidden_dim=num_hidden_dim,
                         num_graph_features=num_graph_features,
                         num_pooling_ops=2)

        self.num_gin = num_gin
        self.num_equiv = num_equiv
        self.num_node_features = num_node_features
        self.num_hidden_dim = num_hidden_dim
        assert self.num_equiv > 0

        self.equiv = []
        self.equiv.append(
            (
                Mastered_EGCL(input_dim=self.num_node_features, hidden_dim=self.num_hidden_dim, output_dim=self.num_hidden_dim),
                'h, edge_index, coord, batch -> h, edge_index, coord, batch'
            )
        )
        for _ in range(self.num_equiv - 1):
            self.equiv.append(
                (
                    Mastered_EGCL(input_dim=self.num_hidden_dim, hidden_dim=self.num_hidden_dim,
                                  output_dim=self.num_hidden_dim),
                    'h, edge_index, coord, batch -> h, edge_index, coord, batch'
                )
            )

        self.equiv = nn.Sequential(
            'h, edge_index, coord, batch',
            self.equiv
        )

        if self.num_gin > 0:
            self.gin_layers = nn.Sequential('x, edge_index', [
                (
                    GINActivatedModule(self.num_hidden_dim, self.num_hidden_dim, self.num_hidden_dim),
                    'x, edge_index -> x, edge_index'
                )
                for _ in range(num_gin)
            ])

        self.pooler = GraphPooler(self.num_hidden_dim)
        self.classifier = Sequential(
            # Dropout(p=0.5),
            Linear(self.pooler.output_dim + num_graph_features, self.num_hidden_dim),
            ELU(alpha=0.1),
            BatchNorm1d(self.num_hidden_dim),
            Dropout(p=0.1),
            Linear(self.num_hidden_dim, num_classes),
            Softmax(dim=1)
        )

    def forward(self, h0, coord0, g0, edge_index, batch):
        """See GNNBase for arguments description."""
        h, _, coord, batch = self.equiv(h0, edge_index, coord0, batch)

        if self.num_gin > 0:
            h, _ = self.gin_layers(h, edge_index)

        # Graph pooling operations
        #x1 = self.gmp(h, batch)
        #x2 = self.gap(h, batch)
        pooled = self.pooler(h, batch)

        x = torch.cat((pooled, g0), dim=1)
        x = self.classifier(x)
        return x


class GIN_GNN(GNNBase):
    def __init__(self, num_classes: int, num_hidden_dim: int, num_graph_features: int,
                 num_node_features: int = 0, num_gin: int = 2, use_coords: bool = True):
        """
        @param num_classes: see BaseGNN
        @param num_node_features: number of features per node
        @param num_equiv: number of E_GCL layers
        @param num_gin: number of gin layers
        """
        super().__init__(num_classes=num_classes, num_hidden_dim=num_hidden_dim,
                         num_graph_features=num_graph_features, num_node_features=num_node_features,
                         num_equiv=0, num_gin=num_gin)

        self.num_gin = num_gin
        self.num_hidden_dim = num_hidden_dim
        self.num_node_features = num_node_features
        self.use_coords = use_coords

        input_additional_dim = 3 if self.use_coords else 0
        self.gin_layers = nn.Sequential('x, edge_index',
            [
                (
                    # +3 because of x-y-z coordinates
                    GINActivatedModule(self.num_node_features + input_additional_dim, self.num_hidden_dim, self.num_hidden_dim),
                    'x, edge_index -> x, edge_index'
                )
            ] + [
                (
                    GINActivatedModule(self.num_hidden_dim, self.num_hidden_dim, self.num_hidden_dim),
                    'x, edge_index -> x, edge_index'
                )
                for _ in range(num_gin)
            ]
        )

        self.pooler = GraphPooler(self.num_hidden_dim)
        self.classifier = Classifier(self.pooler.output_dim + self.num_graph_features,
                                     self.num_hidden_dim, self.num_classes)

    def forward(self, h0, coord0, g0, edge_index, batch):
        """See GNNBase for arguments description."""
        if self.use_coords:
            gin_input = torch.cat((h0, coord0), dim=1)
        else:
            gin_input = h0

        h, _ = self.gin_layers(gin_input, edge_index)

        # Graph pooling operations
        pooled = self.pooler(h, batch)

        x = torch.cat((pooled, g0), dim=1)  # TODO: Change this into a one-hot-encoding
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
        'num_hidden_dim': 16,
        'num_graph_features': 3,
        'num_gin': 2,
    }
    params_GIN = {
        'num_classes': 2,
        'num_hidden_dim': 8,
        'num_graph_features': 3,
        'num_gin': 2,
    }

    # --- Pass sample of CoordToCnc dataset through model
    sample = torch.load(path.joinpath('sample_coarse.pt'))
    sample2 = sample.clone()
    sample2.coord += torch.randn_like(sample2.coord) * 0.001
    sample2.y = 0
    sample2.g_x[0, 2] = 1.0
    sample3 = sample.clone()
    sample3.coord += torch.randn_like(sample2.coord) * 0.001
    sample3.g_x[0, 2] = -1.0
    params_EGNN['num_node_features'] = sample.x.shape[1]
    model = EGNN(**params_EGNN)
    #model = GIN_GNN(**params_EGNN)
    #model = EGNNMastered(**params_EGNN)
    print(model)
    print(get_model_num_params(model), 'parameters')
    model.eval()
    for data in DataLoader([sample, sample2, sample3], batch_size=10):
        output_wss = model(data.x, data.coord, data.g_x, data.edge_index, data.batch)


    # # --- Pass sample of WssToCnc dataset through model
    # sample = torch.load(path.joinpath('sample_wss.pt'))
    # params_EGNN['num_node_features'] = sample.x.shape[1]
    # #model = EGNN(**params_EGNN)
    # model = GIN_GNN(**params_EGNN)
    # model.eval()
    # for data in DataLoader([sample], batch_size=1):
    #     output_nowss = model(data.x, data.coord, data.g_x, data.edge_index, data.batch)
    #
    # # -- Checkpoint and restore a copy of the model
    # checkpoint_model('test_checkpoint.pt', model.state_dict(),
    #                  torch.optim.Adam(model.parameters()).state_dict(), 1, dict())
    # checkpoint = torch.load('test_checkpoint.pt')
    # #restored = EGNN(**params_EGNN)
    # restored = GIN_GNN(**params_EGNN)
    # restored.load_state_dict(checkpoint['model_state_dict'])
    # restored.eval()
    # # Check if this yields the same output
    # for data in DataLoader([sample], batch_size=1):
    #     output_nowss_compare = model(data.x, data.coord, data.g_x, data.edge_index, data.batch)
    #
    # torch.testing.assert_allclose(output_nowss, output_nowss_compare)

