import torch
from torch.nn import Linear, ELU, ReLU, Dropout, BatchNorm1d, Sequential, Softmax
import torch_geometric.nn as nn
import torch.nn.functional as F

import time


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from util.egnn import E_GCL

class EquivBaseline(torch.nn.Module):
    def __init__(self, dataset, num_gin=2, num_equiv=2):
        super().__init__()
        
        self.num_gin=num_gin
        self.num_equiv=num_equiv

        self.gin = self.get_GIN(16, 16, 16)
        self.gin2 = self.get_GIN(16, 16, 16)
        self.gin3 = self.get_GIN(16, 16, 16)

        #self.pool = nn.EdgePooling(19) 

        self.equiv = E_GCL(0, 16, 16, tanh=False, residual=False)
        self.equiv1 = E_GCL(16, 16, 16, tanh=False)
        self.equiv2 = E_GCL(16, 16, 16, tanh=False)
        self.equiv3 = E_GCL(16, 16, 16, tanh=False)
        self.equiv4 = E_GCL(16, 16, 16, tanh=False)
        self.equiv5 = E_GCL(16, 16, 16, tanh=False)
        self.equiv6 = E_GCL(16,16,16, tanh=False)

        self.gmp = nn.global_mean_pool
        self.gap = nn.global_max_pool

        self.classifier = Sequential(
            Dropout(p=0.5),
            Linear(2 * 16 +1, 16),
            ELU(alpha=0.1),
            #torch.nn.Sigmoid(),
            Dropout(p=0.5),
            Linear(16,  dataset.num_classes),
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
    def __init__(self, dataset, num_gin=2, num_equiv=2):
        super().__init__(dataset, num_gin, num_equiv)

    def forward(self, x, edge_index, batch, segment):
        #h = torch.zeros(x.shape[0], 1, device = 'cuda')
        #x = x-x.mean(dim=0)

        #x = self.gin(x.norm(dim=1).unsqueeze(dim=1), edge_index)
        #x = F.elu(x, alpha=0.1)
        #x = self.gin2(x, edge_index)
        #x = F.elu(x, alpha=0.1)

      #  if self.num_gin==0:
      #      rep, pos, _ =self.equiv(None, edge_index, x, None)
      #  else:
      #      rep = self.gin(x.norm(dim=1).unsqueeze(dim=1), edge_index)
      #      rep = F.elu(rep, alpha=0.1)
      #      if self.num_gin>1:
      #          rep = self.gin2(rep, edge_index)
      #          rep = F.elu(rep, alpha=0.1)
      #          if self.num_gin>2:
      #              rep = self.gin3(rep, edge_index)
      #              rep = F.elu(rep, alpha=0.1)
      #      rep, pos = rep, x 
        
      #  rep, pos, _ = self.equiv(None, edge_index, x, None)

       # reppos = torch.cat([rep, pos], dim=1)
       # reppos, edge_index, batch, _ = self.pool(reppos, edge_index, batch)
       # rep, pos = reppos.split([16, 3], dim=1)
       # rep, pos = rep.clone(), pos.clone()
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
        
        
        
        #reppos = torch.cat([rep, pos], dim=1)
        #reppos, edge_index, batch, _ = self.pool(reppos, edge_index, batch)
        #rep, pos = reppos.split([16, 3], dim=1)
        #rep, pos = rep.clone(), pos.clone()
        
        #rep, pos, _ = self.equiv2(rep, edge_index, pos, None)
        
        #rep, pos, _ = self.equiv3(rep, edge_index, pos, None)
        
        #rep2 = torch.cat([rep, pos.norm(dim=1).unsqueeze(dim=1)], dim=1)
        
        #rep2, pos2, _ = self.equiv3(pos.detach().clone().norm(dim=1).unsqueeze(dim=1), edge_index, x, None)
        
        #rep, pos, _ = self.equiv4(rep, edge_index, pos, None)
        #rep, pos, _ = self.equiv5(rep, edge_index, pos, None)
        #rep, pos, _ = self.equiv6(rep, edge_index, pos, None)

        x1 = self.gmp(rep, batch)
        x2 = self.gap(rep, batch)

        #x3 = self.gmp(rep2, batch)
        #x4 = self.gap(rep2, batch)

        #x5 = self.gmp(pos3.norm(dim=1).unsqueeze(dim=1), batch)
        #x6 = self.gap(pos3.norm(dim=1).unsqueeze(dim=1), batch)

        x = torch.cat([x1, x2], dim=1)
        x = torch.cat((x, segment.view(-1, 1)), dim=1)
        x = self.classifier(x)
        return x


class GnnBaseline(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()

        self.gin_input = self.get_GIN(3, 16, 16)
        self.gin16 = self.get_GIN(16, 16, 16)

        self.pool = nn.EdgePooling(16) # this layer could not run on GPU for Jacob
        self.gmp = nn.global_mean_pool
        self.gap = nn.global_max_pool

        self.classifier = Sequential(
            Dropout(p=0.5),
            Linear(2 * 16 + 1, 16),
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


class NoPhysicsGnn(GnnBaseline):
    """Model for WssToCnc and CoordToCnc"""

    def __init__(self, dataset):
        super().__init__(dataset)

    def forward(self, x, edge_index, batch, segment):

        #x = x - x.mean(dim=0)
       # x = x.div(x.)
        x = self.gin_input(x, edge_index)
        x = F.elu(x, alpha=0.1)
        #x, edge_index, batch, _ = self.pool(x, edge_index, batch)

        x = self.gin16(x, edge_index)
        x = F.elu(x, alpha=0.1)
        #x, edge_index, batch, _ = self.pool(x, edge_index, batch)

        x = self.gin16(x, edge_index)
        x = F.elu(x, alpha=0.1)
        x1 = self.gmp(x, batch)
        x2 = self.gap(x, batch)

        x = torch.cat([x1, x2], dim=1)
        x = torch.cat((x, segment.view(-1, 1)), dim=1)   #TODO: CHange this into a one-hot-encoding
        x = self.classifier(x)
        return x
