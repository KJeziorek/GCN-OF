import torch
import torch.nn.functional as F

from torch.nn import BatchNorm1d, InstanceNorm1d
import torch.nn as nn

from model.layers.my_pointnet import MyPointNetConv
from model.layers.my_linear import MyLinear

class Model(torch.nn.Module):
    def __init__(self, ):
        super().__init__()

        self.scale = nn.Parameter(torch.ones(2), requires_grad=True)

        self.conv0 = MyPointNetConv(1+3, 
                                64, 
                                bias=False,
                                num_bits=8,
                                first_layer=True)
        
        self.convs = nn.ModuleList(
            [MyPointNetConv(64+3, 64, bias=False, num_bits=8, first_layer=False) for _ in range(4)]
        )

        self.norms = nn.ModuleList(
            [nn.LayerNorm(64) for _ in range(5)]
        )

        # --- MLP head ---
        mlp_in_dim = 320
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, 128),
            # InstanceNorm is applied outside to support batch index
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),   # (Vx, Vy)
        )

    def forward(self, x, pos, edge_index, batch=None):
        """
        x:          [N, F]    node features (e.g. polarity, maybe extra)
        edge_index: [2, E]    long tensor (source, target)
        pos:        [N, 3]    (x, y, t_norm) coordinates
        batch:      [N]       graph index per node
        """
        # --- backbone: collect embeddings from each layer ---
        embeddings = []
        for i in range(5):
            if i == 0:
                h = self.conv0(x, pos, edge_index)
            else:
                h = self.convs[i-1](x, pos, edge_index)

            # h = self.norms[i](h)
            # h = F.relu(h, inplace=True)
            embeddings.append(h)
            x = h
        z = torch.cat(embeddings, dim=-1)
        flow = self.mlp(z)
        return flow * self.scale