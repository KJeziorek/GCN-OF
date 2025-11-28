import torch
import torch.nn.functional as F

from torch.nn import BatchNorm1d, InstanceNorm1d
import torch.nn as nn

from model.layers.my_pointnet import MyPointNetConv
from model.layers.my_pointtransformer import MyPointTransformerConv
from model.layers.my_linear import MyLinear

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.scale = nn.Parameter(torch.ones(2), requires_grad=False)

        self.conv0 = MyPointTransformerConv(6, 64)
        # self.conv0 = MyPointNetConv(6+3, 64)
        self.convs = nn.ModuleList(
            [MyPointTransformerConv(64, 64) for _ in range(4)]
        )
        # self.convs = nn.ModuleList(
        #     [MyPointNetConv(64+3, 64) for _ in range(4)]
        # )

        self.norms = nn.ModuleList([nn.LayerNorm(64) for _ in range(5)])
        
        self.mlp = nn.Sequential(
            nn.Linear(320, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )

    def forward(self, x, pos, edge_index, batch=None):
        embeddings = []
        for i in range(5):
            h = self.conv0(torch.cat([x,pos/345.], dim=1), pos, edge_index) if i == 0 else self.convs[i-1](x, pos, edge_index)
            h = self.norms[i](h)
            embeddings.append(h)
            x = F.relu(h + x) if i > 0 else F.relu(h)

        z = torch.cat(embeddings, dim=-1)
        z = F.layer_norm(z, z.shape[-1:])
        flow = self.mlp(z) * self.scale
        return flow
