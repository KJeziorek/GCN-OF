from dataset.mvsec.mvsec import MVSECDataset
from omegaconf import omegaconf
from tqdm import tqdm
import matrix_neighbour
import torch


import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_events_edges_3d(positions, features, edges):
    """
    positions: (N,3) tensor of (x,y,t)
    features: (N,) polarity {-1,1} or {0,1}
    edges:    (E,2)
    """

    pos = positions.cpu().numpy()
    xs = pos[:, 0]
    ys = pos[:, 1]
    ts = pos[:, 2]

    ps = features.cpu().numpy()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # scatter events
    scatter = ax.scatter(xs, ys, ts, c=ps, cmap='bwr', s=5, alpha=0.8)

    # connect edges
    edges_np = edges.cpu().numpy()
    for src, dst in edges_np:  # subsample for performance (remove [::10] for full detail)
        ax.plot(
            [xs[src], xs[dst]],
            [ys[src], ys[dst]],
            [ts[src], ts[dst]],
            color="green",
            alpha=0.3,
            linewidth=0.5
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("T")

    ax.set_title("3D Event Graph (x,y,t)")
    ax.view_init(elev=25, azim=135)  # rotate view

    plt.show()


cfg_ds = omegaconf.OmegaConf.load('configs/data/mvsec_indoor.yaml')

print(cfg_ds)

ds = MVSECDataset(cfg=cfg_ds, split='train')

for data in tqdm(ds):
    print(data)