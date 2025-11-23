import torch
import matplotlib.pyplot as plt


def visualize_graph_flow(pos, edges, gt, pred, width=346, height=260, title="Graph + Flow"):
    """
    pos   : [N,3] integer tensor (x,y,t)
    edges : [E,2] integer tensor
    gt    : [N,2] flow vectors GT
    pred  : [N,2] predicted vectors
    """

    pos = pos.cpu().numpy()
    edges = edges.cpu().numpy()
    gt = gt.cpu().numpy()
    pred = pred.cpu().numpy()

    xs, ys = pos[:,0], pos[:,1]

    plt.figure(figsize=(10,7))
    plt.imshow(torch.zeros(height, width), cmap="gray", vmin=0, vmax=1)

    # ---- plot graph edges ----
    for src, dst in edges[:1500]:   # limit number for speed
        x1, y1 = pos[src][:2]
        x2, y2 = pos[dst][:2]
        plt.plot([x1, x2], [y1, y2], linewidth=0.3, color="white", alpha=0.15)

    # ---- plot nodes ----
    plt.scatter(xs, ys, s=3, c="yellow", alpha=0.6)

    # ---- plot GT optical flow ----
    plt.quiver(xs, ys, gt[:,0], gt[:,1], color="lime", scale=50, width=0.003,
               label="Ground Truth")

    # ---- plot predicted optical flow ----
    plt.quiver(xs, ys, pred[:,0], pred[:,1], color="red", scale=50, width=0.003,
               alpha=0.6, label="Prediction")

    plt.gca().invert_yaxis()
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
