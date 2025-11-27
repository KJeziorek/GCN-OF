import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

from dataset.mvsec.mvsec import MVSECDataset
from dataset.mvsec.collate_fn import collate_fn

from model.model import Model
from model.loss_fn import optical_flow_loss, sample_flow_at_nodes
from model.metric_fn import AEE, percent_outliers, flow_accuracy

# ---------------- CONFIG & DATA -----------------
cfg_ds = OmegaConf.load("configs/dataset/mvsec_indoor.yaml")
print(cfg_ds)

train_ds = MVSECDataset(cfg=cfg_ds, split="train")
test_ds  = MVSECDataset(cfg=cfg_ds, split="test")

train_loader = DataLoader(train_ds, batch_size=1, num_workers=1,
                          shuffle=True, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds, batch_size=1, num_workers=1,
                          shuffle=False, collate_fn=collate_fn)


# ---------------- MODEL & OPTIMIZER -------------
device = "cuda"
model = Model().to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-3)


# ==================================================
# ---------------- TRAINING LOOP -------------------
# ==================================================
def train_one_epoch():
    model.train()
    total_loss = []

    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()

        pred = model(batch['x'].unsqueeze(1).to(device),
                     batch['pos'].to(device),
                     batch['edge_index'].to(device),
                     batch['batch'].to(device))

        gt_nodes = batch["flow"].to(device) 

        loss, l1, smooth = optical_flow_loss(pred, 
                                             gt_nodes / 10., 
                                             batch['edge_index'].to(device))
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    return np.mean(total_loss)


# ==================================================
# ----------------  EVALUATION LOOP ----------------
# ==================================================

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

def visualize_and_save(pos, pred, gt, frame_id, save_dir="output_flow_vis", max_points=10000):
    os.makedirs(save_dir, exist_ok=True)

    # Convert to numpy
    pos = pos.cpu().numpy()
    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()

    N = pos.shape[0]
    if N > max_points:
        # randomly select subset for visual clarity
        idx = np.random.choice(N, max_points, replace=False)
        pos  = pos[idx]
        pred = pred[idx]
        gt   = gt[idx]

    # Compute angles
    pred_ang = np.arctan2(pred[:,1], pred[:,0])
    gt_ang   = np.arctan2(gt[:,1], gt[:,0])

    # Normalize angles to [0,1] for colormap
    norm = Normalize(vmin=-np.pi, vmax=np.pi)
    pred_colors = cm.hsv(norm(pred_ang))
    gt_colors   = cm.hsv(norm(gt_ang))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # GT quiver
    axs[0].quiver(pos[:,0], pos[:,1],
                  gt[:,0], gt[:,1],
                  color=gt_colors, angles='xy', scale_units='xy', scale=1)
    axs[0].set_title(f"GT Flow | Frame {frame_id}")
    axs[0].invert_yaxis()

    # Pred quiver
    axs[1].quiver(pos[:,0], pos[:,1],
                  pred[:,0], pred[:,1],
                  color=pred_colors, angles='xy', scale_units='xy', scale=1)
    axs[1].set_title(f"Predicted Flow | Frame {frame_id}")
    axs[1].invert_yaxis()

    filename = os.path.join(save_dir, f"flow_frame_{frame_id:04d}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)

def evaluate():
    model.eval()
    total_aee, total_acc, total_out = [], [], []

    with torch.no_grad():
        for frame_id, batch in enumerate(tqdm(test_loader, desc="Testing")):
            pred = model(batch['x'].unsqueeze(1).to(device),
                         batch['pos'].to(device),
                         batch['edge_index'].to(device),
                         batch['batch'].to(device))

            gt = batch["flow"].to(device)

            pred = pred * 10

            total_aee.append(AEE(pred, gt).item())
            total_acc.append(flow_accuracy(pred, gt).item())
            total_out.append(percent_outliers(pred, gt).item())

            if epoch > 5:
                mask = (batch['batch'] == 0)
                visualize_and_save(batch['pos'][mask],
                                pred[mask].cpu(),
                                gt[mask].cpu(),
                                frame_id)

    return np.mean(total_aee), np.mean(total_acc), np.mean(total_out)

# ==================================================
# ---------------- MAIN TRAINING -------------------
# ==================================================
EPOCHS = 50
best_aee = 1e9

for epoch in range(1, EPOCHS + 1):
    loss = train_one_epoch()
    aee, acc, outl = evaluate()

    print(f"\nEpoch {epoch:02d} | Loss={loss:.4f} | "
          f"AEE={aee:.4f} | Acc={acc:.4f} | Outliers={outl:.2f}% | Scale={model.scale}")
    
    # batch_sample = next(iter(train_loader))

    # with torch.no_grad():
    #     pred = model(batch_sample['x'].unsqueeze(1).to(device),
    #                 batch_sample['pos'].to(device),
    #                 batch_sample['edge_index'].to(device),
    #                 batch_sample['batch'].to(device))

    #     gt_nodes = sample_flow_at_nodes(batch_sample['flow'].to(device),
    #                                     batch_sample['pos'].to(device),
    #                                     batch_sample['batch'].to(device))

    # # visualize only first graph slice in batch
    # mask = (batch_sample['batch'] == 0)
    # visualize_graph_flow(
    #     batch_sample['pos'][mask],
    #     batch_sample['edge_index'],    # edges already global
    #     gt_nodes[mask],
    #     pred[mask]
    # )

    # checkpoint
    # if aee < best_aee:
    #     torch.save(model.state_dict(), "checkpoints/best_flow_model.pth")
    #     best_aee = aee
    #     print(" ✓ Saved best model\n")
