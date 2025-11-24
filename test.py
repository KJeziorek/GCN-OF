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

train_loader = DataLoader(train_ds, batch_size=4, num_workers=4,
                          shuffle=True, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds, batch_size=4, num_workers=4,
                          shuffle=False, collate_fn=collate_fn)


# ---------------- MODEL & OPTIMIZER -------------
device = "cuda"
model = Model().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


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

        gt_nodes = sample_flow_at_nodes(batch['flow'].to(device),
                                        batch['pos'].to(device),
                                        batch['batch'].to(device))

        loss, l1, smooth = optical_flow_loss(pred, gt_nodes, batch['edge_index'].to(device))
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    return np.mean(total_loss)


# ==================================================
# ----------------  EVALUATION LOOP ----------------
# ==================================================
def evaluate():
    model.eval()
    total_aee, total_acc, total_out = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            pred = model(batch['x'].unsqueeze(1).to(device),
                         batch['pos'].to(device),
                         batch['edge_index'].to(device),
                         batch['batch'].to(device))

            gt_nodes = sample_flow_at_nodes(batch['flow'].to(device),
                                            batch['pos'].to(device),
                                            batch['batch'].to(device))

            total_aee.append(AEE(pred, gt_nodes).item())
            total_acc.append(flow_accuracy(pred, gt_nodes).item())
            total_out.append(percent_outliers(pred, gt_nodes).item())

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
