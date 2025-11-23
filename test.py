from dataset.mvsec.mvsec import MVSECDataset
from dataset.mvsec.collate_fn import collate_fn
from omegaconf import omegaconf
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

cfg_ds = omegaconf.OmegaConf.load('configs/data/mvsec_indoor.yaml')

print(cfg_ds)

ds = MVSECDataset(cfg=cfg_ds, split='train')

train_dataloader = DataLoader(ds, batch_size=8, num_workers=4, shuffle=True, collate_fn=collate_fn)

for batch in tqdm(train_dataloader):
    print(batch['x'].shape)
    print(batch['pos'].shape)
    print(batch['edge_index'].shape)
    print(batch['batch'].shape)
    print(batch['flow'].shape)






    #     return {
    #     "x": features,                   # node features
    #     "pos": positions,                # node coords (for GNN)
    #     "edge_index": edges,             # [E,2]
    #     "batch": batch_index,            # graph membership idx
    #     "flow": flows,                   # [B,2,H,W]
    #     "events": raw_events,            # optional raw tensor
    #     "timestamps": timestamps,
    #     "sequences": seq_names,
    #     "cameras": cameras,
    #     "frame_ids": frame_ids,
    # }
