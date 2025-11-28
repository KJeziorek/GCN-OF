import os
import random
from typing import List, Tuple, Dict, Any
import h5py
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    Dataset = object
    TORCH_AVAILABLE = False

import matrix_neighbour


class MVSECDataset(Dataset):
    """
    Optimized MVSEC Dataset
    """

    def __init__(self, cfg, split: str = "train"):
        self.cfg = cfg
        self.split = split
        self.mvsec_root: str = cfg.mvsec_root
        self.camera_mode: str = cfg.camera.mode
        self.event_window: float = float(cfg.event_window)
        self.event_count: int = cfg.event_count

        self.radius_xy: int = cfg.graph.radius_xy
        self.radius_t: int = cfg.graph.radius_t
        self.norm_t: int = cfg.graph.norm_t
        self.filtering: bool = cfg.graph.filtering
        self.delta_t: int = cfg.graph.delta_t

        self.sequence_names: List[str] = list(getattr(cfg.splits, split))

        # Data storage
        self._events = {"left": [], "right": []}
        self._event_times = {"left": [], "right": []}
        self._event_index = {"left": [], "right": []}
        self._flow = []
        self._flow_ts = []
        self._h5_files = []
        self._index: List[Tuple[int, int]] = []

        self.height = None
        self.width = None

        self._load_sequences()

    # ----------------------------------------------------------
    def _load_sequences(self):
        for seq_idx, seq in enumerate(self.sequence_names):
            h5_path = os.path.join(self.mvsec_root, f"{seq}_data.hdf5")

            flow_dir = os.path.join(self.mvsec_root, f"{seq}_gt_flow_dist")
            x_flow_path = os.path.join(flow_dir, "x_flow_dist.npy")
            y_flow_path = os.path.join(flow_dir, "y_flow_dist.npy")
            ts_path = os.path.join(flow_dir, "timestamps.npy")

            if not os.path.isfile(h5_path):
                raise FileNotFoundError(h5_path)
            for p in (x_flow_path, y_flow_path, ts_path):
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"Missing NP array: {p}")

            # ----------------------------------------
            # Load event data
            # ----------------------------------------
            h5 = h5py.File(h5_path, "r")
            self._h5_files.append(h5)

            ev_left = h5["davis"]["left"]["events"][:]        # keep raw
            ev_right = h5["davis"]["right"]["events"][:]
            # ----------------------------------------
            # Load flow timestamps (float64)
            # ----------------------------------------
            x = np.load(x_flow_path, mmap_mode="r")
            y = np.load(y_flow_path, mmap_mode="r")
            ts = np.load(ts_path, mmap_mode="r").astype(np.float64)

            # ----------------------------------------
            # Normalize timestamps: ensure both start at 0
            # ----------------------------------------
            t0 = min(ev_left[0, 2], ev_right[0, 2], ts[0])

            ev_left[:, 2] -= t0
            ev_right[:, 2] -= t0
            ts = ts - t0   # normalized flow timestamps

            # Store processed data
            self._events["left"].append(ev_left)
            self._events["right"].append(ev_right)
            self._event_times["left"].append(ev_left[:, 2])
            self._event_times["right"].append(ev_right[:, 2])
            self._flow.append((x, y))
            self._flow_ts.append(ts)

            if self.height is None:
                self.height, self.width = x.shape[1:3]

            # ----------------------------------------
            # Precompute event slice indices for each frame
            # ----------------------------------------
            for cam in ("left", "right"):
                times = self._event_times[cam][seq_idx]
                t_start = ts - self.event_window
                t_start[t_start < 0] = 0.0

                start = np.searchsorted(times, t_start, side="left")
                end = np.searchsorted(times, ts, side="right")

                self._event_index[cam].append(np.stack([start, end], axis=1))

            # Expand dataset index
            n_frames = len(ts)
            self._index.extend((seq_idx, i) for i in range(n_frames))

        print(f"Loaded {len(self.sequence_names)} sequences, {len(self._index)} frames.")


    # ----------------------------------------------------------
    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seq_idx, frame_idx = self._index[idx]

        cam = (
            "left" if self.camera_mode == "left"
            else "right" if self.camera_mode == "right"
            else ("left" if random.random() < 0.5 else "right")
        )

        return self._load_item(seq_idx, frame_idx, cam)

    # ----------------------------------------------------------
    def _load_item(self, seq_idx, frame_idx, cam):

        # ----- get raw event slice -----
        start_idx, end_idx = self._event_index[cam][seq_idx][frame_idx]
        events = self._events[cam][seq_idx][start_idx:end_idx].copy()

        if self.event_count:
            events = events[:self.event_count]

        # ----- build graph -----
        events_torch = torch.from_numpy(events)
        f, p, e, normals, flows = self.generate_graph(events_torch)  # positions: [N',3]
        f = f.unsqueeze(1)

        f = torch.cat([f, flows/20.], dim=1)

        # Apply edge dropout
        if self.split == "train":
            mask = torch.rand(e.size(0)) > 0.25
            keep = mask | (e[:, 0] == e[:, 1])
            e = e[keep]

        # convert to numpy before interpolation
        p_np = p.numpy()
        xs = p_np[:,0].astype(np.int32)
        ys = p_np[:,1].astype(np.int32)
        ts_norm = p_np[:,2]

        # ----- recover real timestamps -----
        # convert normalized time back to real slice timestamps
        t_min = events[:,2].min()
        ts_real = ts_norm / self.norm_t * self.event_window + t_min

        # ----- vectorized GT interpolation -----
        gt_ts = self._flow_ts[seq_idx]
        idx = np.searchsorted(gt_ts, ts_real, side="right") - 1
        idx = np.clip(idx, 0, len(gt_ts) - 2)

        t0 = gt_ts[idx]
        t1 = gt_ts[idx + 1]
        alpha = (ts_real - t0) / (t1 - t0 + 1e-9)

        x_flow_full, y_flow_full = self._flow[seq_idx]

        xs = np.clip(xs, 0, self.width - 1)
        ys = np.clip(ys, 0, self.height - 1)

        flow = np.zeros((len(xs), 2), dtype=np.float32)
        flow[:,0] = (1 - alpha) * x_flow_full[idx, ys, xs] + alpha * x_flow_full[idx+1, ys, xs]
        flow[:,1] = (1 - alpha) * y_flow_full[idx, ys, xs] + alpha * y_flow_full[idx+1, ys, xs]

        # ----- apply augmentation 
        if self.split == "train":
            # ---- Random horizontal flip ----
            if random.random() < 0.5:
                events[:, 0] = self.width - 1 - events[:, 0]
                p[:, 0] = self.width - 1 - p[:, 0]
                flow[:, 0] *= -1

        # ----- Convert to torch -----
        events = torch.from_numpy(events)
        flow = torch.from_numpy(flow)

        return {
            "events": events,
            "features": f.to(torch.float32),
            "positions": p.to(torch.float32),
            "edges": e.to(torch.long),
            "flow": flow.to(torch.float32),
            "timestamp": float(self._flow_ts[seq_idx][frame_idx]),
            "sequence": self.sequence_names[seq_idx],
            "camera": cam,
            "frame_idx": frame_idx,
        }


    
    # ----------------------------------------------------------
    def generate_graph(self, events):
        # generate graph by first normalising to norm_t time
        t_min = events[:, 2].min()
        t_max = events[:, 2].max()
        clip_events = events.clone()
        clip_events[:, 2] = (clip_events[:, 2] - t_min) / (self.event_window) * self.norm_t
        clip_events = clip_events.to(torch.int64)
        features, positions, edges, normals, flows = matrix_neighbour.generate_edges(clip_events, self.radius_xy, self.radius_t, 346, 260, self.filtering, self.delta_t)
        return features, positions, edges, normals, flows
    
    def _interp_event_flow_pixel(self, seq_idx, t_event, x, y):
        x_flow_full, y_flow_full = self._flow[seq_idx]
        ts = self._flow_ts[seq_idx]

        i = np.searchsorted(ts, t_event, side="right") - 1
        i = np.clip(i, 0, len(ts) - 2)

        t0, t1 = ts[i], ts[i+1]
        alpha = (t_event - t0) / (t1 - t0 + 1e-9)

        vx = (1 - alpha) * x_flow_full[i, y, x] + alpha * x_flow_full[i+1, y, x]
        vy = (1 - alpha) * y_flow_full[i, y, x] + alpha * y_flow_full[i+1, y, x]

        return vx, vy


if __name__ == '__main__':
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    # ---------------- CONFIG & DATA -----------------
    cfg_ds = OmegaConf.load("configs/data/mvsec_indoor.yaml")
    print(cfg_ds)

    train_ds = MVSECDataset(cfg=cfg_ds, split="train")

    for data in train_ds:
        print(data)