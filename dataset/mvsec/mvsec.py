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

        self.radius: int = cfg.graph.radius
        self.norm_t: int = cfg.graph.norm_t

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
        # fast flow read
        x_flow = self._flow[seq_idx][0][frame_idx].copy()
        y_flow = self._flow[seq_idx][1][frame_idx].copy()
        flow = np.stack([x_flow, y_flow], axis=0)


        # fast events selection
        start_idx, end_idx = self._event_index[cam][seq_idx][frame_idx]
        events = self._events[cam][seq_idx][start_idx:end_idx]

        # ------------------- AUGMENTATIONS ----------------------
        if self.split == "train":

            # Temporal warping  (Uniform 0.5–1.5)
            warp = np.random.uniform(0.5, 1.5)
            events[:, 2] *= warp
            x_flow *= warp
            y_flow *= warp

            # XY flip with flow direction update
            if random.random() < 0.5:
                events[:, 0] = self.width - 1 - events[:, 0]   # flip x coord
                x_flow *= -1                                   # reverse horizontal flow
        # --------------------------------------------------------

        # cut events
        if self.event_count:
            events = events[:self.event_count]

        if TORCH_AVAILABLE:
            flow = torch.from_numpy(flow)
            events = torch.from_numpy(events)

        f, p, e = self.generate_graph(events)

        # Edge dropout (except self loops)
        if self.split == "train":
            mask = torch.rand(e.size(0)) > 0.25
            keep = mask | (e[:, 0] == e[:, 1])   # keep self loops
            e = e[keep]

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
        features, positions, edges = matrix_neighbour.generate_edges(clip_events, self.radius, 346, 260, False, 5)
        return features, positions, edges


