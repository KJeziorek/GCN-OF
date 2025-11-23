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
    Dataset = object  # dummy base
    TORCH_AVAILABLE = False


class MVSECDataset(Dataset):
    """
    MVSEC dataset for event-based optical flow.

    - Supports multiple sequences (train/test split via OmegaConf)
    - Loads events and optical flow for left/right DAVIS
    - Returns 2-channel event image [2, H, W] (ON/OFF) and flow [2, H, W] (Vx, Vy)
    - Camera selection: left / right / random
    - Uses a time window (in seconds) BEFORE the flow timestamp to accumulate events
    """

    def __init__(self, cfg, split: str = "train"):
        """
        Args:
            cfg: OmegaConf / DictConfig with fields:
                mvsec_root: root folder for this scenario
                splits:
                  train: [list of sequences]
                  test: [list of sequences]
                camera.mode: 'left' | 'right' | 'random'
                event_window: float, seconds
            split: 'train' or 'test'
        """
        self.cfg = cfg
        self.split = split
        self.mvsec_root: str = cfg.mvsec_root
        self.camera_mode: str = cfg.camera.mode
        self.event_window: float = float(cfg.event_window)

        # list of sequence names, e.g. ["indoor_flying1", "indoor_flying2"]
        self.sequence_names: List[str] = list(getattr(cfg.splits, split))

        # storage
        self._events: Dict[str, List[np.ndarray]] = {"left": [], "right": []}
        self._event_times: Dict[str, List[np.ndarray]] = {"left": [], "right": []}
        self._flows: Dict[str, List[Dict[str, np.ndarray]]] = {"left": [], "right": []}
        self._flow_ts: Dict[str, List[np.ndarray]] = {"left": [], "right": []}
        self._h5_files: List[h5py.File] = []   # keep references to avoid GC issues
        self._index: List[Tuple[int, int]] = []  # (seq_idx, frame_idx) global index
        self.height = None
        self.width = None

        self._load_sequences()

    # -------------------------------------------------------------------------
    # internal loading
    # -------------------------------------------------------------------------    
    def _load_sequences(self):
        """
        Open all HDF5 + flow npz files and build a global index.
        """
        for seq_idx, seq in enumerate(self.sequence_names):
            # Paths for this sequence
            # h5: events for both left/right in single file
            h5_path = os.path.join(self.mvsec_root, f"{seq}_data.hdf5")

            # flow for left camera
            flow_left_path = os.path.join(
                self.mvsec_root, f"{seq}_gt_flow_dist.npz"
            )

            # flow for right camera (adjust name if your files differ)
            flow_right_path = os.path.join(
                self.mvsec_root, f"{seq}_gt_flow_dist_right.npz"
            )

            if not os.path.isfile(h5_path):
                raise FileNotFoundError(h5_path)
            if not os.path.isfile(flow_left_path):
                raise FileNotFoundError(flow_left_path)
            if not os.path.isfile(flow_right_path):
                # If you don't have right flow, you can:
                # - raise here, or
                # - skip adding right camera support
                raise FileNotFoundError(flow_right_path)

            # open data file
            h5 = h5py.File(h5_path, "r")
            self._h5_files.append(h5)

            # load events (left/right)
            ev_left = h5["davis"]["left"]["events"][:]   # (N,4): x, y, t, p
            ev_right = h5["davis"]["right"]["events"][:] # (N,4): x, y, t, p

            # store events + times
            self._events["left"].append(ev_left)
            self._events["right"].append(ev_right)
            self._event_times["left"].append(ev_left[:, 2])
            self._event_times["right"].append(ev_right[:, 2])

            # load flows
            fl_left = np.load(flow_left_path)
            fl_right = np.load(flow_right_path)

            self._flows["left"].append(fl_left)
            self._flows["right"].append(fl_right)
            self._flow_ts["left"].append(fl_left["timestamps"])
            self._flow_ts["right"].append(fl_right["timestamps"])

            # assume left and right have same number of GT frames
            n_frames = len(fl_left["timestamps"])
            if len(fl_right["timestamps"]) != n_frames:
                raise RuntimeError(
                    f"Left/right flow length mismatch in sequence {seq}"
                )

            # set H,W from flow if not set
            if self.height is None or self.width is None:
                self.height, self.width = fl_left["x_flow_dist"].shape[1:3]

            # add to global index
            for i in range(n_frames):
                self._index.append((seq_idx, i))

        print(f"Loaded {len(self.sequence_names)} sequences, "
              f"{len(self._index)} frames total.")

    # -------------------------------------------------------------------------
    # dataset API
    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seq_idx, frame_idx = self._index[idx]

        # pick camera
        if self.camera_mode == "left":
            cam = "left"
        elif self.camera_mode == "right":
            cam = "right"
        else:  # random
            cam = "left" if random.random() < 0.5 else "right"

        return self._load_item(seq_idx, frame_idx, cam)

    # -------------------------------------------------------------------------
    # Core item loading
    # -------------------------------------------------------------------------
    def _load_item(self, seq_idx: int, frame_idx: int, cam: str) -> Dict[str, Any]:
        """
        seq_idx: which sequence in self.sequence_names
        frame_idx: index into flow timestamps for that sequence
        cam: 'left' or 'right'
        """
        assert cam in ("left", "right")
        seq_name = self.sequence_names[seq_idx]

        # Grab flow for this seq/cam
        flow_npz = self._flows[cam][seq_idx]
        flow_ts = self._flow_ts[cam][seq_idx]
        ts_flow = flow_ts[frame_idx]

        x_flow = flow_npz["x_flow_dist"][frame_idx]  # (H,W)
        y_flow = flow_npz["y_flow_dist"][frame_idx]  # (H,W)
        flow = np.stack([x_flow, y_flow], axis=0)    # (2,H,W)

        # Time window for events
        t_start = ts_flow - self.event_window

        events = self._events[cam][seq_idx]
        t_arr = self._event_times[cam][seq_idx]

        # Find index range using binary search (assuming events sorted by time)
        start_idx = np.searchsorted(t_arr, t_start, side="left")
        end_idx = np.searchsorted(t_arr, ts_flow, side="right")
        sel = events[start_idx:end_idx]  # (M,4)

        # Build 2-channel event image (ON/OFF)
        ev_img = self._accumulate_events(sel)

        sample: Dict[str, Any] = {
            "events": ev_img,        # (2,H,W)
            "flow": flow,           # (2,H,W)
            "timestamp": float(ts_flow),
            "sequence": seq_name,
            "frame_idx": frame_idx,
            "camera": cam,
        }

        if TORCH_AVAILABLE:
            sample["events"] = torch.from_numpy(sample["events"]).float()
            sample["flow"] = torch.from_numpy(sample["flow"]).float()

        return sample

    # -------------------------------------------------------------------------
    # Event accumulation
    # -------------------------------------------------------------------------
    def _accumulate_events(self, events: np.ndarray) -> np.ndarray:
        """
        Accumulate events into a 2-channel image [2,H,W]:
        channel 0: positive (ON)
        channel 1: negative (OFF)
        """
        H, W = self.height, self.width
        ev = np.zeros((2, H, W), dtype=np.float32)

        if events.size == 0:
            return ev

        xs = events[:, 0].astype(np.int32)
        ys = events[:, 1].astype(np.int32)
        ps = events[:, 3].astype(np.int8)

        # for safety, clip to valid range
        xs = np.clip(xs, 0, W - 1)
        ys = np.clip(ys, 0, H - 1)

        # positive / negative masks
        pos_mask = ps > 0
        neg_mask = ps <= 0

        # accumulate ON events
        np.add.at(ev[0], (ys[pos_mask], xs[pos_mask]), 1.0)
        # accumulate OFF events
        np.add.at(ev[1], (ys[neg_mask], xs[neg_mask]), 1.0)

        return ev
