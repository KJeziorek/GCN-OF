import os
import numpy as np

def convert_npz_to_npy(mvsec_root: str, sequence: str):
    """
    Converts MVSEC <sequence>_gt_flow_dist.npz into folder:
      <sequence>_gt_flow_dist/x_flow_dist.npy
      <sequence>_gt_flow_dist/y_flow_dist.npy
      <sequence>_gt_flow_dist/timestamps.npy
    """

    npz_path = os.path.join(mvsec_root, f"{sequence}_gt_flow_dist.npz")
    out_dir = os.path.join(mvsec_root, f"{sequence}_gt_flow_dist")

    if not os.path.isfile(npz_path):
        raise FileNotFoundError(npz_path)

    # Create directory if missing
    os.makedirs(out_dir, exist_ok=True)

    # Load npz lazily
    fl = np.load(npz_path)

    # Extract data
    x = fl["x_flow_dist"].astype(np.float32)
    y = fl["y_flow_dist"].astype(np.float32)
    ts = fl["timestamps"].astype(np.float64)

    # Save to arranged tree
    np.save(os.path.join(out_dir, "x_flow_dist.npy"), x)
    np.save(os.path.join(out_dir, "y_flow_dist.npy"), y)
    np.save(os.path.join(out_dir, "timestamps.npy"), ts)

    print(f"[OK] Converted {sequence}: saved into {out_dir}")
    print(f"     x {x.shape}, y {y.shape}, ts {ts.shape}")


if __name__ == "__main__":
    mvsec_root = "data/mvsec/indoor_flying"
    sequences = ["indoor_flying1", "indoor_flying2", "indoor_flying3", "indoor_flying4"]

    for seq in sequences:
        convert_npz_to_npy(mvsec_root, seq)
