import h5py
import numpy as np
import matplotlib.pyplot as plt
import bisect

# Load data
data = h5py.File('data/mvsec/indoor_flying/indoor_flying1_data.hdf5', 'r')
events = data['davis']['left']['events'][:]  # (N, 4) [x, y, t, p]

image_inds = data['davis']['left']['image_raw_event_inds'][:]  # event index per image
image_frames = data['davis']['left']['image_raw'][:]           # grayscale frames (uint8)

of = np.load('data/mvsec/indoor_flying/indoor_flying1_gt_flow_dist.npz')
xflow = of['x_flow_dist']
yflow = of['y_flow_dist']
flow_ts = of['timestamps']

H, W = xflow.shape[1], xflow.shape[2]

for idx in range(333, 444):
    t_flow = flow_ts[idx]
    t_start = t_flow - 0.4

    # nearest grayscale frame to optical flow timestamp
    # find the event index closest to t_flow
    event_times = events[:, 2]
    frame_event_idx = np.searchsorted(event_times, t_flow)
    # find the closest image frame index from event index lists
    img_idx = np.searchsorted(image_inds, frame_event_idx) - 1
    img_idx = np.clip(img_idx, 0, len(image_frames) - 1)
    frame = image_frames[img_idx]

    # select events near timestamp window
    ev_mask = (event_times >= t_start) & (event_times <= t_flow)
    sel = events[ev_mask]

    # build polarity visualization
    ev_rgb = np.zeros((H, W, 3), dtype=np.float32)
    for x, y, t, p in sel:
        xi, yi = int(x), int(y)
        if 0 <= yi < H and 0 <= xi < W:
            if p > 0:
                ev_rgb[yi, xi, 0] += 1.0  # red
            else:
                ev_rgb[yi, xi, 2] += 1.0  # blue
    
    ev_rgb[ev_rgb > 5] = 5
    ev_rgb = np.clip(ev_rgb / (np.max(ev_rgb) if np.max(ev_rgb) > 0 else 1), 0, 1)

    # compute flow magnitude
    flow_mag = np.sqrt(xflow[idx]**2 + yflow[idx]**2)

    # === Plot ===
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title(f"Grayscale Frame idx={img_idx}")
    plt.imshow(frame, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Events (red=ON, blue=OFF)")
    plt.imshow(ev_rgb)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Optical Flow Magnitude")
    plt.imshow(flow_mag, cmap='inferno')
    plt.axis("off")

    plt.show()
