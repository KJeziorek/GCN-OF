import os
import imageio

folder = "output_flow_vis"
if not os.path.isdir(folder):
    print("Folder not found:", folder)
else:
    files = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
    print("Found", len(files), "images")
    frames = []
    for f in files:
        frames.append(imageio.v2.imread(os.path.join(folder, f)))
    gif_path = os.path.join(folder, "flow_vis.gif")
    if frames:
        imageio.mimsave(gif_path, frames, duration=0.05)
        gif_path
    else:
        print("No frames found")