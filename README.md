```bash
# Create conda env
conda create -n gcn_of python=3.9
conda activate gcn_of

# Install PyTorch (CUDA 12.1/12.8 depending on your setup; adjust if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# or:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install Python dependencies
pip install \
    omegaconf \
    opencv-python \
    matplotlib \
    psutil \
    wandb \
    lightning \
    numba \
    pybind11 \
    tqdm \
    pandas \
    loguru \
    pycocotools

# Install HDF5 related libs
conda install h5py
conda install -c conda-forge blosc-hdf5-plugin

# Inside main folder
cd GCN-OF
python setup.py build_ext --inplace
```

---

### Data

**Optical flow GT:**

* Download MVSEC optical flow ground truth from:
  [https://daniilidis-group.github.io/mvsec/download/](https://daniilidis-group.github.io/mvsec/download/)

**Pre-processed MVSEC data:**

* Additional data from:
  [https://drive.google.com/drive/folders/1rwyRk26wtWeRgrAx_fgPc-ubUzTFThkV](https://drive.google.com/drive/folders/1rwyRk26wtWeRgrAx_fgPc-ubUzTFThkV)

**Convert `.npz` → `.npy`:**

Use `convert_npz_to_npy.py` inside `dataset/utils` to convert the original MVSEC `.npz` flow files into `.npy`:

```bash
python dataset/utils/convert_npz_to_npy.py
```

---

### Expected data tree

Your `data` folder should look like this:

```text
data/
└── mvsec
    └── indoor_flying
        ├── indoor_flying1_data.hdf5
        ├── indoor_flying1_gt_flow_dist
        │   ├── timestamps.npy
        │   ├── x_flow_dist.npy
        │   └── y_flow_dist.npy
        ├── indoor_flying1_gt_flow_dist.npz
        ├── indoor_flying2_data.hdf5
        ├── indoor_flying2_gt_flow_dist
        │   ├── timestamps.npy
        │   ├── x_flow_dist.npy
        │   └── y_flow_dist.npy
        ├── indoor_flying2_gt_flow_dist.npz
        ├── indoor_flying3_data.hdf5
        ├── indoor_flying3_gt_flow_dist
        │   ├── timestamps.npy
        │   ├── x_flow_dist.npy
        │   └── y_flow_dist.npy
        ├── indoor_flying3_gt_flow_dist.npz
        ├── indoor_flying4_data.hdf5
        ├── indoor_flying4_gt_flow_dist
        │   ├── timestamps.npy
        │   ├── x_flow_dist.npy
        │   └── y_flow_dist.npy
        └── indoor_flying4_gt_flow_dist.npz
```
