# MT-MCNet: Memory Transformer Model for Automatic Modulation Classification

This repository contains the reference implementation and pretrained weights for the IEEE paper **"MT-MCNet: Memory Transformer Model for Automatic Modulation Classification"**. The paper PDF is included as `MT-MCNet.pdf`.

## Abstract

Automatic modulation classification (AMC) is a key component of cognitive radio networks, requiring robust performance under noisy and dynamic channel conditions. Although traditional deep learning–based AMC methods have shown promising results, their effectiveness is often limited by inadequate feature extraction and weak modeling of long-range temporal dependencies. In this work, we propose MT-MCNet, a memory-based Transformer architecture that incorporates a persistent latent memory into the self-attention mechanism to capture global modulation patterns across training batches.
To enhance robustness against channel noise and signal amplitude variations, a Scale-Aware Gating (SAG) module is introduced, which dynamically suppresses noise using global channel statistics. Furthermore, Sparse Sharpness-Aware Minimization (Sparse SAM) is employed to improve generalization by guiding optimization toward flatter minima while promoting parameter sparsity. Experiments on the DeepSig RML2016A and RML2016B datasets demonstrate that MT-MCNet achieves 93.73% and 94.31% classification accuracy, respectively, outperforming state-of-the-art Transformer-based approaches with larger model sizes. These results highlight the proposed model’s superior accuracy, robustness, and computational efficiency for AMC tasks.

## Contents

- `MT_MCNet.py`: MT-MCNet model definition.
- `train_memory.py`: training script for RML2016/RML2018A datasets.
- `validation.py`: evaluation utilities (primarily for RML2016/RML2018A).
- `MT-MCNet_best_2016_A.pth`: pretrained weights for **RML2016.10A**.
- `MT-MCNet_best_2016_B.pth`: pretrained weights for **RML2016.10B**.

## Model Summary

MT-MCNet combines:

- A multi-kernel temporal encoder for IQ signals.
- Titans-style persistent memory tokens for long-range signal context.
- A memory-augmented transformer stack.
- A dual-branch classifier head for stable modulation predictions.

Refer to `MT_MCNet.py` for architectural details and to `MT-MCNet.pdf` for the full methodology.

## Requirements

Python 3.8+ and common ML packages:

- `torch`
- `numpy`
- `tqdm`
- `h5py` (for RML2018A)
- `wandb` (optional, for logging)
- `scikit-learn` and `thop` (only needed for `validation.py`)

Install a minimal set:

```bash
pip install torch numpy tqdm h5py wandb
```

## Datasets

Supported datasets:

- **RML2016.10A / RML2016.10B**: pickle format with IQ samples.
- **RML2018A**: HDF5 format.

Data formats expected by the scripts:

- RML2016: `[N, 2, 128]` or `[N, 128, 2]` (auto-converted).
- RML2018A: `[N, 2, T]` or `[N, T, 2]` (auto-converted), `T` typically 128 or 1024.

## Training

Example (RML2016.10B):

```bash
python train_memory.py \
  --data_path /path/to/RML2016.10b.dat \
  --dataset_type rml2016 \
  --batch_size 200 \
  --epochs 200 \
  --num_classes 10
```

Example (RML2018A):

```bash
python train_memory.py \
  --data_path /path/to/GOLD_XYZ_OSC.0001_1024.hdf5 \
  --dataset_type rml2018a \
  --batch_size 200 \
  --epochs 200 \
  --num_classes 24
```

Checkpoints are saved in `checkpoint/` by default.

## Using Pretrained Weights (RML2016.10A / 10B)

You can load MT-MCNet and the provided weights directly in PyTorch:

```python
import torch
from MT_MCNet import MT_MCNet

num_classes = 10  # RML2016.10A/B
model = MT_MCNet(num_classes=num_classes)

state = torch.load("MT-MCNet_best_2016_A.pth", map_location="cpu")
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
state = {k.replace("module.", ""): v for k, v in state.items()}
model.load_state_dict(state, strict=True)
model.eval()
```

Use `MT-MCNet_best_2016_B.pth` for RML2016.10B.

## Evaluation

`validation.py` provides end-to-end evaluation for supported datasets, including confusion matrices and accuracy-vs-SNR plots. Review the script before running to ensure model class and checkpoint format match your experiment.

```bash
python validation.py \
  --data_path /path/to/RML2016.10b.dat \
  --dataset_type rml2016 \
  --checkpoint_path /path/to/checkpoint.pth \
  --batch_size 128
```

## Paper

See `MT-MCNet.pdf` for the full IEEE paper and methodological details.

## Citation

If you use this code or model weights, please cite the paper:

```
MT-MCNet: Memory Transformer Model for Automatic Modulation Classification
```
