
"""
Validation script for test set evaluation (RML2016/RML2018A).
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import h5py

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from MT_MCNet import MT_MCNet


def forward_no_memory(model, inputs, supports_update_memory):
    def normalize(output):
        if isinstance(output, (tuple, list)):
            return output[0]
        return output
    if supports_update_memory is False:
        return normalize(model(inputs)), False
    if supports_update_memory is True:
        return normalize(model(inputs, update_memory=False)), True
    try:
        return normalize(model(inputs, update_memory=False)), True
    except TypeError as exc:
        if "update_memory" in str(exc):
            return normalize(model(inputs)), False
        raise


class RML2016DataLoader:
    def __init__(self, filename):
        self.filename = filename
        self.p = self.load_data()

    def load_data(self):
        with open(self.filename, "rb") as f:
            u = pickle._Unpickler(f)
            u.encoding = "latin1"
            p = u.load()
        return p

    def get_data(self):
        snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], self.p.keys())))), [1, 0])

        X = []
        lbl = []
        Z = []
        for mod in mods:
            for snr in snrs:
                X.append(self.p[(mod, snr)])
                for i in range(self.p[(mod, snr)].shape[0]):
                    lbl.append((mod, snr))
                    Z.append(snr)

        X = np.vstack(X)
        Z = np.vstack(Z)

        # RML2016: [N, 2, 128] -> [N, 128, 2]
        if len(X.shape) == 3:
            if X.shape[1] == 2 and X.shape[2] == 128:
                X = np.transpose(X, (0, 2, 1))
            elif X.shape[1] == 128 and X.shape[2] == 2:
                pass
            else:
                if X.shape[1] < X.shape[2]:
                    X = np.transpose(X, (0, 2, 1))
        return X, Z, lbl, snrs, mods


class RML2018ADataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.h5_path = self._find_hdf5_file()

    def _find_hdf5_file(self):
        if os.path.isfile(self.dataset_path) and (self.dataset_path.endswith(".h5") or self.dataset_path.endswith(".hdf5")):
            return self.dataset_path
        if os.path.isdir(self.dataset_path):
            for file in os.listdir(self.dataset_path):
                if file.endswith(".h5") or file.endswith(".hdf5"):
                    return os.path.join(self.dataset_path, file)
        raise FileNotFoundError(f"No HDF5 file found in {self.dataset_path}")

    def get_data(self, num_samples=None):
        print(f"Loading RML2018A data from: {self.h5_path}")
        with h5py.File(self.h5_path, "r") as f:
            X_full = f["X"][:]  # [N, 2, T] or [N, T, 2]
            Y_full = f["Y"][:]  # [N, num_classes]
            Z_full = f["Z"][:]  # [N]

            if num_samples is not None and num_samples < len(X_full):
                indices = np.random.choice(len(X_full), num_samples, replace=False)
                X_full = X_full[indices]
                Y_full = Y_full[indices]
                Z_full = Z_full[indices]

            if X_full.shape[1] == 2 and len(X_full.shape) == 3:
                X_full = np.transpose(X_full, (0, 2, 1))
            if len(X_full.shape) != 3 or X_full.shape[2] != 2:
                print(f"Warning: Unexpected RML2018A X shape after conversion: {X_full.shape}. Expected [N, T, 2].")

            if len(Z_full.shape) > 1:
                Z_full = Z_full.flatten()

            signal_length = X_full.shape[1]
            snrs = sorted(np.unique(Z_full).tolist())
            num_classes = Y_full.shape[1]
            mods = list(range(num_classes))

            lbl = []
            for i in range(len(X_full)):
                class_idx = np.argmax(Y_full[i])
                snr_val = int(Z_full[i]) if isinstance(Z_full[i], (np.ndarray, np.generic)) else Z_full[i]
                lbl.append((class_idx, snr_val))

            Z_full = np.array(Z_full).flatten()
            return X_full, Z_full, lbl, snrs, mods, Y_full, signal_length


def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy) + 1])
    yy1[np.arange(len(yy)), yy] = 1
    return yy1


def load_test_split(data_path, dataset_type, num_samples=None, split_seed=2016):
    if dataset_type.lower() == "rml2016":
        dataset_load = RML2016DataLoader(data_path)
        X, Z, lbl, snrs, mods = dataset_load.get_data()
        Y_full = None
        signal_length = X.shape[1]
    elif dataset_type.lower() == "rml2018a":
        dataset_load = RML2018ADataLoader(data_path)
        X, Z, lbl, snrs, mods, Y_full, signal_length = dataset_load.get_data(num_samples=num_samples)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Must be 'rml2016' or 'rml2018a'")

    np.random.seed(split_seed)
    n_examples = int(X.shape[0])
    n_train = int(n_examples * 0.7)
    n_val = int(n_examples * 0.15)

    train_idx = np.random.choice(range(n_examples), size=n_train, replace=False)
    remaining_idx = list(set(range(n_examples)) - set(train_idx))
    val_idx = np.random.choice(remaining_idx, size=n_val, replace=False)
    test_idx = list(set(remaining_idx) - set(val_idx))

    X_test = X[test_idx]
    Z_test = Z[test_idx]

    if Y_full is not None:
        Y_test = Y_full[test_idx]
        num_classes = Y_full.shape[1]
    else:
        Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))
        num_classes = len(mods)

    X_test = np.reshape(X_test, (-1, 1, signal_length, 2))
    return X_test, Y_test, Z_test, snrs, mods, num_classes, signal_length


def load_model(
    checkpoint_path,
    num_classes,
    device,
    embed_dim,
    depth,
    memory_tokens,
    heads,
    use_dataparallel=False,
    strict_load=True
):
    model = MT_MCNet(
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        memory_tokens=memory_tokens,
        heads=heads
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    # Handle DataParallel prefixes
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=strict_load)
    model = model.to(device)
    if use_dataparallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.eval()
    return model


# -------------------------------------------------
# Full evaluation + per-SNR confusion + accuracy-vs-SNR
# -------------------------------------------------
def evaluate_test_set(
    model,
    device,
    X_test,
    Y_test,
    Z_test,
    batch_size,
    mods,
    save_dir
):
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    os.makedirs(save_dir, exist_ok=True)
    supports_update_memory = None

    # ---------------- Overall test evaluation ----------------
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_test, dtype=torch.float32)
    test_loader = DataLoader(
        TensorDataset(X_tensor, Y_tensor),
        batch_size=batch_size,
        shuffle=False
    )

    total_loss, total_correct, total_count = 0, 0, 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs, supports_update_memory = forward_no_memory(model, x, supports_update_memory)

            y_true = y.argmax(dim=1)
            loss = criterion(outputs, y_true)

            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * x.size(0)
            total_correct += (preds == y_true).sum().item()
            total_count += x.size(0)

    print(f"Test Loss: {total_loss/total_count:.4f}, "
          f"Test Acc: {total_correct/total_count:.4f}")

    # ---------------- Per-SNR evaluation ----------------
    snr_levels = np.sort(np.unique(Z_test.reshape(-1)))
    class_acc = {mod: [] for mod in mods}
    results = {}

    for snr in snr_levels:
        idx = Z_test.reshape(-1) == snr
        if idx.sum() == 0:
            continue

        X_snr = torch.tensor(X_test[idx], dtype=torch.float32).to(device)
        Y_snr = torch.tensor(Y_test[idx], dtype=torch.float32).to(device)

        snr_loader = DataLoader(
            TensorDataset(X_snr, Y_snr),
            batch_size=batch_size,
            shuffle=False
        )

        y_true, y_pred = [], []

        with torch.no_grad():
            for x, y in snr_loader:
                outputs, supports_update_memory = forward_no_memory(model, x, supports_update_memory)

                y_true.extend(y.argmax(dim=1).cpu().numpy())
                y_pred.extend(outputs.argmax(dim=1).cpu().numpy())

        # ---- Confusion matrix ----
        cm, cm_percent, acc = plot_confusion_matrix(
            y_true, y_pred, snr, mods, save_dir
        )

        np.save(os.path.join(save_dir, f"cm_raw_snr_{snr}.npy"), cm)
        np.save(os.path.join(save_dir, f"cm_percent_snr_{snr}.npy"), cm_percent)

        results[snr] = {"accuracy": acc}

        # ---- Per-modulation accuracy ----
        total = cm.sum(axis=1)
        correct = np.diag(cm)
        for i, mod in enumerate(mods):
            class_acc[mod].append(
                correct[i] / total[i] if total[i] > 0 else 0.0
            )

        print(f"SNR {snr:>3} dB → Accuracy: {acc*100:.2f}%")

    # ---------------- Accuracy vs SNR plot ----------------
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'H', '+', 'x', '1']
    plt.figure(figsize=(10, 8))

    for (mod, accs), mk in zip(class_acc.items(), markers):
        plt.plot(
            snr_levels,
            accs,
            marker=mk,
            linewidth=2,
            label=mod
        )

    plt.xlabel("SNR (dB)", fontsize=15, fontweight="bold")
    plt.ylabel("Accuracy", fontsize=15, fontweight="bold")
    plt.xticks(fontsize=13, fontweight="bold")
    plt.yticks(fontsize=13, fontweight="bold")
    plt.grid(True)
    plt.legend(loc="upper left", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "modulation_accuracy_vs_snr.png"), dpi=300)
    plt.show()

    return results, class_acc



import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score

# -------------------------------
# Confusion matrix plotting
# -------------------------------
def plot_confusion_matrix(y_true, y_pred, snr, labels, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(8, 8))
    im = plt.imshow(cm_percent, cmap=plt.cm.Blues)

    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, f"{val}\n{cm_percent[i, j]:.1f}",
                 ha="center", va="center",
                 color="white" if cm_percent[i, j] > 50 else "black",
                 fontsize=9)

    plt.title(f"SNR = {snr} dB | Acc = {acc*100:.2f}%", fontweight="bold")
    plt.xlabel("Predicted Label", fontweight="bold")
    plt.ylabel("True Label", fontweight="bold")
    plt.xticks(range(len(labels)), labels, rotation=60)
    plt.yticks(range(len(labels)), labels)
    plt.colorbar(im, fraction=0.046)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"confusion_snr_{snr}.png"), dpi=300)
    plt.close()

    return cm, cm_percent, acc


import torch
from thop import profile
from thop import clever_format
def main():
    parser = argparse.ArgumentParser(description="Validate test set for RML2016/RML2018A")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset file (RML2016) or directory/file (RML2018A)")
    parser.add_argument("--dataset_type", type=str, default="rml2016", choices=["rml2016", "rml2018a"],
                        help="Dataset type: rml2016 (pickle) or rml2018a (HDF5)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Optional limit on number of samples to load (None for all)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for evaluation")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Seed for train/val/test split (match training)")
    parser.add_argument("--embed_dim", type=int, default=64,
                        help="Model embed_dim (must match training)")
    parser.add_argument("--depth", type=int, default=4,
                        help="Number of transformer blocks")
    parser.add_argument("--memory_tokens", type=int, default=32,
                        help="Number of memory tokens")
    parser.add_argument("--heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--strict_load", action="store_true", default=True,
                        help="Fail if checkpoint does not exactly match model")
    parser.add_argument("--no_strict_load", dest="strict_load", action="store_false",
                        help="Allow partial checkpoint load (not recommended)")
    parser.add_argument("--use_dataparallel", action="store_true", default=False,
                        help="Enable DataParallel for evaluation")

    args = parser.parse_args()

    X_test, Y_test, Z_test, snrs, mods, num_classes, signal_length = load_test_split(
        args.data_path,
        args.dataset_type,
        num_samples=args.num_samples,
        split_seed=args.split_seed
    )
    print(f"Test samples: {len(X_test)}")
    print(f"Signal length: {signal_length}, Num classes: {num_classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        args.checkpoint_path,
        num_classes,
        device,
        embed_dim=args.embed_dim,
        depth=args.depth,
        memory_tokens=args.memory_tokens,
        heads=args.heads,
        use_dataparallel=args.use_dataparallel,
        strict_load=args.strict_load
    )
    model.eval()

    # Example input for FLOPs (must match model forward input)
    input_data = torch.randn(1, 1, 128, 2).to(device)

    # Unwrap DataParallel for THOP
    model_for_profile = model.module if isinstance(model, torch.nn.DataParallel) else model

    flops, params = profile(model_for_profile, inputs=(input_data,))
    flops, params = clever_format([flops, params], "%.3f")

    print(f"FLOPs: {flops}")
    print(f"Parameters: {params}")

    # model = load_model(args.checkpoint_path, num_classes, device)
    evaluate_test_set(model, device, X_test, Y_test, Z_test, args.batch_size, mods, "results")


if __name__ == "__main__":
    main()
