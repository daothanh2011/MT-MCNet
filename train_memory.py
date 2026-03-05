# -*- coding: utf-8 -*-
"""
Training script for RML2016b_DTNet with Titans-style Memory
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import os
import sys
import wandb
import h5py

# Import the memory-enhanced DTNet
from rml2016_dtnet import DTNet, config as model_config
from MT_MCNet import MT_MCNet

# Data loader for RML2016.10b (pickle format)
class RML2016DataLoader:
    def __init__(self, filename):
        self.filename = filename
        self.p = self.load_data()

    def load_data(self):
        with open(self.filename, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
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
        
        # RML2016 data format: [N, 2, 128] (channels first) -> convert to [N, 128, 2] (time first)
        # Check if data is in channels-first format
        print(f"RML2016 data shape before conversion: {X.shape}")
        if len(X.shape) == 3:
            if X.shape[1] == 2 and X.shape[2] == 128:
                # Format: [N, 2, 128] -> transpose to [N, 128, 2]
                X = np.transpose(X, (0, 2, 1))  # [N, 128, 2]
                print(f"Converted RML2016 data from [N, 2, 128] to [N, 128, 2] format")
            elif X.shape[1] == 128 and X.shape[2] == 2:
                # Already in correct format [N, 128, 2]
                print(f"RML2016 data already in [N, 128, 2] format")
            else:
                print(f"Warning: Unexpected RML2016 data shape: {X.shape}")
                # Try to infer: if second dim is smaller, it might be channels
                if X.shape[1] < X.shape[2]:
                    print(f"Attempting transpose based on dimension sizes...")
                    X = np.transpose(X, (0, 2, 1))
                    print(f"After transpose: {X.shape}")
        else:
            print(f"Warning: Unexpected RML2016 data shape: {X.shape}, expected 3D array")
        
        print(f"RML2016 data shape after conversion: {X.shape}")

        return X, Z, lbl, snrs, mods


# Data loader for RML2018A (HDF5 format)
class RML2018ADataLoader:
    def __init__(self, dataset_path):
        """
        Initialize RML2018A data loader.
        
        Args:
            dataset_path: Path to directory containing the HDF5 file, or direct path to HDF5 file
        """
        self.dataset_path = dataset_path
        self.h5_path = self._find_hdf5_file()
        
    def _find_hdf5_file(self):
        """Find HDF5 file in the dataset directory."""
        if os.path.isfile(self.dataset_path) and (self.dataset_path.endswith('.h5') or self.dataset_path.endswith('.hdf5')):
            return self.dataset_path
        
        # If it's a directory, search for HDF5 file
        if os.path.isdir(self.dataset_path):
            for file in os.listdir(self.dataset_path):
                if file.endswith('.h5') or file.endswith('.hdf5'):
                    return os.path.join(self.dataset_path, file)
        
        raise FileNotFoundError(f"No HDF5 file found in {self.dataset_path}")
    
    def get_data(self, num_samples=None):
        """
        Load data from HDF5 file.
        
        Args:
            num_samples: Optional limit on number of samples to load (None for all)
        
        Returns:
            X: IQ data [N, 2, T] or [N, T, 2] where T can be 128 or 1024
            Z: SNR values [N]
            Y: One-hot labels [N, num_classes]
            snrs: Unique SNR values
            mods: Modulation class names (indices)
        """
        print(f"Loading RML2018A data from: {self.h5_path}")
        
        with h5py.File(self.h5_path, 'r') as f:
            print(f"Keys in HDF5 file: {list(f.keys())}")
            
            # Load data
            X_full = f['X'][:]  # [N, 2, T] or [N, T, 2] - IQ data
            Y_full = f['Y'][:]  # [N, num_classes] - One-hot labels
            Z_full = f['Z'][:]  # [N] - SNR values
            
            # Limit samples if specified
            if num_samples is not None and num_samples < len(X_full):
                indices = np.random.choice(len(X_full), num_samples, replace=False)
                X_full = X_full[indices]
                Y_full = Y_full[indices]
                Z_full = Z_full[indices]
            
            print(f"Loaded {len(X_full)} samples")
            print(f"X shape: {X_full.shape}")
            print(f"Y shape: {Y_full.shape}")
            print(f"Z shape: {Z_full.shape}")
            
            # Handle different X formats: [N, 2, T] or [N, T, 2]
            if X_full.shape[1] == 2 and len(X_full.shape) == 3:
                # Format: [N, 2, T] -> transpose to [N, T, 2]
                X_full = np.transpose(X_full, (0, 2, 1))  # [N, T, 2]
            
            if len(X_full.shape) != 3 or X_full.shape[2] != 2:
                print(f"Warning: Unexpected RML2018A X shape after conversion: {X_full.shape}. Expected [N, T, 2].")
            
            # Handle Z shape: flatten if needed
            if len(Z_full.shape) > 1:
                Z_full = Z_full.flatten()
            
            # Keep original signal length - leverage Titans memory for long sequences
            # Longer sequences allow memory to accumulate more information across time
            signal_length = X_full.shape[1]
            print(f"Signal length: {signal_length} samples (keeping full length for memory accumulation)")
            
            if signal_length not in [128, 1024]:
                print(f"Warning: Unexpected signal length: {signal_length}. Expected 128 or 1024.")
                print("Proceeding with detected signal length...")
            
            # Get unique SNRs
            snrs = sorted(np.unique(Z_full).tolist())
            
            # Get number of classes from Y shape
            num_classes = Y_full.shape[1]
            mods = list(range(num_classes))  # Class indices
            
            # Create lbl list similar to RML2016 format (for compatibility)
            # Each entry is (class_index, snr)
            lbl = []
            for i in range(len(X_full)):
                class_idx = np.argmax(Y_full[i])
                snr_val = int(Z_full[i]) if isinstance(Z_full[i], (np.ndarray, np.generic)) else Z_full[i]
                lbl.append((class_idx, snr_val))
            
            # Ensure Z is 1D array
            Z_full = np.array(Z_full).flatten()
            
            # Return signal_length for model initialization
            return X_full, Z_full, lbl, snrs, mods, Y_full, signal_length


def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1


class SparseSAM(torch.optim.Optimizer):
    """
    Lightweight Sparse SAM wrapper.
    Only a random (1 - sparsity) portion of parameters participates in the
    perturbation step to reduce overhead while retaining SAM benefits.
    """
    def __init__(self, params, base_optimizer, rho=0.05, sparsity=0.5):
        if not isinstance(base_optimizer, torch.optim.Optimizer):
            raise TypeError("base_optimizer must be a torch.optim.Optimizer instance")
        if rho < 0:
            raise ValueError(f"rho must be non-negative, got {rho}")
        if not 0.0 <= sparsity < 1.0:
            raise ValueError(f"sparsity must be in [0,1), got {sparsity}")
        
        self.base_optimizer = base_optimizer
        defaults = dict(rho=rho, sparsity=sparsity)
        super().__init__(params, defaults)
        
        # Share param groups with the base optimizer to keep lr schedulers/logging aligned
        self.param_groups = self.base_optimizer.param_groups
        self._init_masks()

    def _init_masks(self):
        """Create a fixed random mask for perturbation sparsity."""
        sparsity = self.defaults["sparsity"]
        for group in self.param_groups:
            group.setdefault("rho", self.defaults["rho"])
            group.setdefault("sparsity", sparsity)
            for p in group["params"]:
                mask = (torch.rand_like(p) > sparsity).float()
                mask = mask.to(p.device)
                mask.requires_grad = False
                self.state[p]["mask"] = mask

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def _grad_norm(self):
        norms = []
        shared_device = self.param_groups[0]["params"][0].device
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                masked_grad = p.grad * self.state[p]["mask"]
                norms.append(masked_grad.norm(p=2).to(shared_device))
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        if grad_norm == 0:
            return
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                e_w.data = e_w.data * self.state[p]["mask"]
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p].get("e_w", 0.0))
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        # Prefer explicit first/second steps in the training loop.
        if closure is None:
            raise RuntimeError("SparseSAM.step requires a closure.")
        loss = closure()
        return loss


def gradients_have_nan(model):
    """Utility to detect NaN/Inf in gradients."""
    for param in model.parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                return True
    return False


def forward_with_update_memory(model, inputs, update_memory, supports_update_memory):
    def normalize(output):
        if isinstance(output, (tuple, list)) and len(output) == 2:
            return output
        return output, torch.zeros((), device=inputs.device)
    if supports_update_memory is False:
        return normalize(model(inputs)), False
    if supports_update_memory is True:
        return normalize(model(inputs, update_memory=update_memory)), True
    try:
        return normalize(model(inputs, update_memory=update_memory)), True
    except TypeError as exc:
        if "update_memory" in str(exc):
            return normalize(model(inputs)), False
        raise


def train_rml2016_dtnet_memory(
    data_path='RML2016.10b.dat',
    dataset_type='rml2016',  # 'rml2016' or 'rml2018a'
    batch_size=256,
    epochs=200,
    learning_rate=2e-3,
    warmup_epochs=5,
    warmup_init_lr=1e-5,
    weight_decay=5e-5,
    optimizer_type='sparse_sam',  # 'adam' or 'sparse_sam'
    sam_rho=0.05,
    sam_sparsity=0.5,
    num_classes=None,  # Auto-detect from data if None
    use_memory=True,
    checkpoint_dir='checkpoint',
    save_interval=20,
    patience=20,
    use_wandb=True,
    wandb_project='rml2016-dtnet-memory',
    wandb_name="sparse_sam",
    num_samples=None  # Optional limit on number of samples to load
):
    """
    Train RML2016/RML2018A DTNet with Titans-style memory
    
    Args:
        data_path: Path to dataset file (RML2016.10b.dat) or directory (RML2018A)
        dataset_type: 'rml2016' for pickle format or 'rml2018a' for HDF5 format
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        warmup_epochs: Number of warmup epochs
        warmup_init_lr: Initial LR for warmup
        weight_decay: Weight decay for optimizer
        num_classes: Number of modulation classes (auto-detect if None)
        use_memory: Whether to use Titans-style memory
        checkpoint_dir: Directory to save checkpoints
        save_interval: Save checkpoint every N epochs
        patience: Early stopping patience
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: W&B project name
        wandb_name: W&B run name (None for auto-generated)
        num_samples: Optional limit on number of samples to load
        optimizer_type: Optimizer to use ('adam' or 'sparse_sam')
        sam_rho: Perturbation intensity for Sparse SAM
        sam_sparsity: Fraction of parameters skipped in SAM perturbation (0-1)
    """
    
    # Note: Memory config and wandb will be initialized after we know signal_length
    # This allows us to configure memory appropriately for the signal length
    
    # Load data first to detect signal_length
    print(f"Loading {dataset_type.upper()} data...")
    
    signal_length = 128  # Default for RML2016
    if dataset_type.lower() == 'rml2016':
        dataset_load = RML2016DataLoader(data_path)
        X, Z, lbl, snrs, mods = dataset_load.get_data()
        Y_full = None  # RML2016 doesn't have pre-computed one-hot labels
        signal_length = X.shape[1]  # Detect signal length
    elif dataset_type.lower() == 'rml2018a':
        dataset_load = RML2018ADataLoader(data_path)
        X, Z, lbl, snrs, mods, Y_full, signal_length = dataset_load.get_data(num_samples=num_samples)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Must be 'rml2016' or 'rml2018a'")
    
    print(f"Total samples: {X.shape[0]}")
    print(f"Signal shape: {X.shape[1:]}")
    print(f"Signal length: {signal_length} samples")
    print(f"Modulations: {len(mods)} classes")
    print(f"SNRs: {snrs}")
    
    # Auto-detect num_classes if not specified
    if num_classes is None:
        if Y_full is not None:
            num_classes = Y_full.shape[1]
        else:
            num_classes = len(mods)
        print(f"Auto-detected num_classes: {num_classes}")
    
    # Configure memory for the detected signal length
    # Longer sequences (1024) benefit from larger chunk sizes for better memory accumulation
    if use_memory:
        model_config.memory["use_memory"] = True
        # Adjust chunk size based on signal length
        # For 1024-length: use larger chunks (16-32) to leverage long-sequence memory
        # For 128-length: use smaller chunks (4-8) for fine-grained patterns
        
        model_config.memory["momentum"] = True  # Momentum maintains information across noisy samples
        model_config.memory["weight_decay"] = True  # Controlled forgetting
        model_config.memory["low_snr_mode"] = True
        print(f"  - Signal length: {signal_length}")
        print(f"  - Chunk size: {model_config.memory['chunk_size']}")
        print(f"  - Momentum: {model_config.memory['momentum']}")
        print(f"  - Weight decay: {model_config.memory['weight_decay']}")
        print(f"  - Long sequences allow memory to accumulate more information across time")
    
    # Initialize wandb (after we know signal_length and memory config)
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            entity="daothanh",
            config={
                'data_path': data_path,
                'batch_size': batch_size,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'warmup_epochs': warmup_epochs,
                'warmup_init_lr': warmup_init_lr,
                'weight_decay': weight_decay,
                'num_classes': num_classes,
                'use_memory': use_memory,
                'checkpoint_dir': checkpoint_dir,
                'save_interval': save_interval,
                'patience': patience,
                'model': 'MT_MCNet',
                'dataset': dataset_type.upper(),
                'dataset_path': data_path,
                'memory_chunk_size': model_config.memory.get("chunk_size", 8),
                'signal_length': signal_length,
                'memory_momentum': model_config.memory.get("momentum", True),
                'memory_weight_decay': model_config.memory.get("weight_decay", True),
                'memory_low_snr_mode': model_config.memory.get("low_snr_mode", False),
                'label_smoothing': 0.1,
                'optimizer_type': optimizer_type,
                'sam_rho': sam_rho,
                'sam_sparsity': sam_sparsity,
            }
        )
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if use_wandb:
        wandb.config.update({'device': str(device)})
    
    # Split data
    np.random.seed(42)
    n_examples = int(X.shape[0])
    n_train = int(n_examples * 0.7)
    n_val = int(n_examples * 0.15)
    n_test = n_examples - n_train - n_val
    
    train_idx = np.random.choice(range(n_examples), size=n_train, replace=False)
    remaining_idx = list(set(range(n_examples)) - set(train_idx))
    val_idx = np.random.choice(remaining_idx, size=n_val, replace=False)
    test_idx = list(set(remaining_idx) - set(val_idx))
    
    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]
    Z_train = Z[train_idx]
    Z_val = Z[val_idx]
    Z_test = Z[test_idx]
    
    # Convert labels to one-hot
    if Y_full is not None:
        # RML2018A: Use pre-computed one-hot labels
        Y_train = Y_full[train_idx]
        Y_val = Y_full[val_idx]
        Y_test = Y_full[test_idx]
    else:
        # RML2016: Convert from label tuples to one-hot
        Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
        Y_val = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), val_idx)))
        Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))
    
    # Reshape for model: [N, 1, signal_length, 2]
    X_train = np.reshape(X_train, (-1, 1, signal_length, 2))
    X_val = np.reshape(X_val, (-1, 1, signal_length, 2))
    X_test = np.reshape(X_test, (-1, 1, signal_length, 2))
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    if use_wandb:
        wandb.config.update({
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'num_modulations': len(mods),
            'num_snrs': len(snrs)
        })
    
    # Create datasets
    X_train_tensor = torch.Tensor(X_train)
    Y_train_tensor = torch.Tensor(Y_train)
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Model - use detected signal length
    print(f"\nInitializing MT_MCNet with memory={use_memory}...")
    print(f"Using signal_size=({signal_length}, 2) for model initialization")
    #model = DTNet(signal_size=(signal_length, 2), num_classes=num_classes, use_memory=use_memory)
    model = MT_MCNet(num_classes=num_classes)
    
    # Use DataParallel if multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
        if use_wandb:
            wandb.config.update({'num_gpus': torch.cuda.device_count()})
    
    model = model.to(device)
    
    # Loss and optimizer
    # For low SNR, we can use label smoothing to help with noisy predictions
    criterion = nn.CrossEntropyLoss().to(device)
    
    if optimizer_type.lower() == 'sparse_sam':
        base_optimizer = optim.Adam([p for n, p in model.named_parameters() if "memory.M" not in n], lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999),
    eps=1e-10)
        optimizer = SparseSAM([p for n, p in model.named_parameters() if "memory.M" not in n], base_optimizer=base_optimizer, rho=sam_rho, sparsity=sam_sparsity)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        print(f"Using Sparse SAM optimizer (rho={sam_rho}, sparsity={sam_sparsity}) with Adam base.")
    else:
        optimizer = optim.Adam([p for n, p in model.named_parameters() if "memory.M" not in n], lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999),
    eps=1e-10)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        print("Using Adam optimizer.")
    
    # Learning rate scheduler for better low SNR convergence
    scheduler_target = optimizer.base_optimizer if isinstance(optimizer, SparseSAM) else optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        scheduler_target, mode='min', factor=0.7, patience=3, verbose=True, min_lr=1e-6
    )
    
    # Training history
    best_accuracy = 0
    best_loss = float('inf')
    wait = 0
    val_loss_average = []
    accuracy_values_average = []
    accuracy_train_average = []
    training_loss = []
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)
    
    for epoch in range(epochs):
        # Warmup LR
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_lr = warmup_init_lr + (learning_rate - warmup_init_lr) * ((epoch + 1) / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr
        print(f'\nEpoch {epoch+1}/{epochs}')
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # ========== Training ==========
        model.train()
        overall_accuracy_total = 0
        train_loss = 0
        running_loss = 0.0
        successful_batches = 0  # Track number of successful batches (not skipped)
        
        pre_Y_train = None
        Y_train_snr_convert = None
        
        progress = tqdm(trainloader, desc=f"Training")
        supports_update_memory = None
        for i, data in enumerate(progress):
            inputs, labels_original = data
            inputs = inputs.to(device)
            labels = labels_original.to(device)
            
            # Forward pass
            try:
                (outputs, memory_loss), supports_update_memory = forward_with_update_memory(
                    model, inputs, update_memory=True, supports_update_memory=supports_update_memory
                )
                if torch.is_tensor(memory_loss):
                    memory_loss = memory_loss.mean()
                
                # Check for NaN in outputs before computing loss
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"\nWarning: NaN/Inf detected in model outputs at batch {i}")
                    print(f"  outputs shape: {outputs.shape}")
                    print(f"  outputs dtype: {outputs.dtype}")
                    print(f"  outputs has NaN: {torch.isnan(outputs).any().item()}")
                    print(f"  outputs has Inf: {torch.isinf(outputs).any().item()}")
                    print(f"  outputs sample (first 3): {outputs[:3] if outputs.shape[0] >= 3 else outputs}")
                    print(f"  inputs shape: {inputs.shape}")
                    print(f"  inputs has NaN: {torch.isnan(inputs).any().item()}")
                    print(f"  inputs has Inf: {torch.isinf(inputs).any().item()}")
                    optimizer.zero_grad()
                    continue
                
                labels_argmax = torch.argmax(labels, dim=1)
                
                # Validate labels_argmax
                if labels_argmax.max() >= num_classes or labels_argmax.min() < 0:
                    print(f"\nWarning: Invalid labels_argmax at batch {i}")
                    print(f"  labels_argmax: {labels_argmax}")
                    print(f"  labels_argmax min/max: {labels_argmax.min().item()}/{labels_argmax.max().item()}")
                    print(f"  num_classes: {num_classes}")
                    print(f"  labels shape: {labels.shape}")
                    print(f"  labels sample: {labels[:3] if labels.shape[0] >= 3 else labels}")
                    optimizer.zero_grad()
                    continue
                
                # Loss
                loss = criterion(outputs, labels_argmax) + 0.1 * memory_loss
                
            except Exception as e:
                print(f"Warning: Exception during forward pass at batch {i}: {e}")
                import traceback
                traceback.print_exc()
                optimizer.zero_grad()
                continue
            
            loss_to_log = None
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent NaN/exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if gradients_have_nan(model):
                print(f"Warning: NaN/Inf detected in gradients at batch {i}, skipping update")
                optimizer.zero_grad()
                continue
            
            if isinstance(optimizer, SparseSAM):
                # First SAM step
                optimizer.first_step(zero_grad=True)
                
                # Second forward/backward on perturbed weights
                (outputs_second, memory_loss), supports_update_memory = forward_with_update_memory(
                    model, inputs, update_memory=False, supports_update_memory=supports_update_memory
                )
                if torch.is_tensor(memory_loss):
                    memory_loss = memory_loss.mean()
                if torch.isnan(outputs_second).any() or torch.isinf(outputs_second).any():
                    print(f"Warning: NaN/Inf detected in SAM second forward at batch {i}, skipping update")
                    optimizer.zero_grad()
                    continue
                loss_second = criterion(outputs_second, labels_argmax) + 0.1 * memory_loss
                loss_second.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                if gradients_have_nan(model):
                    print(f"Warning: NaN/Inf detected in gradients (SAM second step) at batch {i}, skipping update")
                    optimizer.zero_grad()
                    continue
                
                optimizer.second_step(zero_grad=True)
                # Apply memory update after backward to avoid in-place autograd issues
                mem_model = model.module if isinstance(model, nn.DataParallel) else model
                if model.training and hasattr(mem_model, "apply_memory_update"):
                    mem_model.apply_memory_update()
                loss_to_log = (loss.item() + loss_second.item()) / 2.0
            else:
                optimizer.step()
                # Apply memory update after backward to avoid in-place autograd issues
                mem_model = model.module if isinstance(model, nn.DataParallel) else model
                if model.training and hasattr(mem_model, "apply_memory_update"):
                    mem_model.apply_memory_update()
                loss_to_log = loss.item()
            
            # Only count successful batches
            successful_batches += 1
            running_loss += loss_to_log
            train_loss += loss_to_log
            
            # Collect predictions (only for successful batches)
            outputs_np = outputs.detach().cpu().numpy()
            labels_np = labels_original.numpy()
            
            # Ensure consistent shapes
            if outputs_np.shape[1] != num_classes:
                print(f"Warning: Output shape mismatch at batch {i}: {outputs_np.shape}, expected {num_classes} classes")
                continue
            
            if pre_Y_train is None:
                pre_Y_train = outputs_np
                Y_train_snr_convert = labels_np
            else:
                # Check shapes match before vstack
                if pre_Y_train.shape[1] == outputs_np.shape[1]:
                    pre_Y_train = np.vstack((pre_Y_train, outputs_np))
                    Y_train_snr_convert = np.vstack((Y_train_snr_convert, labels_np))
                else:
                    print(f"Warning: Shape mismatch at batch {i}, skipping prediction collection")
            
            # Calculate running average only over successful batches
            if successful_batches > 0:
                progress.set_postfix({"loss": f"{running_loss / successful_batches:.4f}"})
            else:
                progress.set_postfix({"loss": "N/A"})
            
            # Log batch metrics to wandb
            if use_wandb and (i + 1) % 50 == 0:
                wandb.log({
                    'train/batch_loss': loss_to_log,
                    'train/learning_rate': optimizer.param_groups[0]['lr'],
                    'train/epoch_progress': (i + 1) / len(trainloader)
                }, step=epoch * len(trainloader) + i)

        
        # Calculate training accuracy
        if pre_Y_train is not None and pre_Y_train.shape[0] > 0:
            conf_train = np.zeros([len(mods), len(mods)])
            for i in range(0, pre_Y_train.shape[0]):
                try:
                    j = list(Y_train_snr_convert[i, :]).index(1)
                    k = int(np.argmax(pre_Y_train[i, :]))
                    conf_train[j, k] = conf_train[j, k] + 1
                except (ValueError, IndexError) as e:
                    continue
        else:
            print("Warning: No valid predictions collected for training accuracy calculation")
            conf_train = np.zeros([len(mods), len(mods)])
        
        cor_train = np.sum(np.diag(conf_train))
        ncor_train = np.sum(conf_train) - cor_train
        train_acc = cor_train / (cor_train + ncor_train)
        accuracy_train_average.append(train_acc)
        # Calculate average loss only over successful batches
        if successful_batches > 0:
            training_loss.append(train_loss / successful_batches)
        else:
            training_loss.append(0.0)
            print("Warning: No successful batches in this epoch!")
        
        print(f"Train Loss: {training_loss[-1]:.4f}, Train Acc: {train_acc:.4f}, Successful Batches: {successful_batches}/{len(trainloader)}")
        
        # Update learning rate scheduler based on validation loss
        # (will be updated after validation)
        
        # Log training metrics to wandb
        if use_wandb:
            wandb.log({
                'train/epoch_loss': training_loss[-1],
                'train/epoch_accuracy': train_acc,
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'epoch': epoch + 1
            })
        
        # ========== Validation ==========
        model.eval()
        Z_val_flat = Z_val.reshape((len(Z_val)))
        SNRs = np.unique(Z_val_flat)
        
        loss_values = {snr: [] for snr in SNRs}
        accuracy_values = {snr: [] for snr in SNRs}
        
        with torch.no_grad():
            for snr in SNRs:
                X_val_snr = X_val[Z_val_flat == snr]
                Y_val_snr = Y_val[Z_val_flat == snr]
                X_val_snr_tensor = torch.Tensor(X_val_snr)
                Y_val_snr_tensor = torch.Tensor(Y_val_snr).long()
                val_dataset = TensorDataset(X_val_snr_tensor, Y_val_snr_tensor)
                valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                
                val_loss = 0
                pre_Y_val = None
                Y_val_snr_convert = None
                
                for i, data in enumerate(valloader):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    (outputs, memory_loss), supports_update_memory = forward_with_update_memory(
                        model, inputs, update_memory=False, supports_update_memory=supports_update_memory
                    )
                    if torch.is_tensor(memory_loss):
                        memory_loss = memory_loss.mean()
                    labels_loss = torch.argmax(labels, dim=1).long()
                    loss = criterion(outputs, labels_loss) + 0.1 * memory_loss
                    val_loss += loss.item()
                    
                    outputs_np = outputs.detach().cpu().numpy()
                    labels_np = labels.detach().cpu().numpy()
                    
                    # Ensure consistent shapes
                    if outputs_np.shape[1] != num_classes:
                        continue
                    
                    if pre_Y_val is None:
                        pre_Y_val = outputs_np
                        Y_val_snr_convert = labels_np
                    else:
                        # Check shapes match before vstack
                        if pre_Y_val.shape[1] == outputs_np.shape[1]:
                            pre_Y_val = np.vstack((pre_Y_val, outputs_np))
                            Y_val_snr_convert = np.vstack((Y_val_snr_convert, labels_np))
                        else:
                            continue
                
                # Calculate accuracy for this SNR
                conf = np.zeros([len(mods), len(mods)])
                for i in range(0, X_val_snr.shape[0]):
                    j = list(Y_val_snr_convert[i, :]).index(1)
                    k = int(np.argmax(pre_Y_val[i, :]))
                    conf[j, k] = conf[j, k] + 1
                
                cor = np.sum(np.diag(conf))
                ncor = np.sum(conf) - cor
                overall_accuracy = cor / (cor + ncor)
                
                loss_values[snr].append(val_loss / len(valloader))
                accuracy_values[snr].append(overall_accuracy)
                print(f"SNR {snr:3d}dB: Loss={val_loss/len(valloader):.4f}, Acc={overall_accuracy:.4f}")
                
                # Log per-SNR metrics to wandb
                if use_wandb:
                    wandb.log({
                        f'val/snr_{snr}dB_loss': val_loss / len(valloader),
                        f'val/snr_{snr}dB_accuracy': overall_accuracy
                    })
        
        # Average validation metrics
        average_loss = sum(loss_values[snr][-1] for snr in SNRs) / len(SNRs)
        average_accuracy = sum(accuracy_values[snr][-1] for snr in SNRs) / len(SNRs)
        
        val_loss_average.append(average_loss)
        accuracy_values_average.append(average_accuracy)
        
        print(f"Val Loss: {average_loss:.4f}, Val Acc: {average_accuracy:.4f}")
        
        # Update learning rate scheduler (after warmup)
        if epoch + 1 > warmup_epochs:
            scheduler.step(average_loss)
        
        # Log validation metrics to wandb
        if use_wandb:
            wandb.log({
                'val/epoch_loss': average_loss,
                'val/epoch_accuracy': average_accuracy,
                'train/learning_rate': optimizer.param_groups[0]['lr'],
                'epoch': epoch + 1
            })
        
        # Save checkpoint
        if epoch % save_interval == 0 or epoch == epochs - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f'DTNet_memory_epoch_64_2018{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            
            # Log checkpoint to wandb
            if use_wandb:
                wandb.save(checkpoint_path)
        
        # Early stopping and learning rate scheduling
        if average_loss < best_loss:
            best_loss = average_loss
            best_epoch = epoch + 1
            wait = 0
            # Save best model
            best_model_path = os.path.join(checkpoint_dir, 'DTNet_memory_best_64_2018.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved (loss: {best_loss:.4f})")
            
            # Log best model to wandb
            if use_wandb:
                wandb.save(best_model_path)
                wandb.run.summary['best_val_loss'] = best_loss
                wandb.run.summary['best_val_loss_epoch'] = best_epoch
        else:
            wait += 1
            # Learning rate is handled by scheduler, but we can log it
            if use_wandb:
                wandb.log({'train/lr_decay': optimizer.param_groups[0]["lr"]})
            
            if wait >= patience:
                print(f'\nEarly stopping at epoch {epoch+1} due to no improvement in validation loss.')
                print(f'Best epoch: {best_epoch}, Best loss: {best_loss:.4f}')
                if use_wandb:
                    wandb.log({'early_stopped': True, 'stopped_at_epoch': epoch + 1})
                break
        
        if average_accuracy > best_accuracy:
            best_accuracy = average_accuracy
            best_epoch_accuracy = epoch + 1
            if use_wandb:
                wandb.run.summary['best_val_accuracy'] = best_accuracy
                wandb.run.summary['best_val_accuracy_epoch'] = best_epoch_accuracy
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best accuracy: {best_accuracy:.4f} at epoch {best_epoch_accuracy}")
    print(f"Best loss: {best_loss:.4f} at epoch {best_epoch}")
    print("="*60)
    
    # Final summary to wandb
    if use_wandb:
        wandb.run.summary.update({
            'final_train_loss': training_loss[-1],
            'final_val_loss': val_loss_average[-1],
            'final_train_accuracy': accuracy_train_average[-1],
            'final_val_accuracy': accuracy_values_average[-1],
            'total_epochs': len(training_loss)
        })
        wandb.finish()
    
    return {
        'training_loss': training_loss,
        'val_loss': val_loss_average,
        'train_acc': accuracy_train_average,
        'val_acc': accuracy_values_average,
        'best_accuracy': best_accuracy,
        'best_epoch': best_epoch
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train RML2016 DTNet with Titans-style Memory')
    parser.add_argument('--data_path', type=str, default='/home/pnu_r/AMC/2018/GOLD_XYZ_OSC.0001_1024.hdf5', #/home/pnu_r/AMC/RML2016.10b.dat, /home/pnu_r/AMC/RML2016.10a_dict.pkl, /home/pnu_r/AMC/2018/GOLD_XYZ_OSC.0001_1024.hdf5
                        help='Path to dataset file (RML2016.10b.dat) or directory (RML2018A)')
    parser.add_argument('--dataset_type', type=str, default='rml2018a', choices=['rml2016', 'rml2018a'],
                        help='Dataset type: rml2016 (pickle) or rml2018a (HDF5)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Optional limit on number of samples to load (None for all)')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--warmup_init_lr', type=float, default=1e-5,
                        help='Warmup initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--num_classes', type=int, default=24,
                        help='Number of modulation classes')
    parser.add_argument('--use_memory', action='store_true', default=True,
                        help='Use Titans-style memory')
    parser.add_argument('--no_memory', dest='use_memory', action='store_false',
                        help='Disable memory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_interval', type=int, default=20,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--use_wandb', action='store_true', default=True,
                        help='Use Weights & Biases logging')
    parser.add_argument('--no_wandb', dest='use_wandb', action='store_false',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='rml2016-dtnet-memory',
                        help='W&B project name')
    parser.add_argument('--wandb_name', type=str, default="Adam",
                        help='W&B run name (None for auto-generated)')
    
    args = parser.parse_args()
    
    train_rml2016_dtnet_memory(
        data_path=args.data_path,
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        warmup_epochs=args.warmup_epochs,
        warmup_init_lr=args.warmup_init_lr,
        weight_decay=args.weight_decay,
        num_classes=args.num_classes,
        use_memory=args.use_memory,
        checkpoint_dir=args.checkpoint_dir,
        save_interval=args.save_interval,
        patience=args.patience,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        num_samples=args.num_samples
    )




