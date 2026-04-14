"""
ML for Health 2026 -- Assignment 1: Brain Tumor Classification
==============================================================
Dataset: Kaggle Brain Tumor MRI Classification
Classes: glioma | meningioma | pituitary | no_tumor

Instructions
------------
- Complete every function marked with TODO.
- Do NOT change function signatures.
- Run `pytest tests/ -v` locally before pushing to main.
- Write any questions or issues in experiences.md.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


# ---------------------------------------------------------------------------
# Reproducibility helper (do not modify)
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===========================================================================
# Exercise 3 -- Metrics
# ===========================================================================

def compute_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Compute precision, recall, and F1-score for a single class.

    Parameters
    ----------
    tp : int  -- true positives
    fp : int  -- false positives
    fn : int  -- false negatives

    Returns
    -------
    (precision, recall, f1) as floats rounded to 4 decimal places.

    Example (glioma from Exercise 3):
        compute_metrics(42, 8, 18) -> (0.84, 0.7, 0.7636)
    """
    # TODO 3a: compute precision  =  TP / (TP + FP)
    # TODO 3b: compute recall     =  TP / (TP + FN)
    # TODO 3c: compute F1         =  2 * precision * recall / (precision + recall)
    raise NotImplementedError


# ===========================================================================
# Exercise 5 -- Class Imbalance
# ===========================================================================

def compute_class_weights(class_counts: dict[str, int]) -> dict[str, float]:
    """Compute per-class weights using the formula  w_c = N / (K * n_c).

    Parameters
    ----------
    class_counts : dict mapping class name -> number of training samples
        Example: {"glioma": 120, "meningioma": 240,
                  "pituitary": 300, "no_tumor": 840}

    Returns
    -------
    dict mapping class name -> weight (rounded to 4 decimal places).

    Example (Exercise 5):
        compute_class_weights({"glioma": 120, "meningioma": 240,
                               "pituitary": 300, "no_tumor": 840})
        -> {"glioma": 3.125, "meningioma": 1.5625,
            "pituitary": 1.25,  "no_tumor": 0.4464}
    """
    # TODO 5a: N = total samples across all classes
    # TODO 5a: K = number of classes
    # TODO 5a: for each class c:  w_c = N / (K * n_c)
    raise NotImplementedError


# ===========================================================================
# Exercise 2 -- Data loading
# ===========================================================================

def build_dataloaders(
    train_root: Path,
    test_root: Path,
    val_fraction: float = 0.2,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test DataLoaders.

    The validation set is carved out of the training folder using a
    random index-level split (fixed seed for reproducibility).
    Augmentation (RandomHorizontalFlip) is applied only to the training set.

    Parameters
    ----------
    train_root    : path to the Training/ folder (contains one sub-folder per class)
    test_root     : path to the Testing/ folder
    val_fraction  : fraction of training images to use for validation
    batch_size    : mini-batch size
    seed          : random seed for the split

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # TODO 2a: load the full training folder twice using datasets.ImageFolder --
    #          once with train_tfms (for training) and once with eval_tfms (for val)
    # TODO 2a: load the test folder with eval_tfms

    # TODO 2b: split training indices into train/val using torch.randperm with the
    #          given seed; val gets the first int(val_fraction * n_total) indices

    # TODO 2c: wrap with DataLoader (shuffle=True for train, False for val/test)

    raise NotImplementedError


# ===========================================================================
# Exercise 2 -- Simple CNN baseline
# ===========================================================================

class SimpleCNN(nn.Module):
    """A minimal CNN with 2 convolutional blocks and a linear classifier head.

    Architecture (to implement):
        Block 1: Conv2d(3, 16, 3, padding=1) -> ReLU -> MaxPool2d(2)
        Block 2: Conv2d(16, 32, 3, padding=1) -> ReLU -> MaxPool2d(2)
        Head   : AdaptiveAvgPool2d(1) -> Flatten -> Linear(32, num_classes)
    """

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        # TODO 2b: define self.block1, self.block2, self.head using nn.Sequential
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 2b: pass x through block1, block2, head
        raise NotImplementedError


# ===========================================================================
# Exercise 2 -- Pretrained ResNet18 (scaffolded for you)
# ===========================================================================

def build_resnet18(num_classes: int = 4) -> nn.Module:
    """Load a pretrained ResNet18 and replace the final FC layer."""
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ===========================================================================
# Exercise 2 -- Training and evaluation
# ===========================================================================

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Run one epoch of training or evaluation.

    Parameters
    ----------
    model     : the neural network
    loader    : DataLoader for this split
    criterion : loss function
    optimizer : pass an optimizer for training, None for evaluation
    device    : torch device

    Returns
    -------
    (avg_loss, avg_accuracy, all_targets, all_preds)
    - avg_loss     : mean loss over all samples
    - avg_accuracy : fraction of correct predictions
    - all_targets  : numpy array of ground-truth labels
    - all_preds    : numpy array of predicted labels
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        # TODO 2c: forward pass, compute loss, get predictions (argmax)
        # TODO 2c: if training, call loss.backward() and optimizer.step()
        # TODO 2c: accumulate total_loss, total_correct, total_samples
        # TODO 2c: extend all_preds and all_targets with detached CPU values

        raise NotImplementedError

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc, np.array(all_targets), np.array(all_preds)
