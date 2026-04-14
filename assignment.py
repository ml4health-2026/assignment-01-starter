"""
ML for Health 2026 -- Assignment 1: Brain Tumor MRI Classification
===================================================================
Dataset : Kaggle Brain Tumor MRI Classification (4 classes)
Classes : glioma | meningioma | pituitary | no_tumor

Instructions
------------
- Complete every function marked with TODO. Do NOT change function signatures.
- Run `pytest tests/ -v` locally to check your work before pushing.
- Push to `main` only for final submission; save work-in-progress on a branch.
- Record issues and reflections in experiences.md.
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
# Reproducibility  (do not modify)
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===========================================================================
# Exercise 3 -- Evaluation Metrics
# ===========================================================================

def compute_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Compute precision, recall, and F1-score for a single class (one-vs-rest).

    Parameters
    ----------
    tp : true positives
    fp : false positives
    fn : false negatives

    Returns
    -------
    (precision, recall, f1)  -- each rounded to 4 decimal places.

    """
    # TODO: compute and return (precision, recall, f1), each rounded to 4 decimal places
    raise NotImplementedError


# ===========================================================================
# Exercise 5 -- Class Imbalance
# ===========================================================================

def compute_class_weights(class_counts: dict[str, int]) -> dict[str, float]:
    """Compute inverse-frequency class weights.

    Parameters
    ----------
    class_counts : dict mapping class name -> number of training samples.

    Returns
    -------
    dict mapping class name -> weight (rounded to 4 decimal places).

    """
    # TODO: compute and return the per-class weights, rounded to 4 decimal places
    raise NotImplementedError


# ===========================================================================
# Exercise 2a -- Data Loading
# ===========================================================================

def build_dataloaders(
    train_root: Path,
    test_root: Path,
    val_fraction: float = 0.2,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test DataLoaders.

    - The validation set is carved from the training folder with a fixed seed.
    - Data augmentation (RandomHorizontalFlip) is applied only on the training set.
    - Validation and test sets use deterministic transforms only.

    Parameters
    ----------
    train_root   : path to the Training/ folder (one sub-folder per class)
    test_root    : path to the Testing/ folder
    val_fraction : fraction of training images reserved for validation
    batch_size   : mini-batch size for all loaders
    seed         : random seed for the train/val index split

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

    # TODO: load the datasets, split train into train/val using the seed, return three DataLoaders
    raise NotImplementedError


# ===========================================================================
# Exercise 2b -- Simple CNN Baseline
# ===========================================================================

class SimpleCNN(nn.Module):
    """Minimal 2-block CNN classifier.

    Architecture to implement
    -------------------------
    block1 : Conv2d(3→16, kernel=3, padding=1) → ReLU → MaxPool2d(2)
    block2 : Conv2d(16→32, kernel=3, padding=1) → ReLU → MaxPool2d(2)
    head   : AdaptiveAvgPool2d(1) → Flatten → Linear(32, num_classes)

    Use nn.Sequential for each block and the head.
    """

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        # TODO: define self.block1, self.block2, and self.head
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: forward pass through the three components
        raise NotImplementedError


# ===========================================================================
# Exercise 2b -- Pretrained ResNet18  (provided — no changes needed)
# ===========================================================================

def build_resnet18(num_classes: int = 4) -> nn.Module:
    """Return a ResNet18 pretrained on ImageNet with a new classification head."""
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ===========================================================================
# Exercise 2c -- Training and Evaluation Loop
# ===========================================================================

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Run one full pass over a DataLoader (training or evaluation).

    Pass an optimizer for training mode; pass None for evaluation mode.

    Parameters
    ----------
    model     : the neural network
    loader    : DataLoader for the split to process
    criterion : loss function (e.g. CrossEntropyLoss)
    optimizer : optimizer for training; None for validation / test
    device    : torch device (cpu or cuda)

    Returns
    -------
    (avg_loss, avg_accuracy, all_targets, all_preds)
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0
    all_preds:   list[int] = []
    all_targets: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        # TODO: forward pass, backward pass (if training), accumulate statistics

    # This guard fires if the loop body was not implemented.
    # Remove it (or it will go away naturally) once the TODOs above are filled in.
    if total_samples == 0:
        raise NotImplementedError("Implement the loop body above (TODOs 2c-i to 2c-iii).")

    avg_loss = total_loss / total_samples
    avg_acc  = total_correct / total_samples
    return avg_loss, avg_acc, np.array(all_targets), np.array(all_preds)
