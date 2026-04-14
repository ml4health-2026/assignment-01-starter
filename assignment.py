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

    Formulae
    --------
    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    F1        = 2 * precision * recall / (precision + recall)

    Parameters
    ----------
    tp : true positives
    fp : false positives
    fn : false negatives

    Returns
    -------
    (precision, recall, f1)  -- each rounded to 4 decimal places.

    Example
    -------
    >>> compute_metrics(42, 8, 18)
    (0.84, 0.7, 0.7636)
    """
    # TODO 3a: compute precision
    # TODO 3b: compute recall
    # TODO 3c: compute F1  (use the precision and recall you computed above)
    # TODO    : return (round(precision, 4), round(recall, 4), round(f1, 4))
    raise NotImplementedError


# ===========================================================================
# Exercise 5 -- Class Imbalance
# ===========================================================================

def compute_class_weights(class_counts: dict[str, int]) -> dict[str, float]:
    """Compute inverse-frequency class weights: w_c = N / (K * n_c).

    A higher weight means the model will be penalised more for mistakes
    on that class — useful when rare classes matter clinically.

    Parameters
    ----------
    class_counts : dict mapping class name -> number of training samples.
        Example: {"glioma": 120, "meningioma": 240,
                  "pituitary": 300, "no_tumor": 840}

    Returns
    -------
    dict mapping class name -> weight (rounded to 4 decimal places).

    Example
    -------
    >>> compute_class_weights({"glioma": 120, "meningioma": 240,
    ...                        "pituitary": 300, "no_tumor": 840})
    {"glioma": 3.125, "meningioma": 1.5625, "pituitary": 1.25, "no_tumor": 0.4464}
    """
    # TODO 5a: N = total number of samples (sum of all counts)
    # TODO 5b: K = number of classes
    # TODO 5c: for each class c, compute w_c = N / (K * n_c)
    # TODO    : return a dict with the same keys, values rounded to 4 decimal places
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

    # TODO 2a-i : load the training folder *twice* with datasets.ImageFolder --
    #             once with train_tfms (used for the train split)
    #             once with eval_tfms  (used for the val split, no augmentation)
    # TODO 2a-ii: load the test folder with eval_tfms

    # TODO 2a-iii: create a list of all indices, then split into train/val
    #              use torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))
    #              val gets the first int(val_fraction * n_total) indices

    # TODO 2a-iv : wrap each split in a Subset, then in a DataLoader
    #              shuffle=True for train, shuffle=False for val and test

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
        # TODO 2b-i: define self.block1 as nn.Sequential(...)
        # TODO 2b-ii: define self.block2 as nn.Sequential(...)
        # TODO 2b-iii: define self.head  as nn.Sequential(...)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 2b-iv: pass x through block1, then block2, then head; return result
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

        # TODO 2c-i  : forward pass
        #   outputs = model(images)          # shape: (batch, num_classes)
        #   loss    = criterion(outputs, labels)
        #   preds   = outputs.argmax(dim=1)  # predicted class index

        # TODO 2c-ii : backward pass (only when training)
        #   loss.backward()
        #   optimizer.step()

        # TODO 2c-iii: accumulate batch statistics
        #   total_loss    += loss.item() * images.size(0)
        #   total_correct += (preds == labels).sum().item()
        #   total_samples += images.size(0)
        #   all_preds.extend(preds.detach().cpu().tolist())
        #   all_targets.extend(labels.detach().cpu().tolist())

    # This guard fires if the loop body was not implemented.
    # Remove it (or it will go away naturally) once the TODOs above are filled in.
    if total_samples == 0:
        raise NotImplementedError("Implement the loop body above (TODOs 2c-i to 2c-iii).")

    avg_loss = total_loss / total_samples
    avg_acc  = total_correct / total_samples
    return avg_loss, avg_acc, np.array(all_targets), np.array(all_preds)
