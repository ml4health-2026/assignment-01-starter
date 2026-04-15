import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Assignment 1: Brain Tumor MRI Classification
    **ML for Health 2026**

    ---

    ### What you will do
    1. Frame the clinical ML problem and choose appropriate metrics
    2. Explore the dataset and think carefully about data splits
    3. Implement metric computation and class-weight calculation by hand
    4. Build a simple CNN baseline and a pretrained ResNet18
    5. Train, evaluate, and compare both models

    ### How to work
    - All coding tasks are in **`assignment.py`**. Implement the `TODO` sections there.
    - Run this notebook cell by cell to check your results as you go.
    - Written answers go directly in the markdown cells below (replace the *italicised placeholders*).
    - Run `pytest tests/ -v` in the terminal at any time to check the auto-graded parts.
    - Push to `main` only for **final submission**. Use a separate branch to save work in progress.
    - Record difficulties or open questions in `experiences.md`.

    ### Setup
    If you haven't yet downloaded the dataset:
    ```bash
    conda activate ml4health
    python download_data.py
    ```
    """)
    return


@app.cell
def _():
    # Environment setup — run this cell first
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    from torchvision import datasets, transforms
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

    from assignment import (
        set_seed,
        compute_metrics,
        compute_class_weights,
        build_dataloaders,
        SimpleCNN,
        build_resnet18,
        run_epoch,
    )

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATA_ROOT = Path('brain_tumor_data')
    print('Device:', device)
    return (
        ConfusionMatrixDisplay,
        DATA_ROOT,
        SimpleCNN,
        build_dataloaders,
        build_resnet18,
        classification_report,
        compute_class_weights,
        compute_metrics,
        confusion_matrix,
        datasets,
        device,
        nn,
        plt,
        run_epoch,
        torch,
        transforms,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 1 — Understanding the Problem

    ### Exercise 1: Task Framing

    Before touching any code, make sure you understand what you are building and why.
    The dataset contains brain MRI images labelled with one of four classes:
    $$\{\text{glioma},\ \text{meningioma},\ \text{pituitary},\ \text{no\_tumor}\}$$

    **1a.** Formally define the supervised learning problem: what is the input $x$, the output $y$, and the prediction task?

    > *Your answer.*

    **1b.** Propose one **primary metric** and one **secondary metric** for evaluating this classifier. Justify each choice briefly — think about what a clinician would care about.

    > *Your answer.*

    **1c.** Why can plain accuracy be misleading when class counts are unequal?

    > *Your answer.*

    **1d.** Give one concrete example of **data leakage** that could make results look better than they really are in a medical imaging dataset.

    > *Your answer.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 2 — Exploring the Data

    Run the cells below to get familiar with the dataset before building anything.
    No implementation needed here.
    """)
    return


@app.cell
def _(DATA_ROOT, datasets, transforms):
    # Load training and test folders (no augmentation, just for inspection)
    inspect_tfms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_full = datasets.ImageFolder(DATA_ROOT / 'Training', transform=inspect_tfms)
    test_full  = datasets.ImageFolder(DATA_ROOT / 'Testing',  transform=inspect_tfms)

    class_names = train_full.classes
    print('Classes     :', class_names)
    print('Train total :', len(train_full))
    print('Test  total :', len(test_full))

    # Count images per class in the training set
    from collections import Counter
    train_counts = Counter(label for _, label in train_full.samples)
    print('\nTraining class counts:')
    for idx, name in enumerate(class_names):
        print(f'  {name:>12} : {train_counts[idx]}')
    return class_names, train_counts, train_full


@app.cell
def _(class_names, plt, train_full):
    # Visualise a few samples from each class
    _fig, _axes = plt.subplots(len(class_names), 4, figsize=(10, 3 * len(class_names)))
    for row, (cls_idx, cls_name) in enumerate(enumerate(class_names)):
        samples = [(img, lbl) for img, lbl in train_full if lbl == cls_idx][:4]
        for col, (img, _) in enumerate(samples):
            _axes[row, col].imshow(img.permute(1, 2, 0).numpy())
            _axes[row, col].axis('off')
            if col == 0:
                _axes[row, col].set_title(cls_name, fontsize=11, fontweight='bold')
    plt.suptitle('Sample images per class', fontsize=13)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 2: Data Splits and Leakage

    Imagine you have MRI slices from 250 patients, each with multiple images.

    **2a.** Why is a random image-level split (ignoring which patient each image belongs to) risky?

    > *Your answer.*

    **2b.** Propose a patient-level train / validation / test split ratio and briefly justify it.

    > *Your answer.*

    **2c.** Should data augmentation (flips, rotations, …) be applied to training, validation, or test data? Why?

    > *Your answer.*

    **2d.** Why should the held-out test set remain untouched until the very end of the project?

    > *Your answer.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 3 — Evaluation Metrics

    ### Exercise 3: Implement `compute_metrics` in `assignment.py`

    Assume a hypothetical classifier was evaluated on a test set. For the class **glioma**,
    in a one-vs-rest setting, the confusion counts came out as:

    $$TP = 42, \quad FP = 8, \quad FN = 18$$

    **Your tasks:**
    1. Implement `compute_metrics(tp, fp, fn)` in `assignment.py` to compute precision, recall, and F1-score.
    2. Run the cell below to verify your results.
    3. Fill in the TODO in the code cell: based on your recall value, how many percent of true glioma cases does this hypothetical model miss?
    """)
    return


@app.cell
def _(compute_metrics):
    precision, recall, f1 = compute_metrics(tp=42, fp=8, fn=18)

    print(f'Precision : {precision:.4f}')
    print(f'Recall    : {recall:.4f}')
    print(f'F1-score  : {f1:.4f}')

    # TODO: using the recall value above, compute the percentage of true glioma cases missed
    missed_pct = None  # replace with your calculation
    print(f'Missed glioma cases: {missed_pct:.1f}%')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 4 — Handling Class Imbalance

    ### Exercise 4: Implement `compute_class_weights` in `assignment.py`

    Look back at the class distribution you printed in Part 2. The training set is not balanced —
    some tumor types appear far more often than others. If we train a classifier on this data as-is,
    it will be biased towards the majority class and may systematically fail on the rarer ones,
    which are often the most clinically important.

    One standard remedy is to assign a higher loss weight to underrepresented classes during training,
    so that mistakes on rare classes are penalised more heavily.

    Implement `compute_class_weights(class_counts)` in `assignment.py` to compute such weights,
    then run the cell below.
    """)
    return


@app.cell
def _(class_names, compute_class_weights, train_counts):
    # Build the counts dict from the actual dataset
    actual_counts = {class_names[idx]: count for idx, count in sorted(train_counts.items())}
    print('Class counts:', actual_counts)

    weights = compute_class_weights(actual_counts)
    print('\nClass weights:')
    for cls, w in weights.items():
        print(f'  {cls:>12} : {w:.4f}')

    rarest = max(weights, key=weights.get)
    print(f'\nHighest-weighted (rarest) class: {rarest}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 5 — Building the Pipeline

    ### Exercise 5a: Implement `build_dataloaders` in `assignment.py`

    Build train, validation, and test DataLoaders.
    The validation set is carved from the training folder using a fixed random seed.
    Augmentation applies only to training.
    """)
    return


@app.cell
def _(DATA_ROOT, build_dataloaders):
    train_loader, val_loader, test_loader = build_dataloaders(
        train_root=DATA_ROOT / 'Training',
        test_root=DATA_ROOT / 'Testing',
        val_fraction=0.2,
        batch_size=32,
        seed=42,
    )

    print(f'Train batches : {len(train_loader)}')
    print(f'Val   batches : {len(val_loader)}')
    print(f'Test  batches : {len(test_loader)}')

    # Sanity-check: one batch should have shape (32, 3, 224, 224)
    images, labels = next(iter(train_loader))
    print(f'Batch shape   : {images.shape}  labels shape: {labels.shape}')
    return train_loader, val_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 5b: Implement `SimpleCNN` in `assignment.py`

    Build a minimal 2-block convolutional network.
    The architecture is fully specified in the docstring — implement it using `nn.Sequential`.
    """)
    return


@app.cell
def _(SimpleCNN, device, torch):
    cnn = SimpleCNN(num_classes=4).to(device)

    # Quick shape check — should print torch.Size([2, 4])
    dummy = torch.zeros(2, 3, 224, 224).to(device)
    with torch.no_grad():
        out = cnn(dummy)
    print('Output shape:', out.shape)
    return (cnn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 5c: Implement `run_epoch` in `assignment.py`

    Implement the training / evaluation loop.
    The function handles both modes depending on whether an optimizer is passed.
    Read the docstring and the TODO comment carefully before starting.

    > **Note:** SimpleCNN is an intentionally minimal baseline — two conv blocks with random initialisation.
    > Do not expect strong results. Its purpose is to show what a naive classifier achieves before
    > we bring in pretrained weights. You will compare both models in Exercise 6.
    """)
    return


@app.cell
def _(cnn, device, nn, run_epoch, torch, train_loader, val_loader):
    # Train SimpleCNN for 20 epochs
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
    num_epochs = 20
    history_cnn = []
    for _epoch in range(num_epochs):
        _tr_loss, _tr_acc, _, _ = run_epoch(cnn, train_loader, criterion, optimizer, device)
        _va_loss, _va_acc, _va_tgt, _va_pred = run_epoch(cnn, val_loader, criterion, None, device)
        history_cnn.append(dict(epoch=_epoch + 1, train_loss=_tr_loss, train_acc=_tr_acc, val_loss=_va_loss, val_acc=_va_acc))
        print(f'Epoch {_epoch + 1:02d} | train loss {_tr_loss:.3f} acc {_tr_acc:.3f} | val loss {_va_loss:.3f} acc {_va_acc:.3f}')
    return criterion, history_cnn


@app.cell
def _(history_cnn, plt):
    # Plot training curves
    epochs = [h['epoch'] for h in history_cnn]
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4))
    _axes[0].plot(epochs, [h['train_loss'] for h in history_cnn], label='train')
    _axes[0].plot(epochs, [h['val_loss'] for h in history_cnn], label='val')
    _axes[0].set(title='Loss — SimpleCNN', xlabel='Epoch', ylabel='Loss')
    _axes[0].legend()
    _axes[1].plot(epochs, [h['train_acc'] for h in history_cnn], label='train')
    _axes[1].plot(epochs, [h['val_acc'] for h in history_cnn], label='val')
    _axes[1].set(title='Accuracy — SimpleCNN', xlabel='Epoch', ylabel='Accuracy')
    _axes[1].legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
    ConfusionMatrixDisplay,
    class_names,
    classification_report,
    cnn,
    confusion_matrix,
    criterion,
    device,
    plt,
    run_epoch,
    val_loader,
):
    # Validation classification report + confusion matrix
    _, _, _va_tgt, _va_pred = run_epoch(cnn, val_loader, criterion, None, device)
    print('SimpleCNN — Validation report')
    print(classification_report(_va_tgt, _va_pred, target_names=class_names))
    cm = confusion_matrix(_va_tgt, _va_pred)
    _fig, _ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=_ax, xticks_rotation=45, colorbar=False)
    _ax.set_title('Validation Confusion Matrix — SimpleCNN')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 6 — Transfer Learning: ResNet18

    `build_resnet18` is provided — you only need to call `run_epoch` (which you already implemented).
    Compare how much the pretrained model improves over the SimpleCNN baseline.
    """)
    return


@app.cell
def _(
    build_resnet18,
    criterion,
    device,
    run_epoch,
    torch,
    train_loader,
    val_loader,
):
    resnet = build_resnet18(num_classes=4).to(device)
    opt_resnet = torch.optim.Adam(resnet.parameters(), lr=1e-05)
    history_rn = []
    for _epoch in range(30):
        _tr_loss, _tr_acc, _, _ = run_epoch(resnet, train_loader, criterion, opt_resnet, device)
        _va_loss, _va_acc, _, _ = run_epoch(resnet, val_loader, criterion, None, device)
        history_rn.append(dict(epoch=_epoch + 1, train_loss=_tr_loss, train_acc=_tr_acc, val_loss=_va_loss, val_acc=_va_acc))
        print(f'Epoch {_epoch + 1:02d} | train loss {_tr_loss:.3f} acc {_tr_acc:.3f} | val loss {_va_loss:.3f} acc {_va_acc:.3f}')
    return (resnet,)


@app.cell
def _(
    ConfusionMatrixDisplay,
    class_names,
    classification_report,
    confusion_matrix,
    criterion,
    device,
    plt,
    resnet,
    run_epoch,
    val_loader,
):
    # Validation report + confusion matrix for ResNet18
    _, _, va_tgt_rn, va_pred_rn = run_epoch(resnet, val_loader, criterion, None, device)
    print('ResNet18 — Validation report')
    print(classification_report(va_tgt_rn, va_pred_rn, target_names=class_names))
    cm_rn = confusion_matrix(va_tgt_rn, va_pred_rn)
    _fig, _ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(cm_rn, display_labels=class_names).plot(ax=_ax, xticks_rotation=45, colorbar=False)
    _ax.set_title('Validation Confusion Matrix — ResNet18')
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 6: Model Comparison

    Two hypothetical models evaluated on the same held-out test set:

    | Model  | Accuracy | Macro-F1 | Worst-class Recall |
    |--------|----------|----------|--------------------|
    | CNN-A  | 0.93     | 0.79     | 0.61               |
    | CNN-B  | 0.91     | 0.84     | 0.74               |

    **6a.** Which model would you choose if missing a tumor case carries the highest clinical risk? Justify.

    > *Your answer.*

    **6b.** Which single metric most directly drives that decision?

    > *Your answer.*

    **6c.** What additional checks would you run before finalising the choice?

    > *Your answer.*

    **6d.** Now fill in the table below with the actual results from your SimpleCNN and ResNet18 runs above, and apply the same reasoning to decide which *your* models you would deploy.

    | Model      | Val Accuracy | Val Macro-F1 | Worst-class Recall |
    |------------|-------------|-------------|--------------------|
    | SimpleCNN  |             |             |                    |
    | ResNet18   |             |             |                    |

    > *Your decision and reasoning.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Discussion: Why this is not a clinical-grade benchmark

    Describe at least **two** limitations of the experimental setup used in this assignment
    that would prevent deploying this model in a real clinical setting.

    > *Your answer.*
    """)
    return


if __name__ == "__main__":
    app.run()
