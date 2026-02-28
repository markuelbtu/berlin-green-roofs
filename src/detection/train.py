"""Training pipeline for the green-roof U-Net segmentation model."""

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split

from .model import build_unet


def train_model(
    image_dir: str,
    label_dir: str,
    output_dir: str,
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
    num_classes: int = 2,
    image_size: Tuple[int, int] = (256, 256),
    batch_size: int = 8,
    num_epochs: int = 30,
    learning_rate: float = 1e-4,
    val_split: float = 0.15,
    device: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, list]:
    """Train the U-Net green-roof segmentation model end-to-end.

    Splits the dataset into training and validation sets, trains with
    cross-entropy loss, and saves the best checkpoint by validation IoU.

    Parameters
    ----------
    image_dir : str
        Directory containing image GeoTIFF patches.
    label_dir : str
        Directory containing label GeoTIFF patches.
    output_dir : str
        Directory where model checkpoints will be saved.
    encoder_name : str
        smp encoder backbone name (e.g. ``"resnet34"``).
    encoder_weights : str or None
        Pre-trained weights for the encoder.
    in_channels : int
        Number of image input channels.
    num_classes : int
        Number of segmentation classes (2 for binary classification).
    image_size : tuple of int
        Patch (height, width) fed to the model.
    batch_size : int
        Mini-batch size.
    num_epochs : int
        Total training epochs.
    learning_rate : float
        Initial learning rate for the AdamW optimiser.
    val_split : float
        Fraction of data reserved for validation (0 < val_split < 1).
    device : str or None
        ``"cuda"``, ``"mps"``, or ``"cpu"``.  Auto-detected when ``None``.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        History dictionary with keys ``"train_loss"``, ``"val_loss"``,
        ``"val_iou"`` containing per-epoch lists of float values.
    """
    torch.manual_seed(seed)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dev = torch.device(device)

    os.makedirs(output_dir, exist_ok=True)

    # --- Dataset & DataLoaders -------------------------------------------
    from .dataset import GreenRoofDataset

    dataset = GreenRoofDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        image_size=image_size,
        bands=list(range(in_channels)),
    )

    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=(device == "cuda"))

    # --- Model, loss, optimiser, scheduler --------------------------------
    model = build_unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        num_classes=num_classes,
    ).to(dev)

    criterion = nn.CrossEntropyLoss()
    optimiser = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimiser, T_max=num_epochs, eta_min=1e-6)

    history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_iou": []}
    best_iou = -1.0
    best_ckpt_path = os.path.join(output_dir, "best_model.pth")

    for epoch in range(1, num_epochs + 1):
        # Training pass
        model.train()
        train_loss = _run_epoch(model, train_loader, criterion, dev,
                                optimiser=optimiser)

        # Validation pass
        model.eval()
        val_loss, val_iou = _run_val_epoch(model, val_loader, criterion, dev,
                                           num_classes=num_classes)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)

        print(
            f"Epoch {epoch:03d}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_iou={val_iou:.4f}"
        )

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimiser_state_dict": optimiser.state_dict(),
                    "val_iou": best_iou,
                },
                best_ckpt_path,
            )

    print(f"Training complete. Best val IoU: {best_iou:.4f}")
    print(f"Best checkpoint saved to: {best_ckpt_path}")
    return history


# ---------------------------------------------------------------------------
# Internal epoch helpers
# ---------------------------------------------------------------------------


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimiser: Optional[torch.optim.Optimizer] = None,
) -> float:
    """Run one training epoch and return average loss.

    Parameters
    ----------
    model : nn.Module
        Segmentation model.
    loader : DataLoader
        Data loader for the current split.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Compute device.
    optimiser : Optimizer or None
        If provided, performs a backward pass and parameter update.

    Returns
    -------
    float
        Mean loss over all batches.
    """
    total_loss = 0.0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        if optimiser is not None:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def _run_val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 2,
) -> Tuple[float, float]:
    """Run one validation epoch; return (mean_loss, mean_IoU).

    Parameters
    ----------
    model : nn.Module
        Segmentation model in eval mode.
    loader : DataLoader
        Validation data loader.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Compute device.
    num_classes : int
        Number of segmentation classes.

    Returns
    -------
    tuple of float
        ``(mean_loss, mean_iou)`` averaged over all batches.
    """
    total_loss = 0.0
    total_iou = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            total_iou += _batch_iou(preds, masks, num_classes)

    n = max(len(loader), 1)
    return total_loss / n, total_iou / n


def _batch_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> float:
    """Compute mean Intersection-over-Union for a batch.

    Parameters
    ----------
    preds : torch.Tensor
        Predicted class indices of shape (B, H, W).
    targets : torch.Tensor
        Ground-truth class indices of shape (B, H, W).
    num_classes : int
        Total number of classes.

    Returns
    -------
    float
        Mean IoU across all classes (ignores classes with no pixels).
    """
    ious = []
    for cls in range(num_classes):
        pred_cls = preds == cls
        target_cls = targets == cls
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return float(sum(ious) / len(ious)) if ious else 0.0
