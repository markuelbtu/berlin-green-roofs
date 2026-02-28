"""Inference script: apply a trained U-Net to a full GeoTIFF orthophoto."""

import os
from typing import List, Optional, Tuple

import numpy as np
import rasterio
import torch
from rasterio.transform import Affine
from torch.utils.data import DataLoader, Dataset

from .model import build_unet


class _PatchDataset(Dataset):
    """Internal dataset that tiles a single raster into fixed-size patches.

    Parameters
    ----------
    image : np.ndarray
        Full-image array of shape (C, H, W), values in [0, 1].
    patch_size : int
        Square patch side length in pixels.
    stride : int
        Step size between consecutive patches (< patch_size → overlap).
    """

    def __init__(
        self,
        image: np.ndarray,
        patch_size: int = 256,
        stride: int = 256,
    ) -> None:
        self.image = image
        self.patch_size = patch_size
        self.stride = stride
        _, self.H, self.W = image.shape

        self.coords: List[Tuple[int, int]] = []
        for y in range(0, self.H - patch_size + 1, stride):
            for x in range(0, self.W - patch_size + 1, stride):
                self.coords.append((y, x))

    def __len__(self) -> int:  # noqa: D105
        return len(self.coords)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Return a patch tensor and its top-left pixel coordinate."""
        y, x = self.coords[idx]
        patch = self.image[:, y : y + self.patch_size, x : x + self.patch_size]
        return torch.from_numpy(patch).float(), (y, x)


def predict_raster(
    image_path: str,
    checkpoint_path: str,
    output_path: str,
    encoder_name: str = "resnet34",
    in_channels: int = 3,
    num_classes: int = 2,
    patch_size: int = 256,
    stride: int = 128,
    batch_size: int = 4,
    bands: Optional[List[int]] = None,
    device: Optional[str] = None,
) -> str:
    """Run inference on a full GeoTIFF and save a binary segmentation raster.

    The function tiles the input raster into overlapping patches, runs each
    patch through the model, and stitches the predictions back together using
    an averaging strategy to handle border effects.

    Parameters
    ----------
    image_path : str
        Path to the input GeoTIFF orthophoto.
    checkpoint_path : str
        Path to the ``.pth`` checkpoint produced by :func:`train_model`.
    output_path : str
        Path where the predicted binary raster will be written as a GeoTIFF.
    encoder_name : str
        smp encoder backbone that matches the trained checkpoint.
    in_channels : int
        Number of image channels used during training.
    num_classes : int
        Number of segmentation classes.
    patch_size : int
        Side length of each square inference patch in pixels.
    stride : int
        Step between consecutive patches. Values smaller than *patch_size*
        produce overlapping patches, which reduces border artefacts.
    batch_size : int
        Number of patches per forward pass.
    bands : list of int or None
        Zero-based band indices to read from the raster. Defaults to
        ``[0, 1, 2]`` (RGB).
    device : str or None
        Compute device; auto-detected when ``None``.

    Returns
    -------
    str
        Absolute path to the saved prediction GeoTIFF.
    """
    if bands is None:
        bands = list(range(in_channels))

    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    dev = torch.device(device)

    # --- Load model --------------------------------------------------------
    model = build_unet(
        encoder_name=encoder_name,
        in_channels=in_channels,
        num_classes=num_classes,
        encoder_weights=None,
    ).to(dev)

    checkpoint = torch.load(checkpoint_path, map_location=dev)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()

    # --- Load raster -------------------------------------------------------
    with rasterio.open(image_path) as src:
        rasterio_bands = [b + 1 for b in bands]
        image_data = src.read(rasterio_bands).astype(np.float32)
        profile = src.profile
        transform: Affine = src.transform

    # Normalise
    ch_max = image_data.max(axis=(1, 2), keepdims=True)
    ch_max = np.where(ch_max == 0, 1.0, ch_max)
    image_data = np.clip(image_data / ch_max, 0.0, 1.0)

    _, H, W = image_data.shape

    # --- Tile inference ----------------------------------------------------
    patch_ds = _PatchDataset(image_data, patch_size=patch_size, stride=stride)
    loader = DataLoader(patch_ds, batch_size=batch_size, shuffle=False,
                        num_workers=0)

    # Accumulator arrays for stitching
    score_map = np.zeros((num_classes, H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    with torch.no_grad():
        for patches, (ys, xs) in loader:
            patches = patches.to(dev)
            logits = model(patches)  # (B, num_classes, patch_size, patch_size)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            for i in range(len(ys)):
                y, x = int(ys[i]), int(xs[i])
                score_map[:, y : y + patch_size, x : x + patch_size] += probs[i]
                count_map[y : y + patch_size, x : x + patch_size] += 1.0

    # Avoid division by zero for pixels not covered by any patch
    count_map = np.where(count_map == 0, 1.0, count_map)
    score_map /= count_map

    prediction = score_map.argmax(axis=0).astype(np.uint8)

    # --- Save output raster ------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress="lzw",
    )

    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(prediction[np.newaxis, ...])

    print(f"Prediction saved to: {output_path}")
    return os.path.abspath(output_path)
