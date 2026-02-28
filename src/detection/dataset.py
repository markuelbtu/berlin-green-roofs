"""Dataset utilities for loading GeoTIFF orthophotos and binary green-roof labels."""

import os
from typing import Callable, List, Optional, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class GreenRoofDataset(Dataset):
    """PyTorch Dataset for semantic segmentation of green roofs.

    Loads paired GeoTIFF image patches and corresponding binary label masks.
    Images are expected to be RGB or RGB+NIR GeoTIFFs; labels are single-band
    rasters where 1 = green roof and 0 = background.

    Parameters
    ----------
    image_dir : str
        Directory containing image GeoTIFF files.
    label_dir : str
        Directory containing label GeoTIFF files (same filenames as images).
    transform : callable, optional
        Albumentations-style transform applied to both image and mask.
    image_size : tuple of int, optional
        Target (height, width) to resize patches to. Defaults to (256, 256).
    bands : list of int, optional
        Zero-based band indices to read from the image raster.
        Defaults to [0, 1, 2] (RGB).
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (256, 256),
        bands: Optional[List[int]] = None,
    ) -> None:
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_size = image_size
        self.bands = bands if bands is not None else [0, 1, 2]

        self.filenames: List[str] = sorted(
            f for f in os.listdir(image_dir) if f.lower().endswith((".tif", ".tiff"))
        )
        if not self.filenames:
            raise FileNotFoundError(
                f"No GeoTIFF files found in image directory: {image_dir}"
            )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and return the image/mask pair at *idx*.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        image : torch.Tensor
            Float32 tensor of shape (C, H, W) normalised to [0, 1].
        mask : torch.Tensor
            Long tensor of shape (H, W) with values 0 (background) or
            1 (green roof).
        """
        filename = self.filenames[idx]
        image_path = os.path.join(self.image_dir, filename)
        label_path = os.path.join(self.label_dir, filename)

        image = self._load_image(image_path)
        mask = self._load_mask(label_path)

        if self.transform is not None:
            # albumentations expects HWC uint8 images and HW masks
            image_hwc = (image.transpose(1, 2, 0) * 255).astype(np.uint8)
            augmented = self.transform(image=image_hwc, mask=mask)
            image_hwc = augmented["image"]
            mask = augmented["mask"]
            image = image_hwc.transpose(2, 0, 1).astype(np.float32) / 255.0

        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask).long()
        return image_tensor, mask_tensor

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_image(self, path: str) -> np.ndarray:
        """Read selected bands from a GeoTIFF and return a (C, H, W) float32 array."""
        with rasterio.open(path) as src:
            # rasterio bands are 1-indexed
            rasterio_bands = [b + 1 for b in self.bands]
            data = src.read(rasterio_bands)  # shape: (C, H, W)

        data = data.astype(np.float32)
        # Normalise to [0, 1]; clip potential outliers
        band_max = data.max(axis=(1, 2), keepdims=True)
        band_max = np.where(band_max == 0, 1.0, band_max)
        data = np.clip(data / band_max, 0.0, 1.0)

        data = self._resize_array(data, self.image_size, is_mask=False)
        return data

    def _load_mask(self, path: str) -> np.ndarray:
        """Read the first band of a label GeoTIFF and return a (H, W) uint8 array."""
        with rasterio.open(path) as src:
            mask = src.read(1)  # shape: (H, W)

        mask = (mask > 0).astype(np.uint8)
        mask = self._resize_array(mask[np.newaxis, ...], self.image_size, is_mask=True)
        return mask.squeeze(0)

    @staticmethod
    def _resize_array(
        array: np.ndarray,
        target_size: Tuple[int, int],
        is_mask: bool,
    ) -> np.ndarray:
        """Resize a (C, H, W) array to *target_size* using PIL.

        Parameters
        ----------
        array : np.ndarray
            Input array of shape (C, H, W).
        target_size : tuple of int
            Desired (height, width).
        is_mask : bool
            If True, uses nearest-neighbour interpolation to preserve label values.

        Returns
        -------
        np.ndarray
            Resized array of shape (C, target_h, target_w).
        """
        from PIL import Image

        resample = Image.NEAREST if is_mask else Image.BILINEAR
        resized_bands = []
        for band in array:
            pil_img = Image.fromarray(band)
            pil_img = pil_img.resize((target_size[1], target_size[0]), resample)
            resized_bands.append(np.array(pil_img))
        return np.stack(resized_bands, axis=0)
