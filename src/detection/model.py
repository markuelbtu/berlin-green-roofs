"""U-Net model builder using the segmentation_models_pytorch library."""

from typing import Optional

import torch
import torch.nn as nn


def build_unet(
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
    num_classes: int = 2,
) -> nn.Module:
    """Build a U-Net segmentation model.

    Uses `segmentation_models_pytorch` (smp) which provides a rich set of
    pre-trained encoders.  Falls back to a lightweight custom U-Net when
    `segmentation_models_pytorch` is not installed.

    Parameters
    ----------
    encoder_name : str
        Name of the timm / smp encoder backbone (e.g. ``"resnet34"``,
        ``"efficientnet-b0"``).
    encoder_weights : str or None
        Pre-trained weights for the encoder.  Pass ``None`` to train from
        scratch.
    in_channels : int
        Number of input image channels (3 for RGB, 4 for RGB+NIR).
    num_classes : int
        Number of output segmentation classes (2 for binary green-roof
        detection).

    Returns
    -------
    nn.Module
        U-Net model ready for training / inference.
    """
    try:
        import segmentation_models_pytorch as smp

        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,  # raw logits; use CrossEntropyLoss
        )
    except ImportError:
        model = _SimpleUNet(in_channels=in_channels, num_classes=num_classes)

    return model


# ---------------------------------------------------------------------------
# Lightweight fallback U-Net (no external dependency)
# ---------------------------------------------------------------------------


class _DoubleConv(nn.Module):
    """Two consecutive Conv-BN-ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.block(x)


class _SimpleUNet(nn.Module):
    """Minimal U-Net implementation used as fallback when smp is unavailable.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    num_classes : int
        Number of output segmentation classes.
    base_filters : int
        Number of feature maps in the first encoder block.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        base_filters: int = 32,
    ) -> None:
        super().__init__()
        f = base_filters

        # Encoder
        self.enc1 = _DoubleConv(in_channels, f)
        self.enc2 = _DoubleConv(f, f * 2)
        self.enc3 = _DoubleConv(f * 2, f * 4)
        self.enc4 = _DoubleConv(f * 4, f * 8)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = _DoubleConv(f * 8, f * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = _DoubleConv(f * 16, f * 8)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = _DoubleConv(f * 8, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = _DoubleConv(f * 4, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = _DoubleConv(f * 2, f)

        self.head = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Logits tensor of shape (B, num_classes, H, W).
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)
