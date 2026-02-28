# Berlin Green Roof Detection & Hydrological Impact Simulation

Final Project for AI4HWS

## Overview

This project detects green roofs in Berlin using Deep Learning (U-Net semantic
segmentation) applied to high-resolution orthophotos (2025) and simulates the
hydrological impact of urban greening scenarios.

## Workflow

| Step | Description | Module |
|------|-------------|--------|
| 1 | **Detection** – segment green roofs from RGB/NIR orthophotos | `src/detection/` |
| 2 | **Classification** – intersect ML results with OSM building footprints | `src/classification/` |
| 3 | **Simulation** – predict stormwater runoff reduction for a 50 % green-roof scenario | `src/simulation/` |

## Project Structure

```
berlin-green-roofs/
├── src/
│   ├── detection/
│   │   ├── dataset.py        # GeoTIFF dataset loader (GreenRoofDataset)
│   │   ├── model.py          # U-Net builder (smp or built-in fallback)
│   │   ├── train.py          # End-to-end training pipeline
│   │   └── predict.py        # Patch-based inference on full rasters
│   ├── classification/
│   │   └── spatial_analysis.py  # Vectorise raster, intersect with buildings
│   └── simulation/
│       └── hydrology.py      # Runoff coefficients (ψ) & scenario comparison
├── notebooks/
│   ├── 01_detection.ipynb    # Train the U-Net and run inference
│   ├── 02_classification.ipynb  # Spatial analysis & status-quo stats
│   └── 03_simulation.ipynb   # Hydrological scenario comparison
├── data/
│   ├── raw/                  # Input orthophotos & building footprints
│   ├── labels/               # Ground-truth label rasters
│   ├── processed/            # Intermediate outputs
│   └── results/              # Model checkpoints, predictions, stats
└── requirements.txt
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your data (see data/README.md)

# 3. Open the notebooks in order
jupyter notebook notebooks/01_detection.ipynb
```

## Technical Stack

- **Framework**: PyTorch / segmentation-models-pytorch
- **Model**: U-Net with ResNet34 encoder (ImageNet pre-trained)
- **Geospatial**: GeoPandas, Rasterio, Shapely
- **Data**: GeoTIFF orthophotos, GeoJSON/SHP building footprints (OSM)

## Hydrological Formula

Stormwater runoff volume is computed using the rational method:

```
V [m³] = Area [m²] × rainfall_depth [m] × ψ
```

where ψ (psi) is the dimensionless runoff coefficient.  Typical values:

| Surface | ψ |
|---------|---|
| Conventional sealed roof | 0.85–0.90 |
| Extensive green roof | 0.30 |
| Intensive green roof | 0.10 |

See `src/simulation/hydrology.py` for the full table of 10 Berlin building types.
