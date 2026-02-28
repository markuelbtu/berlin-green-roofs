# Data Directory

Place your raw data files here before running the notebooks.

## Expected layout

```
data/
├── raw/
│   ├── berlin_ortho_2025.tif        # High-resolution RGB/NIR orthophoto (GeoTIFF)
│   ├── images/                      # Training image patches (GeoTIFF)
│   │   ├── patch_001.tif
│   │   └── ...
│   └── buildings_berlin.geojson     # OSM building footprints (Overpass Turbo export)
├── labels/
│   ├── patch_001.tif                # Binary labels (same filenames as images/)
│   └── ...
├── processed/                       # Intermediate files (auto-generated)
└── results/                         # Model outputs (auto-generated)
    ├── models/
    │   └── best_model.pth
    ├── prediction.tif
    ├── classified_buildings.gpkg
    ├── green_roof_stats.csv
    └── runoff_comparison.csv
```

## Downloading building footprints (Overpass Turbo)

Use the following Overpass QL query at https://overpass-turbo.eu/:

```overpass
[out:json][timeout:60];
area["name"="Berlin"]["admin_level"="4"]->.berlin;
(
  way["building"](area.berlin);
  relation["building"](area.berlin);
);
out body;
>;
out skel qt;
```

Export as GeoJSON and save to `data/raw/buildings_berlin.geojson`.
