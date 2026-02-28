"""Spatial analysis: classify buildings by roof type and green-roof coverage."""

from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape


# ---------------------------------------------------------------------------
# Roof-type classification rules
# ---------------------------------------------------------------------------

# Mapping from OSM building:levels or building tags to a simplified roof type.
# Extend / adjust this dictionary to match your study-area conventions.
ROOF_TYPE_RULES: dict = {
    "flat": "flat",
    "yes": "flat",       # OSM default when no specific value is given
    "apartments": "flat",
    "commercial": "flat",
    "industrial": "flat",
    "office": "flat",
    "retail": "flat",
    "house": "pitched",
    "detached": "pitched",
    "semidetached_house": "pitched",
    "terrace": "pitched",
    "residential": "pitched",
    "church": "other",
    "school": "flat",
    "university": "flat",
    "hospital": "flat",
    "warehouse": "flat",
}


def load_building_footprints(
    path: str,
    crs_target: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Load building footprints from a GeoJSON or Shapefile.

    Parameters
    ----------
    path : str
        Path to the vector file (GeoJSON / SHP / GPKG).
    crs_target : str or None
        EPSG string (e.g. ``"EPSG:25833"``).  If supplied, the data are
        re-projected to this CRS.

    Returns
    -------
    gpd.GeoDataFrame
        Building footprints with at least a ``geometry`` column.
    """
    gdf = gpd.read_file(path)
    if crs_target is not None and gdf.crs is not None:
        gdf = gdf.to_crs(crs_target)
    elif crs_target is not None:
        gdf = gdf.set_crs(crs_target, allow_override=True)
    return gdf


def raster_to_vector(
    prediction_path: str,
    green_roof_value: int = 1,
    min_area_m2: float = 5.0,
) -> gpd.GeoDataFrame:
    """Vectorise a binary segmentation raster into green-roof polygons.

    Parameters
    ----------
    prediction_path : str
        Path to the binary GeoTIFF produced by :func:`predict_raster`.
    green_roof_value : int
        Pixel value that represents green roof (default ``1``).
    min_area_m2 : float
        Minimum polygon area in square metres; smaller polygons are discarded.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with ``geometry`` (Polygon) and ``area_m2`` columns,
        in the same CRS as the input raster.
    """
    with rasterio.open(prediction_path) as src:
        mask_data = src.read(1)
        raster_transform = src.transform
        raster_crs = src.crs

    binary_mask = (mask_data == green_roof_value).astype(np.uint8)

    polygons = []
    for geom, val in shapes(binary_mask, transform=raster_transform):
        if val == 1:
            polygons.append(shape(geom))

    if not polygons:
        return gpd.GeoDataFrame(columns=["geometry", "area_m2"], crs=raster_crs)

    gdf = gpd.GeoDataFrame(geometry=polygons, crs=raster_crs)
    gdf["area_m2"] = gdf.geometry.area
    gdf = gdf[gdf["area_m2"] >= min_area_m2].reset_index(drop=True)
    return gdf


def classify_buildings(
    buildings: gpd.GeoDataFrame,
    green_roof_polygons: gpd.GeoDataFrame,
    building_tag_col: Optional[str] = "building",
) -> gpd.GeoDataFrame:
    """Intersect green-roof polygons with building footprints.

    For each building, this function computes the total green-roof area that
    overlaps with its footprint and assigns a simplified roof type based on
    OSM tags.

    Parameters
    ----------
    buildings : gpd.GeoDataFrame
        Building footprint GeoDataFrame (must have a geometry column).
    green_roof_polygons : gpd.GeoDataFrame
        Green-roof polygon GeoDataFrame from :func:`raster_to_vector`.
    building_tag_col : str or None
        Column in *buildings* that contains OSM building type tags.  If
        ``None``, all buildings are assigned ``"unknown"`` roof type.

    Returns
    -------
    gpd.GeoDataFrame
        A copy of *buildings* with the following additional columns:

        - ``roof_type`` – simplified roof type string.
        - ``building_area_m2`` – total footprint area.
        - ``green_roof_area_m2`` – overlapping green-roof area.
        - ``green_roof_fraction`` – fraction of footprint covered (0–1).
        - ``is_green_roof`` – boolean flag (True if fraction > 0.1).
    """
    # Ensure same CRS
    if buildings.crs != green_roof_polygons.crs and green_roof_polygons.crs is not None:
        green_roof_polygons = green_roof_polygons.to_crs(buildings.crs)

    result = buildings.copy()

    # Assign roof type from OSM tags
    if building_tag_col is not None and building_tag_col in result.columns:
        result["roof_type"] = (
            result[building_tag_col]
            .str.lower()
            .map(ROOF_TYPE_RULES)
            .fillna("other")
        )
    else:
        result["roof_type"] = "unknown"

    result["building_area_m2"] = result.geometry.area

    # Spatial join: find which green-roof polygons intersect each building
    joined = gpd.overlay(
        result[["geometry"]].reset_index().rename(columns={"index": "bldg_idx"}),
        green_roof_polygons[["geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    joined["overlap_area"] = joined.geometry.area

    # Sum overlapping green-roof area per building
    green_area_per_bldg = (
        joined.groupby("bldg_idx")["overlap_area"].sum().rename("green_roof_area_m2")
    )

    result = result.reset_index().rename(columns={"index": "bldg_idx"})
    result = result.merge(green_area_per_bldg, on="bldg_idx", how="left")
    result["green_roof_area_m2"] = result["green_roof_area_m2"].fillna(0.0)
    result["green_roof_fraction"] = (
        result["green_roof_area_m2"] / result["building_area_m2"].replace(0, np.nan)
    ).fillna(0.0)
    result["is_green_roof"] = result["green_roof_fraction"] > 0.1

    result = result.drop(columns=["bldg_idx"])
    return result.set_geometry("geometry")


def compute_green_roof_stats(
    classified_buildings: gpd.GeoDataFrame,
    roof_type_col: str = "roof_type",
) -> pd.DataFrame:
    """Compute green-roof statistics per building/roof type.

    Parameters
    ----------
    classified_buildings : gpd.GeoDataFrame
        Output of :func:`classify_buildings`.
    roof_type_col : str
        Column name containing the roof type label.

    Returns
    -------
    pd.DataFrame
        Summary table with the following columns:

        - ``roof_type``
        - ``total_buildings``
        - ``green_roof_buildings``
        - ``green_roof_percentage``
        - ``total_building_area_m2``
        - ``total_green_roof_area_m2``
        - ``green_roof_area_fraction``
    """
    stats = (
        classified_buildings.groupby(roof_type_col)
        .agg(
            total_buildings=(roof_type_col, "count"),
            green_roof_buildings=("is_green_roof", "sum"),
            total_building_area_m2=("building_area_m2", "sum"),
            total_green_roof_area_m2=("green_roof_area_m2", "sum"),
        )
        .reset_index()
        .rename(columns={roof_type_col: "roof_type"})
    )

    stats["green_roof_percentage"] = (
        stats["green_roof_buildings"] / stats["total_buildings"].replace(0, np.nan) * 100
    ).fillna(0.0)

    stats["green_roof_area_fraction"] = (
        stats["total_green_roof_area_m2"]
        / stats["total_building_area_m2"].replace(0, np.nan)
    ).fillna(0.0)

    return stats
