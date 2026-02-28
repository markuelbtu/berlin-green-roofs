"""Hydrological runoff simulation for the Berlin green-roof scenario analysis.

Implements the rational-method-inspired formula:

    V = Area * rainfall_depth * psi

where *psi* (ψ) is the dimensionless runoff coefficient (0–1) describing the
fraction of rainfall converted to surface runoff for a given surface type.

References
----------
DWA-A 138 / ATV-A 117: typical runoff coefficients for urban surfaces.
"""

from typing import Dict, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Runoff coefficients (ψ values) for typical Berlin building types
# ---------------------------------------------------------------------------

#: Runoff coefficients keyed by simplified building/roof category.
#: These are median values derived from DWA-A 138 guidelines.
#: - Conventional roofs (sealed): ψ ≈ 0.9
#: - Green roofs (extensive):     ψ ≈ 0.3
#: - Green roofs (intensive):     ψ ≈ 0.1
PSI_VALUES: Dict[str, Dict[str, float]] = {
    # building_type: {conventional: ψ, green_roof: ψ}
    "flat_commercial": {"conventional": 0.90, "green_roof": 0.30},
    "flat_residential": {"conventional": 0.90, "green_roof": 0.30},
    "flat_industrial": {"conventional": 0.85, "green_roof": 0.30},
    "flat_office": {"conventional": 0.90, "green_roof": 0.25},
    "flat_retail": {"conventional": 0.90, "green_roof": 0.30},
    "flat_school": {"conventional": 0.88, "green_roof": 0.28},
    "flat_hospital": {"conventional": 0.88, "green_roof": 0.25},
    "flat_warehouse": {"conventional": 0.85, "green_roof": 0.30},
    "pitched_residential": {"conventional": 0.80, "green_roof": 0.35},
    "other": {"conventional": 0.80, "green_roof": 0.35},
}

#: Fallback PSI values used when a building type is not found in PSI_VALUES.
_DEFAULT_PSI: Dict[str, float] = {"conventional": 0.85, "green_roof": 0.30}


def compute_runoff(
    area_m2: float,
    rainfall_depth_mm: float,
    psi: float,
) -> float:
    """Compute stormwater runoff volume for a single surface.

    Uses the simplified rational formula:

        V [m³] = Area [m²] * rainfall_depth [m] * ψ [-]

    Parameters
    ----------
    area_m2 : float
        Surface area in square metres.
    rainfall_depth_mm : float
        Design rainfall depth in millimetres.
    psi : float
        Runoff coefficient (dimensionless, 0–1).

    Returns
    -------
    float
        Runoff volume in cubic metres (m³).
    """
    rainfall_depth_m = rainfall_depth_mm / 1000.0
    return area_m2 * rainfall_depth_m * psi


def compute_status_quo_runoff(
    buildings: pd.DataFrame,
    rainfall_depth_mm: float,
    building_type_col: str = "building_type",
    area_col: str = "building_area_m2",
    green_roof_fraction_col: str = "green_roof_fraction",
) -> pd.DataFrame:
    """Compute runoff volumes for the current (status-quo) situation.

    Each building's roof is divided into its current green and conventional
    fractions.  The corresponding runoff coefficients are applied to each
    portion and summed.

    Parameters
    ----------
    buildings : pd.DataFrame
        Building-level data including area, building type, and current
        green-roof fraction.
    rainfall_depth_mm : float
        Design rainfall event depth in millimetres.
    building_type_col : str
        Column name for building type keys (must match keys in
        :data:`PSI_VALUES`).
    area_col : str
        Column name for building footprint area (m²).
    green_roof_fraction_col : str
        Column name for the existing green-roof fraction (0–1).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an additional ``runoff_m3_status_quo`` column.
    """
    result = buildings.copy()

    def _row_runoff(row: pd.Series) -> float:
        btype = row.get(building_type_col, "other")
        psi_vals = PSI_VALUES.get(str(btype), _DEFAULT_PSI)
        area = float(row.get(area_col, 0.0))
        green_frac = float(row.get(green_roof_fraction_col, 0.0))
        green_frac = max(0.0, min(1.0, green_frac))

        green_area = area * green_frac
        conv_area = area * (1.0 - green_frac)

        return compute_runoff(green_area, rainfall_depth_mm, psi_vals["green_roof"]) + \
               compute_runoff(conv_area, rainfall_depth_mm, psi_vals["conventional"])

    result["runoff_m3_status_quo"] = result.apply(_row_runoff, axis=1)
    return result


def compute_scenario_runoff(
    buildings: pd.DataFrame,
    rainfall_depth_mm: float,
    target_green_roof_fraction: float = 0.50,
    building_type_col: str = "building_type",
    area_col: str = "building_area_m2",
    green_roof_fraction_col: str = "green_roof_fraction",
) -> pd.DataFrame:
    """Compute runoff volumes for a hypothetical green-roof scenario.

    For each building that currently has a green-roof fraction below the
    *target*, the fraction is raised to *target_green_roof_fraction*.
    Buildings already above the target are left unchanged.

    Parameters
    ----------
    buildings : pd.DataFrame
        Building-level data (same schema as for :func:`compute_status_quo_runoff`).
    rainfall_depth_mm : float
        Design rainfall event depth in millimetres.
    target_green_roof_fraction : float
        Desired minimum green-roof fraction for the scenario (default 0.50,
        i.e. 50 % of all eligible roof area converted to green roofs).
    building_type_col : str
        Column name for building type keys.
    area_col : str
        Column name for building footprint area (m²).
    green_roof_fraction_col : str
        Column name for the existing green-roof fraction (0–1).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with two additional columns:

        - ``green_roof_fraction_scenario`` – applied green fraction.
        - ``runoff_m3_scenario`` – runoff volume under the scenario.
    """
    result = buildings.copy()

    result["green_roof_fraction_scenario"] = result[green_roof_fraction_col].clip(
        lower=target_green_roof_fraction, upper=1.0
    )

    def _row_runoff(row: pd.Series) -> float:
        btype = row.get(building_type_col, "other")
        psi_vals = PSI_VALUES.get(str(btype), _DEFAULT_PSI)
        area = float(row.get(area_col, 0.0))
        green_frac = float(row.get("green_roof_fraction_scenario", 0.0))
        green_frac = max(0.0, min(1.0, green_frac))

        green_area = area * green_frac
        conv_area = area * (1.0 - green_frac)

        return compute_runoff(green_area, rainfall_depth_mm, psi_vals["green_roof"]) + \
               compute_runoff(conv_area, rainfall_depth_mm, psi_vals["conventional"])

    result["runoff_m3_scenario"] = result.apply(_row_runoff, axis=1)
    return result


def compare_scenarios(
    buildings: pd.DataFrame,
    rainfall_depth_mm: float,
    target_green_roof_fraction: float = 0.50,
    building_type_col: str = "building_type",
    area_col: str = "building_area_m2",
    green_roof_fraction_col: str = "green_roof_fraction",
    group_by_col: Optional[str] = "building_type",
) -> pd.DataFrame:
    """Run both scenarios and produce a comparison summary.

    Computes status-quo and scenario runoff for every building, then
    aggregates the results by *group_by_col* (or across all buildings if
    ``group_by_col`` is ``None``).

    Parameters
    ----------
    buildings : pd.DataFrame
        Building-level data.
    rainfall_depth_mm : float
        Design rainfall event depth in millimetres.
    target_green_roof_fraction : float
        Scenario target green-roof fraction.
    building_type_col : str
        Column name for building type keys.
    area_col : str
        Column name for building footprint area (m²).
    green_roof_fraction_col : str
        Column name for existing green-roof fraction.
    group_by_col : str or None
        Column to group results by (e.g. ``"building_type"``).  If ``None``,
        a single-row summary for the whole dataset is returned.

    Returns
    -------
    pd.DataFrame
        Summary table with columns:

        - ``group`` (if *group_by_col* is not ``None``)
        - ``total_area_m2``
        - ``total_runoff_status_quo_m3``
        - ``total_runoff_scenario_m3``
        - ``runoff_reduction_m3``
        - ``runoff_reduction_pct``
    """
    df = compute_status_quo_runoff(
        buildings,
        rainfall_depth_mm=rainfall_depth_mm,
        building_type_col=building_type_col,
        area_col=area_col,
        green_roof_fraction_col=green_roof_fraction_col,
    )
    df = compute_scenario_runoff(
        df,
        rainfall_depth_mm=rainfall_depth_mm,
        target_green_roof_fraction=target_green_roof_fraction,
        building_type_col=building_type_col,
        area_col=area_col,
        green_roof_fraction_col=green_roof_fraction_col,
    )

    if group_by_col is not None and group_by_col in df.columns:
        summary = (
            df.groupby(group_by_col)
            .agg(
                total_area_m2=(area_col, "sum"),
                total_runoff_status_quo_m3=("runoff_m3_status_quo", "sum"),
                total_runoff_scenario_m3=("runoff_m3_scenario", "sum"),
            )
            .reset_index()
            .rename(columns={group_by_col: "group"})
        )
    else:
        summary = pd.DataFrame(
            {
                "total_area_m2": [df[area_col].sum()],
                "total_runoff_status_quo_m3": [df["runoff_m3_status_quo"].sum()],
                "total_runoff_scenario_m3": [df["runoff_m3_scenario"].sum()],
            }
        )

    summary["runoff_reduction_m3"] = (
        summary["total_runoff_status_quo_m3"] - summary["total_runoff_scenario_m3"]
    )
    summary["runoff_reduction_pct"] = (
        summary["runoff_reduction_m3"]
        / summary["total_runoff_status_quo_m3"].replace(0, float("nan"))
        * 100
    ).fillna(0.0)

    return summary
