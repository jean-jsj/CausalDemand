from __future__ import annotations

from typing import Any

import pandas as pd

PROMO_DIRECTIONS: tuple[tuple[str, float], ...] = (("promo_plus10", 0.10), ("promo_minus10", -0.10))

PANEL_TOP_PRODUCTS: int = 12
PANEL_TOP_BRANDS: int = 4


def panel_pinned_depth_selections(
    off_base_context: pd.DataFrame,
    products: pd.DataFrame,
    universe: list[str],
    n_products: int = PANEL_TOP_PRODUCTS,
    n_brands: int = PANEL_TOP_BRANDS,
) -> list[dict[str, Any]]:
    universe = sorted(str(p) for p in universe)
    universe_set = set(universe)
    ctx = off_base_context[
        off_base_context["product_id"].astype(str).isin(universe_set)
    ].copy()
    ctx["product_id"] = ctx["product_id"].astype(str)
    units = ctx.groupby("product_id")["units"].sum().reindex(universe).fillna(0.0)
    total = float(units.sum()) or 1.0
    ranked_products = (
        units.rename("units").rename_axis("product_id").reset_index()
        .sort_values(["units", "product_id"], ascending=[False, True])
    )
    brand_of = products.assign(product_id=products["product_id"].astype(str)).set_index(
        "product_id"
    )["brand_code"].astype(str)
    brand_units = (
        units.rename("units").rename_axis("product_id").to_frame()
        .assign(brand_code=lambda d: d.index.map(brand_of))
        .groupby("brand_code")["units"].sum().reset_index()
        .sort_values(["units", "brand_code"], ascending=[False, True])
    )
    common = {
        "channel": "promo_depth",
        "cap": "dont_filter",
        "defined_on": "intended_depth",
        "pinned": "off_cell_window_over_universe_intersection",
        "universe_size": len(universe),
    }
    selections: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked_products.head(n_products).itertuples(), start=1):
        selections.append({
            "rule": f"panel_product{rank:02d}",
            "focal_products": {str(row.product_id)},
            "directions": PROMO_DIRECTIONS,
            "basis": {**common, "units_rank": rank, "units_share": round(float(row.units) / total, 4)},
        })
    for rank, row in enumerate(brand_units.head(n_brands).itertuples(), start=1):
        members = {p for p in universe if brand_of.get(p) == str(row.brand_code)}
        selections.append({
            "rule": f"panel_brand{rank}",
            "focal_products": members,
            "directions": PROMO_DIRECTIONS,
            "basis": {
                **common,
                "brand_code": str(row.brand_code),
                "units_rank": rank,
                "units_share": round(float(row.units) / total, 4),
                "n_products": len(members),
            },
        })
    return selections
