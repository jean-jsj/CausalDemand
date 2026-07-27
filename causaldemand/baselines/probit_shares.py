"""Share-space price sensitivity for the discrete-choice reference estimators.

Recovers d log(share) / d price from the public transactions by a
within-(product-store, week) share regression: fixed effects are absorbed by
alternating weighted demeaning, and the IV variants identify the slope by
instrumenting price with ``supply_cost_proxy``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RANK_BANDWIDTH = 0.35
OMEGA_MIX = 0.5
DEMEAN_PASSES = 4
SEASON_PERIOD_WEEKS = 52
MIN_SHARE = 1e-6


def _weighted_demean(frame: pd.DataFrame, cols: list[str], by: list[str]) -> pd.DataFrame:
    """Alternating-projection weighted demeaning (weights = ``wt`` column)."""
    for _ in range(DEMEAN_PASSES):
        for group in by:
            grouped = frame.groupby(group)
            wsum = grouped["wt"].transform("sum")
            for c in cols:
                frame["_wx"] = frame[c] * frame["wt"]
                frame[c] = frame[c] - grouped["_wx"].transform("sum") / wsum
    return frame.drop(columns=["_wx"], errors="ignore")


def _weighted_slope(sub: pd.DataFrame, use_iv: bool) -> float | None:
    wt = sub["wt"].to_numpy()
    x = sub["price"].to_numpy()
    y = sub["log_share"].to_numpy()
    z = sub["supply_cost_proxy"].to_numpy()
    if use_iv:
        denom = float((wt * z * x).sum())
        return float((wt * z * y).sum()) / denom if abs(denom) > 1e-9 else None
    denom = float((wt * x * x).sum())
    return float((wt * x * y).sum()) / denom if denom > 1e-9 else None


def fit_pooled_clipped(transactions: pd.DataFrame, *, use_iv: bool) -> dict:
    """Small-assortment variant: pooled, unweighted, zero shares clipped.

    On a market of only a few products the sales-weighted positive-rows design
    is dominated by the largest-share product and under-responds; the clipped
    pooled slope behaves better there.
    """
    work = transactions[["product_id", "store_id", "week", "units", "price", "supply_cost_proxy"]].copy()
    totals = work.groupby(["store_id", "week"])["units"].transform("sum")
    work = work[totals > 0].copy()
    work["share"] = (work["units"] / totals[totals > 0]).clip(lower=MIN_SHARE)
    work["log_share"] = np.log(work["share"])
    work["wt"] = 1.0
    work["ps_key"] = work["product_id"].astype(str) + "|" + work["store_id"].astype(str)
    cols = ["log_share", "price", "supply_cost_proxy"]
    work[cols] = work[cols].astype(float)
    work = _weighted_demean(work, cols, by=["ps_key", "week"])
    b = _weighted_slope(work, use_iv)
    return {"b": b if b is not None else 0.0, "b_by_product": {}, "use_iv": use_iv, "n_rows": int(len(work))}


def fit_share_price_sensitivity(transactions: pd.DataFrame, *, use_iv: bool) -> dict:
    """Recover d log(share) / d price, sales-weighted and per product.

    Zero-unit rows carry no weight: the released panel is a positive-sales
    extract of an ~89%-zero observation layer, and clipping zero shares into
    the regression buries the price signal in the clip floor. Sensitivity is
    estimated per product — the scored flagship is the share-highest product,
    whose share responsiveness differs from the pooled average — with the
    pooled slope as fallback for sparse products.
    """
    work = transactions[["product_id", "store_id", "week", "units", "price", "supply_cost_proxy"]].copy()
    totals = work.groupby(["store_id", "week"])["units"].transform("sum")
    keep = (totals > 0) & (work["units"] > 0)
    work = work[keep].copy()
    work["share"] = (work["units"] / totals[keep]).clip(lower=MIN_SHARE)
    work["log_share"] = np.log(work["share"])
    work["wt"] = work["units"].astype(float)
    work["ps_key"] = work["product_id"].astype(str) + "|" + work["store_id"].astype(str)
    cols = ["log_share", "price", "supply_cost_proxy"]
    work[cols] = work[cols].astype(float)
    work = _weighted_demean(work, cols, by=["ps_key", "week"])

    b_pooled = _weighted_slope(work, use_iv)
    if b_pooled is None:
        b_pooled = 0.0
    b_by_product: dict = {}
    for product_id, sub in work.groupby("product_id"):
        if len(sub) < 200:
            continue
        b_j = _weighted_slope(sub, use_iv)
        if b_j is not None and b_j < 0:
            b_by_product[product_id] = b_j
    return {
        "b": b_pooled,
        "b_by_product": b_by_product,
        "use_iv": use_iv,
        "n_rows": int(len(work)),
    }
