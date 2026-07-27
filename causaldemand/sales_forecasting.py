"""sales forecasting demand-prediction metrics.

Revenue-weighted WMAPE (accuracy) and WMPE (bias) on demand forecasts under the observed pricing policy, over the chronological holdout window (the last `counterfactual_eval_weeks` weeks).

Conventions:

* **Truth object = OBSERVED holdout sales** (the conventional, M5-style forecasting target). The holdout weeks' units/dollars are withheld from the public transactions file and live in `hidden/transactions_full_hidden.csv`; participants receive the holdout weeks' prices/promo flags as `public/transactions_holdout_context_public.csv` (the conditional-forecasting inputs). Scoring against observed counts carries an irreducible observation-noise floor shared by all models.
* **Aggregation** — the formulas index products only; errors are summed within product over (store, week) and revenue-weighted across products: `WMAPE = Σ_i w_i Σ_st |q̂−q*| / Σ_i w_i Σ_st |q*|`.
* **Weights** — `w_i` = the product's share of observed public revenue over the TRAINING window (the holdout window's revenue is hidden, so training- window weights keep w_i participant-reproducible).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

KEY_COLUMNS = ["product_id", "store_id", "week"]


def revenue_weights(transactions_window: pd.DataFrame) -> pd.Series:
    """Per-product revenue share over the supplied window (the weight w_i).

    Callers pass the public TRAINING window (holdout revenue is hidden; training-window weights are participant-reproducible).
    """
    revenue = transactions_window.groupby("product_id")["dollars"].sum().astype(float)
    total = float(revenue.sum())
    if total <= 0:
        return pd.Series(1.0 / max(len(revenue), 1), index=revenue.index)
    return revenue / total


def build_demand_truth(transactions_full: pd.DataFrame, eval_weeks: list[int]) -> pd.DataFrame:
    """Observed holdout sales per (product, store, week) — the sales forecasting truth.

    `transactions_full` is the hidden full panel (`hidden/transactions_full_hidden.csv`); the truth is its `units` column over the holdout window.
    """
    window = transactions_full[
        transactions_full["week"].isin(set(int(w) for w in eval_weeks))
    ].copy()
    window["true_units"] = pd.to_numeric(window["units"], errors="coerce").fillna(0.0)
    return window[KEY_COLUMNS + ["true_units"]].reset_index(drop=True)


def demand_prediction_scores(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    weights: pd.Series,
) -> dict[str, Any]:
    """Demand-WMAPE + Demand-WMPE for one submission.

    `predictions` must carry (product_id, store_id, week, predicted_units); `truth` carries the same keys + `true_units`. Scoring runs on the inner join; coverage diagnostics report rows of truth without a prediction — an incomplete submission is flagged, not silently dropped.
    """
    pred = predictions[KEY_COLUMNS + ["predicted_units"]].copy()
    pred["predicted_units"] = pd.to_numeric(pred["predicted_units"], errors="coerce")
    merged = truth.merge(pred, on=KEY_COLUMNS, how="left")
    missing = int(merged["predicted_units"].isna().sum())
    scored = merged.dropna(subset=["predicted_units"]).copy()

    per_product = (
        scored.assign(
            abs_err=(scored["predicted_units"] - scored["true_units"]).abs(),
            signed_err=scored["predicted_units"] - scored["true_units"],
            abs_true=scored["true_units"].abs(),
        )
        .groupby("product_id")[["abs_err", "signed_err", "abs_true"]]
        .sum()
    )
    w = weights.reindex(per_product.index).fillna(0.0).astype(float)
    denominator = float((w * per_product["abs_true"]).sum())
    if denominator <= 0:
        wmape = None
        wmpe = None
    else:
        wmape = float((w * per_product["abs_err"]).sum() / denominator)
        wmpe = float((w * per_product["signed_err"]).sum() / denominator)
    return {
        "metric": "sales_forecasting",
        "demand_wmape": wmape,
        "demand_wmpe": wmpe,
        "n_truth_rows": int(len(truth)),
        "n_scored_rows": int(len(scored)),
        "n_truth_rows_without_prediction": missing,
        "submission_complete": missing == 0,
        "weighting": "per-product observed revenue share over the public training window",
    }
