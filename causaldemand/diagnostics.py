"""Participant insight CSVs: where did my model go wrong?

Flattens a set of `evaluate_submission` scores.json files (one per cell — typically the dev seed's 4 cells) into two spreadsheet-friendly tables:

1. **Forecasting + elasticity diagnostics** (`--out`): rows = cells, columns = the
   sales-forecasting metrics and the full elasticity-recovery scorecard (own-price block; cross-price NDCG, per-class F1, magnitude/bias for all pairs and per true relationship class). Reading a row says which world hurts the model; reading a column says which capability is missing.
2. **Counterfactual intervention matrix** (`--counterfactual-out`): rows = cells, columns = the
   16 protocol interventions, values = the substitution WAPE. The headline scenario is one of these columns; the rest are the robustness surface (e.g. good on single-product moves but collapsing on brand portfolios = weak substitution structure).

Neither file feeds the leaderboard.

Usage:
    python3 -m causaldemand.diagnostics my_scores/*.json \
        --out diagnostics.csv --counterfactual-out interventions.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _forecast_elasticity_row(scores: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"cell": scores.get("cell_slug") or Path(scores.get("cell_dir", "")).name}
    forecasting = scores.get("sales_forecasting", {})
    row["l1_demand_wmape"] = forecasting.get("demand_wmape")
    row["l1_demand_wmpe"] = forecasting.get("demand_wmpe")
    elasticity = scores.get("elasticity_recovery", {})
    own = elasticity.get("own_price") or {}
    for key in ("sign_accuracy", "wmape", "rmse", "wmpe", "mean_signed_error"):
        row[f"l2_own_{key}"] = own.get(key)
    cross = elasticity.get("cross_price") or {}
    row["l2_cross_ndcg"] = cross.get("ndcg")
    row["l2_cross_ndcg_at_5"] = cross.get("ndcg_at_5")
    for cls, block in (cross.get("f1_per_class") or {}).items():
        row[f"l2_cross_f1_{cls}"] = (block or {}).get("f1")
    for key in ("wmape", "rmse", "wmpe", "mean_signed_error"):
        row[f"l2_cross_{key}"] = (cross.get("all_pairs") or {}).get(key)
    for cls, block in (cross.get("by_true_class") or {}).items():
        for key in ("wmape", "rmse", "wmpe"):
            row[f"l2_cross_{key}_{cls}"] = (block or {}).get(key)
    row["l2_unrelated_abs_threshold"] = cross.get("unrelated_abs_threshold")
    return row


def _counterfactual_row(scores: dict[str, Any]) -> dict[str, Any]:
    """One row of the L3 intervention matrix: the substitution WAPE per
    intervention."""
    row: dict[str, Any] = {"cell": scores.get("cell_slug") or Path(scores.get("cell_dir", "")).name}
    counterfactual = scores.get("counterfactual_prediction", {})
    for intervention in counterfactual.get("interventions", []):
        value = intervention.get("substitution_wape")
        row[intervention.get("intervention_id")] = value
    return row


# Column order for the intervention matrix: selection by selection, +10 then −10, matching the protocol table. Unknown ids append at the end.
INTERVENTION_ORDER = [
    f"sweep_{rule}_{direction}"
    for rule in (
        "single_random",
        "single_share_highest",
        "single_share_lowest",
        "single_price_highest",
        "single_price_lowest",
        "brand_leading",
        "brand_smaller",
    )
    for direction in ("plus10", "minus10")
]


def build_tables(score_payloads: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    l12 = pd.DataFrame([_forecast_elasticity_row(s) for s in score_payloads]).sort_values("cell").reset_index(drop=True)
    l3 = pd.DataFrame([_counterfactual_row(s) for s in score_payloads]).sort_values("cell").reset_index(drop=True)
    ordered = ["cell"] + [c for c in INTERVENTION_ORDER if c in l3.columns] + [
        c for c in l3.columns if c != "cell" and c not in INTERVENTION_ORDER
    ]
    return l12, l3[ordered]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="forecasting + elasticity diagnostics CSV")
    parser.add_argument("--counterfactual-out", type=Path, default=None, help="counterfactual intervention-matrix CSV")
    args = parser.parse_args()
    payloads = [json.loads(path.read_text()) for path in args.scores]
    l12, l3 = build_tables(payloads)
    l12.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(l12)} cells x {len(l12.columns) - 1} metrics)")
    if args.counterfactual_out:
        l3.to_csv(args.counterfactual_out, index=False)
        print(f"wrote {args.counterfactual_out} ({len(l3)} cells x {len(l3.columns) - 1} interventions)")


if __name__ == "__main__":
    main()
