"""Assemble scored models into a leaderboard table.

Usage:
    python3 -m causaldemand.leaderboard scores_a.json scores_b.json ... \
        [--out leaderboard.csv] [--format table|markdown|csv] [--aggregate-seeds]

Each input is an `evaluate_submission` output. Rows are ranked (within each cell-type) by `|own-price bias|` ASCENDING — closest to zero first (the counterfactual headline own axis). Own-price bias (signed, for direction) and substitution error are reported as the PAIR, both from the single scenario `sweep_single_share_highest_plus10`; the forecasting and elasticity headline numbers ride along as columns. The README's main arena is the two complex endogeneity-on cell-types.

`--aggregate-seeds` groups scores of the same (model × cell-type) across seeds and reports `mean ± spread` — the maintainer-side mode for building the README table from the eval seeds. `--format markdown` emits the README-ready table. The leaderboard is per cell-type; endogeneity-on cells are the benchmark's main arena.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SEED_SUFFIX = re.compile(r"_seed\d+$")


def _cell_type(scores: dict[str, Any]) -> str:
    slug = scores.get("cell_slug") or Path(scores.get("cell_dir", "")).name
    return _SEED_SUFFIX.sub("", slug)


def leaderboard_rows(score_payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scores in score_payloads:
        counterfactual = scores.get("counterfactual_prediction", {})
        # The headline pair: own-price WMPE (signed) and substitution WAPE, both from the single scenario `sweep_single_share_highest_plus10`.
        headline = counterfactual.get("headline") or {}
        own_block = headline.get("own_price") or {}
        sub_block = headline.get("substitution") or {}
        own_wmpe = own_block.get("own_price_wmpe")
        sub_wape = sub_block.get("substitution_wape")
        n_store_weeks = sub_block.get("n_store_weeks_scored")
        forecasting = scores.get("sales_forecasting", {})
        elasticity = scores.get("elasticity_recovery", {})
        own = elasticity.get("own_price") or {}
        # Actual-arm blocks are absent on synthetic rows, so every .get chain
        # degrades to None. The validity fractions are a diagnostic panel:
        # never combined into one scalar, never ranked.
        data_arm = scores.get("data_arm", "synthetic")
        l4 = scores.get("validity_checks_actual") or {}
        # Forecasting runs on both arms; the actual-arm forecast keeps its own
        # column so the two arms never mix.
        is_actual = data_arm == "actual"
        rows.append(
            {
                "submission": scores.get("submission_name"),
                "cell_type": _cell_type(scores),
                "cell_slug": scores.get("cell_slug") or Path(scores.get("cell_dir", "")).name,
                "data_arm": data_arm,
                "substitution_error": sub_wape,
                "own_price_bias": own_wmpe,
                "counterfactual_n_store_weeks": n_store_weeks,
                "forecast_error": forecasting.get("demand_wmape") if not is_actual else None,
                "forecast_bias": forecasting.get("demand_wmpe") if not is_actual else None,
                "elasticity_own_sign_accuracy": own.get("sign_accuracy"),
                "elasticity_own_error": own.get("wmape"),
                "elasticity_own_bias": own.get("wmpe"),
                "validity_own_sign_frac": (l4.get("own_price_sign") or {}).get("frac_correct_sign"),
                "validity_substitution_frac": (l4.get("substitution_sign") or {}).get(
                    "frac_redistribution_mass_correct"
                ),
                "validity_range_in_band": (l4.get("own_elasticity_range") or {}).get("frac_in_band"),
                "validity_monotonicity_frac": (l4.get("monotonicity") or {}).get("frac_consistent"),
                # The PASS/WARN/FAIL coherence verdict (actual-arm panel headline).
                "validity_gate_verdict": (l4.get("gate") or {}).get("verdict"),
                "actual_forecast_error": forecasting.get("demand_wmape") if is_actual else None,
                "actual_forecast_bias": forecasting.get("demand_wmpe") if is_actual else None,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame):
        # Rank WITHIN each cell-type by |own-price WMPE| ASCENDING (closest-to-zero identification bias first); the reported `own_price_bias` column stays SIGNED (direction). The README's main arena is the two complex endogeneity-on cell types. The actual arm contributes no sort key: actual rows carry `None` own-WMPE → `na_position="last"` sinks them to the bottom of their cell-type block, and the cumcount groupby is (cell_type, data_arm) so `rank` RESTARTS per arm, keeping the actual arm in its own diagnostic partition — never interleaved into, and never a headline ranking of, the synthetic rows.
        frame["_abs_own_wmpe"] = pd.to_numeric(frame["own_price_bias"], errors="coerce").abs()
        frame = frame.sort_values(
            ["cell_type", "_abs_own_wmpe"], ascending=[True, True], na_position="last"
        ).reset_index(drop=True).drop(columns="_abs_own_wmpe")
        frame.insert(0, "rank", frame.groupby(["cell_type", "data_arm"]).cumcount() + 1)
    return frame


def aggregate_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    """Group per (submission × cell_type) across seeds: mean ± cross-seed spread.

    The maintainer-side mode for the README table: the official score is the headline averaged over the eval seeds, with the spread as built-in seed-robustness evidence.
    """
    numeric = [
        "substitution_error",
        "own_price_bias",
        "forecast_error",
        "forecast_bias",
        "elasticity_own_sign_accuracy",
        "elasticity_own_error",
        "elasticity_own_bias",
        # Actual-data arm diagnostics: without these columns the seed-aggregated table would silently DROP every actual-arm number.
        "validity_own_sign_frac",
        "validity_substitution_frac",
        "validity_range_in_band",
        "validity_monotonicity_frac",
        "actual_forecast_error",
        "actual_forecast_bias",
    ]
    # Carry `data_arm` through the grouping so each arm aggregates separately (arms stay in separate partitions; both survive aggregation).
    if "data_arm" not in frame.columns:
        frame = frame.assign(data_arm="synthetic")
    grouped = frame.groupby(["submission", "cell_type", "data_arm"], dropna=False)
    rows = []
    for (submission, cell_type, data_arm), group in grouped:
        row: dict[str, Any] = {
            "submission": submission,
            "cell_type": cell_type,
            "data_arm": data_arm,
            "n_seeds": int(group["cell_slug"].nunique()),
        }
        for col in numeric:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            row[col] = float(values.mean()) if len(values) else None
            row[f"{col}_seed_sd"] = float(values.std(ddof=0)) if len(values) > 1 else None
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        # Same |own-WMPE| ASC direction as `leaderboard_rows`; the rank cumcount restarts per (cell_type, data_arm) so the actual arm is its own partition, never interleaved.
        out["_abs_own_wmpe"] = pd.to_numeric(out["own_price_bias"], errors="coerce").abs()
        out = out.sort_values(
            ["cell_type", "_abs_own_wmpe"], ascending=[True, True], na_position="last"
        ).reset_index(drop=True).drop(columns="_abs_own_wmpe")
        out.insert(0, "rank", out.groupby(["cell_type", "data_arm"]).cumcount() + 1)
    return out


def to_markdown(frame: pd.DataFrame) -> str:
    """README-ready markdown: one table, headline first, ± spread when present."""
    display = frame.copy()
    for col in ("own_price_bias", "substitution_error"):
        if f"{col}_seed_sd" in display.columns:
            display[col] = [
                (
                    f"{m:.3f} ± {s:.3f}"
                    if pd.notna(m) and pd.notna(s)
                    else (f"{m:.3f}" if pd.notna(m) else "")
                )
                for m, s in zip(display[col], display[f"{col}_seed_sd"])
            ]
    display = display[[c for c in display.columns if not c.endswith("_seed_sd")]]
    keep = [
        c
        for c in [
            "rank", "submission", "cell_type", "cell_slug", "data_arm", "n_seeds",
            "own_price_bias", "substitution_error",
            "forecast_error", "elasticity_own_error", "elasticity_own_bias",
            # Actual-arm diagnostic panel columns: rendered alongside, NOT the sort key.
            "actual_forecast_error",
            "validity_own_sign_frac", "validity_substitution_frac",
            "validity_range_in_band", "validity_monotonicity_frac",
        ]
        if c in display.columns
    ]
    display = display[keep]
    for col in display.columns:
        if display[col].dtype == float:
            display[col] = display[col].map(lambda v: f"{v:.3f}" if pd.notna(v) else "")
    header = "| " + " | ".join(display.columns) + " |"
    divider = "|" + "|".join(["---"] * len(display.columns)) + "|"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in display.itertuples(index=False))
    return "\n".join([header, divider, body]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--format", choices=["table", "markdown", "csv"], default="table")
    parser.add_argument("--aggregate-seeds", action="store_true")
    args = parser.parse_args()
    payloads = [json.loads(path.read_text()) for path in args.scores]
    table = leaderboard_rows(payloads)
    if args.aggregate_seeds:
        table = aggregate_seeds(table)
    if args.format == "markdown":
        rendered = to_markdown(table)
    elif args.format == "csv":
        rendered = table.to_csv(index=False)
    else:
        rendered = table.to_string(index=False)
    if args.out:
        args.out.write_text(rendered if args.format != "csv" else table.to_csv(index=False), encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
