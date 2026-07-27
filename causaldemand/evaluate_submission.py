"""Score one submission directory against one benchmark cell directory.

Usage:
    python3 -m causaldemand.evaluate_submission \
        --cell-dir outputs/complex_log_log_endogenous_seed001 \ --submission-dir my_submission/ \ [--out scores.json] [--submission-name my_model]

Tasks whose file is absent from the submission directory are reported as `not_submitted`. See docs/SUBMISSION_FORMAT.md for the file contracts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from causaldemand.headline_decomposition import (
    decomposed_headline,
    focal_from_context,
)
from causaldemand.sales_forecasting import (
    build_demand_truth,
    demand_prediction_scores,
    revenue_weights,
)
from causaldemand.elasticity import elasticity_scores
from causaldemand.validity_checks import (
    FACIAL_TISSUE_OWN_BAND,
    coherence_gate,
    own_elasticity_range_coverage,
    price_direction_from_context,
    validity_scores,
)

FORECAST_FILE = "forecast_predictions.csv"
ELASTICITY_FILE = "elasticity_matrix.csv"
COUNTERFACTUAL_FILE = "counterfactual_deltas.csv"

# Actual-data arm: participants submit TWO files — a sales forecast and the predicted Δq under the public own-price sweep for the validity checks. No elasticity file; the own-elasticity band is DERIVED from the submitted Δq (score_validity).
ACTUAL_FORECAST_FILE = "actual_forecast_predictions.csv"
ACTUAL_VALIDITY_FILE = "actual_validity_deltas.csv"

# Required columns per submission CSV — validated up front so a malformed file yields a clear, participant-facing message instead of a raw pandas KeyError deep inside a pivot/merge. (See docs/SUBMISSION_FORMAT.md.)
FORECAST_COLUMNS = ["product_id", "store_id", "week", "predicted_units"]
ELASTICITY_COLUMNS = ["affected_product_id", "priced_product_id", "elasticity"]
COUNTERFACTUAL_COLUMNS = [
    "intervention_id",
    "product_id",
    "store_id",
    "week",
    "predicted_delta_units",
]


class SubmissionFormatError(ValueError):
    """A submission CSV is unreadable or missing required columns."""


def _read_submission(path: Path, required: list[str], task_label: str) -> pd.DataFrame:
    """Read a submission CSV, raising a participant-friendly error on a bad
    format (empty, unreadable, or missing required columns)."""
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise SubmissionFormatError(
            f"{task_label}: {path.name} is empty (expected columns: {', '.join(required)})."
        ) from exc
    except Exception as exc:  # malformed CSV / encoding / parser error
        raise SubmissionFormatError(
            f"{task_label}: could not read {path.name} ({exc})."
        ) from exc
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SubmissionFormatError(
            f"{task_label}: {path.name} is missing required column(s): "
            f"{', '.join(missing)}. Found: [{', '.join(map(str, df.columns))}]. "
            f"Expected: [{', '.join(required)}]. See docs/SUBMISSION_FORMAT.md."
        )
    return df


# Headline scenario — the headline reads BOTH axes (own-price WMPE + substitution WAPE) from a SINGLE flagship-focal scenario: `sweep_single_share_highest_plus10` (highest-share focal, ±X% via promo depth, all regime-on store-weeks), so the two numbers describe the same submitted response. Headline *cells* for the README leaderboard are the two complex endogeneity-on cells (both families).
HEADLINE_INTERVENTION = "sweep_single_share_highest_plus10"
MAIN_LEADERBOARD_CELL_TYPES = (
    "complex_log_log_endogenous",
    "complex_covariance_probit_endogenous",
)


def _load_cell(cell_dir: Path) -> dict[str, Any]:
    # Released cells carry a sanitized scoring config (`release/ scoring_params.json`: model family, eval-window length, benchmark version — nothing else); maintainer-side full outputs fall back to the resolved run config.
    params_path = cell_dir / "release" / "scoring_params.json"
    if params_path.exists():
        cfg = json.loads(params_path.read_text())
    else:
        cfg = json.loads(
            (cell_dir / "reports" / "run_config_resolved.json").read_text()
        )["config"]
    family = cfg["benchmark_family"]["active_cell"]["model_family"]
    # Scoring needs the hidden full panel (the forecasting observed-sales truth lives there; the public file is the training window only). Available on dev cells by design; on eval cells only the maintainer has it.
    transactions_full = pd.read_csv(cell_dir / "hidden" / "transactions_full_hidden.csv")
    eval_weeks = sorted(transactions_full["week"].unique())[
        -int(cfg["simulation"]["counterfactual_eval_weeks"]) :
    ]
    training = transactions_full[~transactions_full["week"].isin(set(eval_weeks))]
    return {
        "cfg": cfg,
        "family": family,
        "transactions_full": transactions_full,
        "training": training,
        "eval_weeks": [int(w) for w in eval_weeks],
    }


def _truth_frames_by_intervention(cell_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    # Scores every intervention in the released sweep truth (the 16 protocol interventions of the public sweep context).
    path = cell_dir / "hidden" / "counterfactual_sweep_truth_hidden.csv"
    if path.exists():
        frame = pd.read_csv(path)
        for intervention_id, group in frame.groupby("intervention_id"):
            frames[str(intervention_id)] = group.reset_index(drop=True)
    return frames


def _load_headline_context(cell_dir: Path) -> dict[str, Any]:
    """Public sweep context (focal identification only).

    The headline is geometry-BLIND: no distance matrix / products_order is read anywhere in the counterfactual scoring. Only the public sweep context is needed, to map each intervention to its focal product via ``focal_from_context``.
    """
    context_path = cell_dir / "public" / "counterfactual_sweep_context_public.csv"
    context = pd.read_csv(context_path) if context_path.exists() else None
    return {"context": context}


def score_forecasting(cell_dir: Path, cell: dict[str, Any], submission_path: Path) -> dict[str, Any]:
    predictions = _read_submission(submission_path, FORECAST_COLUMNS, "sales forecasting")
    truth = build_demand_truth(cell["transactions_full"], cell["eval_weeks"])
    weights = revenue_weights(cell["training"])
    return demand_prediction_scores(predictions, truth, weights)


def score_elasticity(cell_dir: Path, cell: dict[str, Any], submission_path: Path) -> dict[str, Any]:
    submitted = _read_submission(submission_path, ELASTICITY_COLUMNS, "elasticity recovery")
    truth_long = pd.read_csv(cell_dir / "hidden" / "elasticity_truth_hidden.csv")
    eps_star = truth_long.pivot(
        index="affected_product_id", columns="priced_product_id", values="epsilon_star"
    )
    # The scored truth is the TOTAL elasticity; the conditional (fixed-M switching) matrix, when the DGP emits it, drives the substitute/complement/unrelated stratification.
    eps_star_conditional = None
    if "epsilon_star_conditional" in truth_long.columns:
        eps_star_conditional = truth_long.pivot(
            index="affected_product_id",
            columns="priced_product_id",
            values="epsilon_star_conditional",
        )
    eps_hat = submitted.pivot(
        index="affected_product_id", columns="priced_product_id", values="elasticity"
    )
    weights = revenue_weights(cell["training"])
    return elasticity_scores(
        eps_hat, eps_star, weights, eps_star_conditional=eps_star_conditional
    )


def score_counterfactual(
    cell_dir: Path,
    cell: dict[str, Any],
    submission_path: Path,
    submission_name: str,
    dump_values: Path | None = None,  # accepted and ignored; reserved for a per-store-week error dump.
) -> dict[str, Any]:
    """Decomposed counterfactual prediction headline.

    Every intervention is scored on the geometry-blind decomposition — own-price signed WMPE (focal Δq) + substitution unsigned WAPE (competitor-only Δq), both category-netted via ΔM=ΣΔq, both pooled (micro-averaged) over store-weeks. The headline reads BOTH axes from the SINGLE scenario `sweep_single_share_highest_plus10`.
    """
    deltas = _read_submission(submission_path, COUNTERFACTUAL_COLUMNS, "counterfactual prediction")
    truths = _truth_frames_by_intervention(cell_dir)
    geom = _load_headline_context(cell_dir)

    interventions: list[dict[str, Any]] = []
    for intervention_id, truth in sorted(truths.items()):
        sub = deltas[deltas["intervention_id"] == intervention_id]
        merged = truth.merge(
            sub[["product_id", "store_id", "week", "predicted_delta_units"]],
            on=["product_id", "store_id", "week"],
            how="left",
        )
        # Omitted rows score as delta-hat = 0 (no predicted demand change) — omission cannot hide a store-week (see SUBMISSION_FORMAT.md).
        merged["dq_pred"] = pd.to_numeric(
            merged["predicted_delta_units"], errors="coerce"
        ).fillna(0.0)
        merged["dq_true"] = merged["true_counterfactual_units"].astype(float) - merged[
            "baseline_units"
        ].astype(float)

        focal = (
            focal_from_context(geom["context"], intervention_id)
            if geom["context"] is not None
            else None
        )
        if focal is None:
            result: dict[str, Any] = {
                "intervention_id": intervention_id,
                "status": "no_focal_in_public_context",
            }
        else:
            result = {
                "intervention_id": intervention_id,
                **decomposed_headline(
                    merged[["product_id", "store_id", "week", "baseline_units", "dq_true", "dq_pred"]],
                    focal,
                ),
                "n_submitted_rows": int(len(sub)),
                "n_truth_rows_without_submission": int(
                    merged["predicted_delta_units"].isna().sum()
                ),
            }
        interventions.append(result)

    by_id = {r.get("intervention_id"): r for r in interventions}
    hl = by_id.get(HEADLINE_INTERVENTION) or {}
    return {
        "metric": "counterfactual_wmpe_wape_pair",
        "spec_reference": "docs/SUBMISSION_FORMAT.md (counterfactual prediction headline)",
        "headline_components": ["own_price_wmpe", "substitution_wape"],
        "headline": {
            "scenario": HEADLINE_INTERVENTION,
            "rank_metric": "own_price.abs_own_price_wmpe",   # |signed WMPE|, ascending
            "own_price": {
                "own_price_wmpe": hl.get("own_price_wmpe"),
                "n_store_weeks_focal_missing": hl.get("n_store_weeks_focal_missing"),
            },
            "substitution": {
                "substitution_wape": hl.get("substitution_wape"),
                "n_store_weeks_scored": hl.get("n_store_weeks_scored"),
            },
        },
        "interventions": interventions,
    }


# Tissue own-price band: the module default DEFAULT_OWN_BAND = (-5.0, -0.2) is the WIDE generic-CPG prior; facial tissue pins a narrower band. Single source of truth — the citation-grounded FACIAL_TISSUE_OWN_BAND (Hoch, Kim, Montgomery & Rossi 1995 tissue ≈ -2; Tellis 1988 mean -1.76). Passed EXPLICITLY at every validity_checks call site.
VALIDITY_TISSUE_BAND = FACIAL_TISSUE_OWN_BAND


def _weighted_fraction(pairs: list[tuple[float | None, float | None]]) -> float | None:
    """Counts-weighted mean of per-focal fractions.

    Each ``(fraction, weight)`` pair carries the focal's fraction (e.g. ``frac_correct_sign``) and the count the underlying check returns (``n_evaluated`` / ``total_competitor_mass`` / ``n_store_weeks``). ``None`` fractions (or non-positive weights) are dropped. Returns ``None`` if nothing contributes.
    """
    num = 0.0
    den = 0.0
    for frac, weight in pairs:
        if frac is None or weight is None or weight <= 0:
            continue
        num += float(frac) * float(weight)
        den += float(weight)
    return (num / den) if den > 0 else None


def score_validity(cell: dict[str, Any], submission_path: Path) -> dict[str, Any]:
    """Actual-data-arm validity scoring.

    Pure wiring around the ``causaldemand.validity_checks`` bundle. Consumes ONLY the public ``cell["sweep_context"]`` + the submitted Δq (ACTUAL_VALIDITY_FILE); reads NO hidden truth, NO ``cell_dir``, NO ``hidden/`` file. For each focal in the full own-price sweep (every product moved once, BOTH signs) it runs the label-free own-sign / substitution-sign / monotonicity checks, and derives an own-elasticity range coverage from the submitted Δq (NO elasticity file) over the full J-diagonal, scored against the explicit tissue band.
    """
    deltas = _read_submission(submission_path, COUNTERFACTUAL_COLUMNS, "validity checks (actual arm)")
    context = cell["sweep_context"]

    # Group the sweep interventions by focal product, splitting each focal's ± pair by the label-free sign reader. `focal_from_context` maps an intervention_id to its focal; `price_direction_from_context` gives +1/-1.
    per_focal: dict[str, dict[str, Any]] = {}
    for intervention_id in context["intervention_id"].astype(str).unique():
        focal = focal_from_context(context, intervention_id)
        if focal is None:
            continue
        sign = price_direction_from_context(context, intervention_id)
        if sign is None:
            continue
        # LEFT-merge submitted Δq onto the sweep rows for this intervention; omitted rows coerce to dq_pred = 0.0 (same omission rule as counterfactual prediction).
        rows = context[context["intervention_id"].astype(str) == intervention_id][
            ["product_id", "store_id", "week", "baseline_units", "baseline_price", "intervention_price"]
        ].copy()
        sub = deltas[deltas["intervention_id"].astype(str) == intervention_id]
        frame = rows.merge(
            sub[["product_id", "store_id", "week", "predicted_delta_units"]],
            on=["product_id", "store_id", "week"],
            how="left",
        )
        frame["dq_pred"] = pd.to_numeric(
            frame["predicted_delta_units"], errors="coerce"
        ).fillna(0.0)
        slot = "up" if sign > 0 else "dn"
        per_focal.setdefault(str(focal), {})[slot] = frame

    # Derived own-elasticity diagonal — NO elasticity file. For each focal, eps_j = (Σ dq_pred_focal / Σ baseline_units_focal) / pct_move, using the +x% leg consistently so eps_j is a single scalar per product.
    focals = sorted(per_focal.keys())
    eps_by_focal: dict[str, float] = {}
    for focal, slots in per_focal.items():
        leg = slots["up"] if "up" in slots else slots.get("dn")
        if leg is None:
            continue
        foc = leg[leg["product_id"].astype(str) == str(focal)]
        base_sum = float(pd.to_numeric(foc["baseline_units"], errors="coerce").fillna(0.0).sum())
        dq_sum = float(pd.to_numeric(foc["dq_pred"], errors="coerce").fillna(0.0).sum())
        bp = pd.to_numeric(foc["baseline_price"], errors="coerce")
        ip = pd.to_numeric(foc["intervention_price"], errors="coerce")
        pct = ((ip - bp) / bp).replace([float("inf"), float("-inf")], pd.NA).dropna()
        if base_sum == 0 or pct.empty or float(pct.iloc[0]) == 0.0:
            continue
        eps_by_focal[focal] = (dq_sum / base_sum) / float(pct.iloc[0])

    # J×J diagonal frame; off-diagonal NaN (own_elasticity_range_coverage reads only np.diag). Range coverage is computed ONCE over the full diagonal here; the per-focal loop passes eps_hat=None so it is never double-counted.
    if focals:
        diag = pd.DataFrame(
            np.full((len(focals), len(focals)), np.nan),
            index=focals,
            columns=focals,
        )
        for focal in focals:
            if focal in eps_by_focal:
                diag.loc[focal, focal] = eps_by_focal[focal]
        own_elasticity_range = own_elasticity_range_coverage(diag, band=VALIDITY_TISSUE_BAND)
    else:
        own_elasticity_range = own_elasticity_range_coverage(
            pd.DataFrame(), band=VALIDITY_TISSUE_BAND
        )

    # Per-focal label-free checks. eps_hat=None in the loop (range coverage is the single call above); the band is a no-op when eps_hat is None but is still passed explicitly.
    own_sign_pairs: list[tuple[float | None, float | None]] = []
    sub_sign_pairs: list[tuple[float | None, float | None]] = []
    count_pairs: list[tuple[float | None, float | None]] = []
    mono_pairs: list[tuple[float | None, float | None]] = []
    for focal in focals:
        slots = per_focal[focal]
        frame_up = slots.get("up")
        frame_dn = slots.get("dn")
        if frame_up is not None:
            scored = validity_scores(
                frame_up,
                focal,
                price_increase=True,        # score the +x% leg; the -x% is the pair
                eps_hat=None,               # range coverage computed once above
                band=VALIDITY_TISSUE_BAND,    # tissue band, passed explicitly
                paired_frame=frame_dn,      # enables monotonicity across the ± pair
            )
        elif frame_dn is not None:
            scored = validity_scores(
                frame_dn,
                focal,
                price_increase=False,
                eps_hat=None,
                band=VALIDITY_TISSUE_BAND,
            )
        else:
            continue
        osign = scored.get("own_price_sign", {})
        own_sign_pairs.append((osign.get("frac_correct_sign"), osign.get("n_evaluated")))
        ssign = scored.get("substitution_sign", {})
        sub_sign_pairs.append(
            (ssign.get("frac_redistribution_mass_correct"), ssign.get("total_competitor_mass"))
        )
        # Unweighted per-competitor count (weight = #competitor observations), so many small wrong competitors can't hide behind one big correct one.
        count_pairs.append(
            (
                ssign.get("frac_competitors_correct_count"),
                ssign.get("n_competitor_observations_scored"),
            )
        )
        mono = scored.get("monotonicity")
        if mono is not None:
            mono_pairs.append((mono.get("frac_consistent"), mono.get("n_store_weeks")))

    result: dict[str, Any] = {
        "metric": "validity_checks_actual",
        "own_price_sign": {
            "frac_correct_sign": _weighted_fraction(own_sign_pairs),
        },
        "substitution_sign": {
            "frac_redistribution_mass_correct": _weighted_fraction(sub_sign_pairs),
            "frac_competitors_correct_count": _weighted_fraction(count_pairs),
        },
        "own_elasticity_range": own_elasticity_range,
        "monotonicity": {
            "frac_consistent": _weighted_fraction(mono_pairs),
        },
        "n_focals": len(focals),
        "band": list(VALIDITY_TISSUE_BAND),
    }
    # Fold the panel rates into a PASS/WARN/FAIL verdict: coherence_gate reads own_price_sign / substitution_sign / own_elasticity_range / monotonicity off the assembled result — the actual-arm's headline verdict.
    result["gate"] = coherence_gate(result)
    return result


def _result_header(cell: dict[str, Any], cell_slug: str, cell_dir: Any) -> dict[str, Any]:
    """The shared scores-dict header (used by both `evaluate` and
    `evaluate_prebuilt`). `cell_dir` is echoed as a string when a dir exists,
    else `None` (a pre-built actual fixture has no dir on disk)."""
    return {
        "schema_version": 2,
        "benchmark_version": cell["cfg"].get("benchmark_version", "unversioned"),
        "submission_name": None,  # set by caller
        "cell_dir": str(cell_dir) if cell_dir is not None else None,
        "cell_slug": cell_slug,
        "family": cell["family"],
        "evaluation_weeks": cell["eval_weeks"],
        "headline_statistic": "pooled_own_wmpe_substitution_wape",
        "spec_reference": "docs/SUBMISSION_FORMAT.md",
    }


def evaluate(
    cell_dir: Path,
    submission_dir: Path,
    submission_name: str,
    dump_values: Path | None = None,
) -> dict[str, Any]:
    cell = _load_cell(cell_dir)
    scores: dict[str, Any] = _result_header(cell, Path(cell_dir).name, cell_dir)
    scores["submission_name"] = submission_name
    # A malformed file in one task must not sink the others: invalid tasks report a clear status, well-formed tasks still score.
    for task, filename, scorer in (
        ("sales_forecasting", FORECAST_FILE, score_forecasting),
        ("elasticity_recovery", ELASTICITY_FILE, score_elasticity),
    ):
        path = submission_dir / filename
        if not path.exists():
            scores[task] = {"status": "not_submitted"}
            continue
        try:
            scores[task] = scorer(cell_dir, cell, path)
        except SubmissionFormatError as exc:
            scores[task] = {"status": "invalid_format", "error": str(exc)}
    counterfactual_path = submission_dir / COUNTERFACTUAL_FILE
    if not counterfactual_path.exists():
        scores["counterfactual_prediction"] = {"status": "not_submitted"}
    else:
        try:
            scores["counterfactual_prediction"] = score_counterfactual(
                cell_dir, cell, counterfactual_path, submission_name, dump_values=dump_values
            )
        except SubmissionFormatError as exc:
            scores["counterfactual_prediction"] = {"status": "invalid_format", "error": str(exc)}
    return scores


def evaluate_prebuilt(
    cell: dict[str, Any],
    submission_dir: Path,
    submission_name: str,
    dump_values: Path | None = None,
) -> dict[str, Any]:
    """Actual-arm entry point.

    Takes the PRE-BUILT actual-cell dict directly (from ``load_actual_cell`` / ``build_fixture_actual_cell``) — NO ``_load_cell``, NO ``cell_dir``, NO ``hidden/`` read. Scores sales forecasting (held-out forecast, which runs on both arms) and validity checks (label-free validity), and stamps the truth-requiring tasks ``not_applicable_actual_data``. Synthetic cells NEVER reach here — they go through ``evaluate(cell_dir, …)``.
    """
    if cell.get("data_arm") != "actual":
        raise ValueError(
            "evaluate_prebuilt is the actual-arm path; "
            "synthetic cells use evaluate(cell_dir, …)"
        )
    cell_slug = cell.get("cell_slug", cell.get("family", "actual"))
    scores: dict[str, Any] = _result_header(cell, cell_slug, cell.get("cell_dir"))
    scores["submission_name"] = submission_name

    # sales forecasting: score_forecasting never dereferences cell_dir → pass None; cell["transactions_full"/eval_weeks/training] are the real held-out observed sales.
    forecast_path = submission_dir / ACTUAL_FORECAST_FILE
    if not forecast_path.exists():
        scores["sales_forecasting"] = {"status": "not_submitted"}
    else:
        try:
            scores["sales_forecasting"] = score_forecasting(None, cell, forecast_path)
        except SubmissionFormatError as exc:
            scores["sales_forecasting"] = {"status": "invalid_format", "error": str(exc)}

    # validity checks: label-free validity on the public sweep + submitted Δq.
    validity_path = submission_dir / ACTUAL_VALIDITY_FILE
    if not validity_path.exists():
        scores["validity_checks_actual"] = {"status": "not_submitted"}
    else:
        try:
            scores["validity_checks_actual"] = score_validity(cell, validity_path)
        except SubmissionFormatError as exc:
            scores["validity_checks_actual"] = {"status": "invalid_format", "error": str(exc)}

    # Truth-requiring tasks do not apply on real data (no hidden truth).
    scores["elasticity_recovery"] = {"status": "not_applicable_actual_data"}
    scores["counterfactual_prediction"] = {"status": "not_applicable_actual_data"}
    scores["data_arm"] = "actual"
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-dir", required=True, type=Path)
    parser.add_argument("--submission-dir", required=True, type=Path)
    parser.add_argument("--submission-name", default="submission")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--dump-values",
        type=Path,
        default=None,
        help="Reserved; accepted and ignored.",
    )
    parser.add_argument(
        "--reference-scores",
        type=Path,
        default=None,
        help="Directory of reference scores.json files (ships with the benchmark); "
        "prints a ranked table with your model placed among them.",
    )
    args = parser.parse_args()
    scores = evaluate(
        args.cell_dir, args.submission_dir, args.submission_name, dump_values=args.dump_values
    )
    payload = json.dumps(scores, indent=2, default=float)
    if args.out:
        args.out.write_text(payload, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(payload)
    if args.reference_scores is not None:
        from causaldemand.leaderboard import leaderboard_rows

        reference_payloads = [
            json.loads(path.read_text())
            for path in sorted(args.reference_scores.glob("*.json"))
        ]
        cell_slug = Path(args.cell_dir).name
        same_cell = [
            p for p in reference_payloads if p.get("cell_slug", Path(p.get("cell_dir", "")).name) == cell_slug
        ]
        table = leaderboard_rows(same_cell + [scores])
        print(f"\n=== your placement on {cell_slug} (references + you) ===")
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
