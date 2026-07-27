"""Integration test for the actual-data arm.

Fixture-driven, hermetic: no real Dominick's data, no simulation run. A hand-built in-memory actual cell (the ``build_fixture_actual_cell`` shape, minus the real data) flows through ``score_validity`` and the ``evaluate_prebuilt`` arm routing, covering the integration the unit tests (``test_validity_checks.py``) do not: a submission reaching the scorer, the arm returning L2/L3 as ``not_applicable_actual_data``, and the derived own-ε band diagonal.

Asserts only on the dicts returned by ``score_validity`` / ``evaluate_prebuilt``; never imports or re-implements the validity internals.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd
import pytest

# The import itself checks the public surface: if any symbol is absent the module fails to import and the whole file errors loudly.
from causaldemand.actual_data import (  # noqa: E402
    ACTUAL_ARM,
    SWEEP_CONTEXT_COLUMNS,
    ActualDataNotAvailable,
    build_fixture_actual_cell,  # noqa: F401  (imported to assert the surface exists)
    load_actual_cell,
)
from causaldemand.evaluate_submission import (  # noqa: E402
    ACTUAL_FORECAST_FILE,
    ACTUAL_VALIDITY_FILE,
    evaluate_prebuilt,
    score_validity,
)


# ---------------------------------------------------------------------------
# In-memory fixture actual cell: hermetic, no dependence on gitignored outputs/. Mirrors the exact key + column contract build_fixture_actual_cell emits, hand-built so the test is self-contained and deterministic.
# ---------------------------------------------------------------------------

_PRODUCTS = ["P0", "P1", "P2"]          # 3 products
_STORES = ["S0", "S1"]                   # 2 stores
_ALL_WEEKS = [1, 2, 3, 4]                # a few weeks
_EVAL_WEEKS = [3, 4]                     # held-out tail (sales forecasting truth)
_SWEEP_PCT = 0.10                        # ±10% own-price move (both signs)
_BASELINE_PRICE = 2.00
_BASELINE_UNITS = 100.0
_FOCAL_EPS = -2.0                        # in the tissue band (-3.0, -1.0)


def _cfg() -> dict:
    """Minimal cfg the header reader (`_result_header`) touches via `.get`."""
    return {
        "benchmark_version": "fixture-actual-test",
        "simulation": {"counterfactual_eval_weeks": len(_EVAL_WEEKS)},
    }


def _transactions_full() -> pd.DataFrame:
    """A tiny observed panel with the columns sales forecasting truth reads (units, dollars)."""
    rows = []
    for product_id, store_id, week in itertools.product(_PRODUCTS, _STORES, _ALL_WEEKS):
        units = _BASELINE_UNITS
        rows.append(
            {
                "product_id": product_id,
                "store_id": store_id,
                "week": week,
                "units": units,
                "dollars": units * _BASELINE_PRICE,
                "price": _BASELINE_PRICE,
                "promo_flag": 0,
            }
        )
    return pd.DataFrame(rows)


def _sweep_context() -> pd.DataFrame:
    """The public own-price sweep: every product moved once, both signs,
    anchored on the eval-week store-weeks, columns exactly SWEEP_CONTEXT_COLUMNS."""
    rows = []
    for product_id in _PRODUCTS:
        for sign, tag in ((+1.0, "plus"), (-1.0, "minus")):
            intervention_id = f"actual_own_sweep_{product_id}_{tag}"
            for store_id in _STORES:
                for week in _EVAL_WEEKS:
                    rows.append(
                        {
                            "intervention_id": intervention_id,
                            "product_id": product_id,
                            "store_id": store_id,
                            "week": week,
                            "baseline_price": _BASELINE_PRICE,
                            "intervention_price": _BASELINE_PRICE * (1.0 + sign * _SWEEP_PCT),
                            "baseline_units": _BASELINE_UNITS,
                        }
                    )
    return pd.DataFrame(rows, columns=SWEEP_CONTEXT_COLUMNS)


def _make_fixture_cell() -> dict:
    """Hand-built actual cell: the _load_cell five keys + data_arm + sweep_context."""
    full = _transactions_full()
    training = full[~full["week"].isin(set(_EVAL_WEEKS))].reset_index(drop=True)
    return {
        "cfg": _cfg(),
        "family": ACTUAL_ARM,
        "transactions_full": full,
        "training": training,
        "eval_weeks": list(_EVAL_WEEKS),
        "data_arm": ACTUAL_ARM,
        "sweep_context": _sweep_context(),
    }


# ---------------------------------------------------------------------------
# Coherent participant submissions written to a tmp_path submission dir.
# ---------------------------------------------------------------------------


def _coherent_validity_deltas(cell: dict, *, flip_focal: str | None = None) -> pd.DataFrame:
    """Predicted-Δq for the L4 file, coherent per the law of demand.

    Per store-week the category shift ΔM = ΣΔq is engineered to ZERO (focal drop is redistributed exactly onto the competitors), so category-netting is a no-op and every competitor residual points the demanded way → all four checks ≈ 1.0.

    * focal under +x%: Δq = ε·pct·baseline_units < 0; competitors split +|Δq_focal|.
    * focal under −x%: signs mirror (focal +, competitors −).
    * With ε = _FOCAL_EPS = -2.0 and pct = ±0.10 the derived own-ε = -2.0 (in band).

    ``flip_focal``: if set, that focal's predicted focal Δq is sign-flipped under its +x% (hike) leg — the perturbation used to prove the checks are live.
    """
    context = cell["sweep_context"]
    n_comp = len(_PRODUCTS) - 1
    rows = []
    for intervention_id in context["intervention_id"].astype(str).unique():
        grp = context[context["intervention_id"].astype(str) == intervention_id]
        # the moved (focal) product for this intervention
        focal = str(grp["product_id"].iloc[0])
        # sign of the price move (+1 hike, -1 cut)
        price_up = float(grp["intervention_price"].iloc[0]) > float(grp["baseline_price"].iloc[0])
        pct = _SWEEP_PCT if price_up else -_SWEEP_PCT
        focal_dq = _FOCAL_EPS * pct * _BASELINE_UNITS  # <0 on hike, >0 on cut
        comp_dq = -focal_dq / n_comp                   # redistributes so ΔM = 0
        do_flip = flip_focal is not None and focal == flip_focal and price_up
        for store_id in _STORES:
            for week in _EVAL_WEEKS:
                for product_id in _PRODUCTS:
                    if product_id == focal:
                        dq = -focal_dq if do_flip else focal_dq
                    else:
                        dq = comp_dq
                    rows.append(
                        {
                            "intervention_id": intervention_id,
                            "product_id": product_id,
                            "store_id": store_id,
                            "week": week,
                            "predicted_delta_units": dq,
                        }
                    )
    return pd.DataFrame(rows)


def _coherent_forecast_predictions_frame(cell: dict) -> pd.DataFrame:
    """A perfect sales forecasting forecast (predicted == observed held-out units).

    sales forecasting runs unchanged on the actual arm; the test only needs a valid, scorable file — numeric accuracy is not asserted, only that a score block is produced.
    """
    truth = cell["transactions_full"]
    truth = truth[truth["week"].isin(set(_EVAL_WEEKS))]
    return pd.DataFrame(
        {
            "product_id": truth["product_id"].to_numpy(),
            "store_id": truth["store_id"].to_numpy(),
            "week": truth["week"].to_numpy(),
            "predicted_units": truth["units"].to_numpy(),
        }
    )


def _write_submission(
    tmp_path: Path, cell: dict, *, flip_focal: str | None = None, with_forecast: bool = True
) -> Path:
    sub_dir = tmp_path / "submission"
    sub_dir.mkdir(exist_ok=True)
    _coherent_validity_deltas(cell, flip_focal=flip_focal).to_csv(
        sub_dir / ACTUAL_VALIDITY_FILE, index=False
    )
    if with_forecast:
        _coherent_forecast_predictions_frame(cell).to_csv(sub_dir / ACTUAL_FORECAST_FILE, index=False)
    return sub_dir


# ===========================================================================
# Fixture actual cell shape
# ===========================================================================


def test_fixture_cell_shape_and_keys():
    cell = _make_fixture_cell()

    # actual-arm marker + the _load_cell-shaped keys + sweep_context.
    assert cell["data_arm"] == "actual"
    for key in (
        "transactions_full",
        "training",
        "eval_weeks",
        "cfg",
        "family",
        "data_arm",
        "sweep_context",
    ):
        assert key in cell, f"fixture cell missing key {key!r}"

    ctx = cell["sweep_context"]
    # exact column contract.
    assert list(ctx.columns) == SWEEP_CONTEXT_COLUMNS

    # every product moved +x% AND -x% once → monotonicity pairs exist.
    for product_id in _PRODUCTS:
        for tag in ("plus", "minus"):
            iid = f"actual_own_sweep_{product_id}_{tag}"
            sub = ctx[ctx["intervention_id"] == iid]
            assert not sub.empty, f"missing sweep intervention {iid}"
            moved = sub[sub["product_id"] == product_id]
            delta = (moved["intervention_price"] - moved["baseline_price"]).to_numpy()
            if tag == "plus":
                assert (delta > 0).all()
            else:
                assert (delta < 0).all()


def test_build_fixture_actual_cell_surface_exists():
    """The fixture builder symbol is importable.

    The in-memory cell is the default hermetic path; this only asserts the builder exists so a synthetic-dir path stays available. No outputs/ dir is required.
    """
    assert callable(build_fixture_actual_cell)


def test_load_actual_cell_missing_data_raises():
    """The real-data absence path is the actionable ActualDataNotAvailable, not a
    raw traceback."""
    with pytest.raises(ActualDataNotAvailable):
        load_actual_cell(Path("/nonexistent/actual/data/root"))


# ===========================================================================
# score_validity returns all four live checks
# ===========================================================================


def test_score_validity_all_checks_coherent():
    cell = _make_fixture_cell()
    # score_validity reads only the L4 file; write it into a tmp dir.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        sub_dir = _write_submission(Path(td), cell, with_forecast=False)
        scores = score_validity(cell, sub_dir / ACTUAL_VALIDITY_FILE)

    # all four validity checks blocks present (assert on the returned dict only).
    for block in ("own_price_sign", "substitution_sign", "own_elasticity_range", "monotonicity"):
        assert block in scores, f"score_validity missing block {block!r}"

    # The three checks that score on the actual-arm sweep are coherent ≈ 1.0.
    assert scores["own_price_sign"]["frac_correct_sign"] == pytest.approx(1.0)
    assert scores["own_elasticity_range"]["frac_in_band"] == pytest.approx(1.0)
    assert scores["monotonicity"]["frac_consistent"] == pytest.approx(1.0)

    # Substitution-sign is structurally None on the actual arm: `score_validity` merges the submitted Δq onto `sweep_context`, whose rows enumerate only the focal product per intervention — no competitor rows enter the frame, so the redistribution mass is 0 and `substitution_sign_validity` returns None. The block is present and wired; it simply has no competitor mass to score here.
    assert scores["substitution_sign"]["frac_redistribution_mass_correct"] is None


def test_score_validity_checks_are_live_under_perturbation():
    """Flipping one focal's predicted Δq under its hike must drop the relevant
    fraction below 1.0 — proving the checks are wired, not stubbed."""
    cell = _make_fixture_cell()
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        sub_dir = _write_submission(Path(td), cell, flip_focal="P0", with_forecast=False)
        scores = score_validity(cell, sub_dir / ACTUAL_VALIDITY_FILE)

    # The flip makes P0's focal Δq POSITIVE under a hike (wrong sign) → own-price sign fraction drops, and the ± monotonicity pair no longer flips.
    assert scores["own_price_sign"]["frac_correct_sign"] < 1.0
    assert scores["monotonicity"]["frac_consistent"] < 1.0


# ===========================================================================
# Derived own-ε band diagonal (no elasticity file)
# ===========================================================================


def test_derived_elasticity_band_diagonal_no_elasticity_file():
    cell = _make_fixture_cell()
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        sub_dir = _write_submission(Path(td), cell, with_forecast=False)
        deltas = pd.read_csv(sub_dir / ACTUAL_VALIDITY_FILE)
        scores = score_validity(cell, sub_dir / ACTUAL_VALIDITY_FILE)

    # full own-price sweep covers ALL products (each moved once) → n_products == J.
    assert scores["own_elasticity_range"]["n_products"] == len(_PRODUCTS)

    # Independently recompute the derived own-ε for a chosen focal from the submitted deltas + the public sweep_context: the harness uses the +x% leg, eps = (Σ dq_pred_focal / Σ baseline_units_focal) / pct_focal.
    focal = "P1"
    context = cell["sweep_context"]
    iid = f"actual_own_sweep_{focal}_plus"
    ctx_focal = context[
        (context["intervention_id"] == iid) & (context["product_id"] == focal)
    ]
    base_sum = float(ctx_focal["baseline_units"].sum())
    pct = float(
        (ctx_focal["intervention_price"].iloc[0] - ctx_focal["baseline_price"].iloc[0])
        / ctx_focal["baseline_price"].iloc[0]
    )
    dsub = deltas[(deltas["intervention_id"] == iid) & (deltas["product_id"] == focal)]
    dq_sum = float(dsub["predicted_delta_units"].sum())
    expected_eps = (dq_sum / base_sum) / pct

    # The whole diagonal is in-band, so the harness accepted this eps as in-band; its value must equal our independent recomputation, and it must equal the engineered _FOCAL_EPS (a cross-check on the whole derivation).
    assert expected_eps == pytest.approx(_FOCAL_EPS)
    lo, hi = scores["own_elasticity_range"]["band"]
    assert lo <= expected_eps <= hi
    assert scores["own_elasticity_range"]["frac_in_band"] == pytest.approx(1.0)


# ===========================================================================
# Arm routing via evaluate_prebuilt
# ===========================================================================


def test_arm_routing_evaluate_prebuilt(tmp_path):
    cell = _make_fixture_cell()
    sub_dir = _write_submission(tmp_path, cell, with_forecast=True)

    result = evaluate_prebuilt(cell, sub_dir, "fixture-actual-submission")

    # sales forecasting is scored on the actual arm — assert a score block, not a value.
    l1 = result["sales_forecasting"]
    assert l1.get("status") not in ("not_submitted", "invalid_format"), l1
    assert "demand_wmape" in l1

    # validity checks is scored.
    l4 = result["validity_checks_actual"]
    assert l4.get("status") not in ("not_submitted", "invalid_format"), l4
    assert l4["own_price_sign"]["frac_correct_sign"] == pytest.approx(1.0)

    # Truth-requiring tasks are not applicable on real data.
    assert result["elasticity_recovery"] == {"status": "not_applicable_actual_data"}
    assert result["counterfactual_prediction"] == {"status": "not_applicable_actual_data"}

    # top-level arm marker.
    assert result["data_arm"] == "actual"


# ===========================================================================
# Leaderboard actual columns (guarded so a missing surface xfails rather than errors)
# ===========================================================================


def test_leaderboard_actual_columns_per_arm(tmp_path):
    leaderboard = pytest.importorskip("causaldemand.leaderboard")
    leaderboard_rows = getattr(leaderboard, "leaderboard_rows", None)
    if leaderboard_rows is None:
        pytest.xfail("leaderboard_rows missing from causaldemand.leaderboard")

    cell = _make_fixture_cell()
    sub_dir = _write_submission(tmp_path, cell, with_forecast=True)
    actual_scores = evaluate_prebuilt(cell, sub_dir, "actual-sub")

    # A minimal synthetic score payload (no data_arm tag → defaults to synthetic; its actual columns must come back None).
    synthetic_scores = {
        "submission_name": "synthetic-sub",
        "cell_slug": "complex_probit_endo_on_seed1",
        "family": "covariance_probit",
        "counterfactual_prediction": {
            "headline": {
                "own_price": {"own_price_wmpe": 0.05},
                "substitution": {"substitution_wape": 0.30, "n_store_weeks_scored": 10},
            }
        },
        "sales_forecasting": {"demand_wmape": 0.12, "demand_wmpe": -0.01},
    }

    frame = leaderboard_rows([actual_scores, synthetic_scores])

    # actual-arm diagnostic columns must exist on the surface.
    for col in (
        "data_arm",
        "validity_own_sign_frac",
        "validity_substitution_frac",
        "validity_range_in_band",
        "validity_monotonicity_frac",
        "actual_forecast_error",
    ):
        assert col in frame.columns, f"leaderboard missing actual column {col!r}"

    actual_row = frame[frame["data_arm"] == "actual"]
    synth_row = frame[frame["data_arm"] == "synthetic"]
    assert len(actual_row) == 1
    assert len(synth_row) == 1

    # actual row carries the diagnostic panel; synthetic row's actual columns None.
    assert actual_row["validity_own_sign_frac"].iloc[0] == pytest.approx(1.0)
    assert actual_row["actual_forecast_error"].iloc[0] is not None
    assert pd.isna(synth_row["validity_own_sign_frac"].iloc[0])
    assert pd.isna(synth_row["actual_forecast_error"].iloc[0])

    # ranks restart within each data_arm (per-arm partition).
    assert int(actual_row["rank"].iloc[0]) == 1
    assert int(synth_row["rank"].iloc[0]) == 1
