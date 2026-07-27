"""Unit tests for the decomposed counterfactual prediction headline.

own = signed WMPE on the category-netted focal Δq (pooled); substitution = unsigned WAPE (raw-mass L1) on the category-netted competitor Δq (pooled). Both net each side by its own category shift ΔM = ΣΔq, then micro-average (pool numerators and denominators, divide once). Hand-computed deterministic cases.
"""

from __future__ import annotations

import pandas as pd
import pytest

from causaldemand.headline_decomposition import (
    decomposed_headline,
    focal_from_context,
)


def _frame(rows, store="S1", week=1) -> pd.DataFrame:
    """rows = (product_id, baseline_units, dq_true, dq_pred)."""
    return pd.DataFrame(
        [(store, week, p, b, t, h) for (p, b, t, h) in rows],
        columns=["store_id", "week", "product_id", "baseline_units", "dq_true", "dq_pred"],
    )


# --- own-price WMPE ---------------------------------------------------------


def test_own_and_sub_zero_for_perfect_prediction():
    # ΔM_true = ΔM_pred = 0, exact prediction -> both axes exactly 0.
    frame = _frame([("F", 100, -20, -20), ("A", 50, 8, 8), ("B", 50, 12, 12)])
    out = decomposed_headline(frame, "F")
    assert out["own_price_wmpe"] == pytest.approx(0.0)
    assert out["substitution_wape"] == pytest.approx(0.0)


def test_own_price_wmpe_signs_the_bias():
    # Over-shoot the focal loss (-28 vs -20); competitors keep ΔM_pred = 0 so netting is a no-op on the focal -> own WMPE = (-28 - -20)/20 = -0.4 (signed).
    frame = _frame([("F", 100, -20, -28), ("A", 50, 10, 14), ("B", 50, 10, 14)])
    out = decomposed_headline(frame, "F")
    assert out["own_price_wmpe"] == pytest.approx(-0.4)


# --- substitution WAPE ------------------------------------------------------


def test_substitution_wape_zero_for_perfect_competitors():
    frame = _frame([("F", 100, -20, -20), ("A", 50, 8, 8), ("B", 50, 6, 6), ("C", 50, 6, 6)])
    out = decomposed_headline(frame, "F")
    assert out["substitution_wape"] == pytest.approx(0.0)


def test_substitution_wape_measures_competitor_mass_error():
    # rt = {A:8, B:6, C:6}; rp = {A:8, B:10, C:2}; ΔM_pred = 0 (no netting shift). |err| = 0 + 4 + 4 = 8; |true| = 8 + 6 + 6 = 20 -> WAPE = 0.4.
    frame = _frame([("F", 100, -20, -20), ("A", 50, 8, 8), ("B", 50, 6, 10), ("C", 50, 6, 2)])
    out = decomposed_headline(frame, "F")
    assert out["substitution_wape"] == pytest.approx(0.4)


def test_category_netting_isolates_both_axes():
    # Prediction = true substitution + a WRONG category shift (ΔM_pred=-40). Netting each side by its own ΔM removes the category term -> own=0, sub=0 (a category-magnitude error is graded on neither axis).
    base = {"F": 100.0, "A": 50.0, "B": 50.0, "C": 50.0}
    total = sum(base.values())
    dq_true = {"F": -20.0, "A": 8.0, "B": 6.0, "C": 6.0}  # ΣΔq* = 0
    wrong_dm = -40.0
    rows = [(p, base[p], dq_true[p], dq_true[p] + wrong_dm * base[p] / total) for p in base]
    out = decomposed_headline(_frame(rows), "F")
    assert out["own_price_wmpe"] == pytest.approx(0.0, abs=1e-9)
    assert out["substitution_wape"] == pytest.approx(0.0, abs=1e-9)


# --- pooling (micro-average across store-weeks) -----------------------------


def test_pooled_micro_average_across_store_weeks():
    # S1 own contribution -4 (|true| 20); S2 own contribution 0 (|true| 10). Pooled: own_signed_sum -4 / own_abs_true_sum 30 -> WMPE = -4/30.
    f = pd.concat(
        [
            _frame([("F", 100, -20, -24), ("A", 50, 10, 12), ("B", 50, 10, 12)], store="S1"),
            _frame([("F", 100, -10, -10), ("A", 50, 5, 5), ("B", 50, 5, 5)], store="S2"),
        ]
    )
    out = decomposed_headline(f, "F")
    assert out["own_price_wmpe"] == pytest.approx(-4 / 30)
    assert out["n_store_weeks_scored"] == 2


# --- degenerate & missing-focal edge cases ----------------------------------


def test_pure_category_scaling_returns_none():
    # dq_true = ΔM * share exactly -> every netted residual is 0 -> denominators 0 -> both axes None (a separate degenerate case from a real 0.0).
    base = {"F": 100.0, "A": 50.0, "B": 50.0}
    total = sum(base.values())
    dm = -30.0
    rows = [(p, base[p], dm * base[p] / total, dm * base[p] / total) for p in base]
    out = decomposed_headline(_frame(rows), "F")
    assert out["own_price_wmpe"] is None
    assert out["substitution_wape"] is None


def test_focal_missing_is_counted():
    frame = _frame([("A", 50, 8, 8), ("B", 50, 6, 6)])  # no focal F row
    out = decomposed_headline(frame, "F")
    assert out["n_store_weeks_focal_missing"] == 1
    assert out["own_price_wmpe"] is None  # no focal contribution -> denominator 0


# --- helper -----------------------------------------------------------------


def test_focal_from_context_finds_moved_product():
    ctx = pd.DataFrame(
        {
            "intervention_id": ["s"] * 3,
            "product_id": ["F", "A", "B"],
            "baseline_price": [10.0, 5.0, 6.0],
            "intervention_price": [11.0, 5.0, 6.0],  # only F moves
        }
    )
    assert focal_from_context(ctx, "s") == "F"
