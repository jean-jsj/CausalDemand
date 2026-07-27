"""Unit tests for validity checks validity checks (label-free, ground-truth-free).

Every metric reads only predictions + public price moves — no ``dq_true``, no ``eps_star`` — so these scores work on real POS data. Hand-computed cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causaldemand.validity_checks import (
    coherence_gate,
    cross_elasticity_plausibility,
    own_elasticity_range_coverage,
    own_price_sign_validity,
    price_direction_from_context,
    substitution_sign_validity,
    sweep_monotonicity,
    validity_scores,
)


def _frame(rows, store="S1", week=1):
    """rows = (product_id, baseline_units, dq_pred). No dq_true: label-free."""
    return pd.DataFrame(
        [(store, week, p, b, h) for (p, b, h) in rows],
        columns=["store_id", "week", "product_id", "baseline_units", "dq_pred"],
    )


# --- own-price sign validity ------------------------------------------------


def test_own_sign_hike_rewards_negative_focal():
    frame = _frame([("F", 100, -20), ("A", 50, 8), ("B", 50, 12)])
    out = own_price_sign_validity(frame, "F", price_increase=True)
    assert out["frac_correct_sign"] == pytest.approx(1.0)
    assert out["expected_sign"] == -1


def test_own_sign_wrong_direction_caught():
    # Focal units rise under a hike — violates the law of demand.
    frame = _frame([("F", 100, +20), ("A", 50, -8), ("B", 50, -12)])
    out = own_price_sign_validity(frame, "F", price_increase=True)
    assert out["frac_correct_sign"] == pytest.approx(0.0)
    assert out["frac_wrong_sign"] == pytest.approx(1.0)


def test_own_sign_zero_is_ambiguous_not_scored():
    frame = _frame([("F", 100, 0.0), ("A", 50, 0.0)])
    out = own_price_sign_validity(frame, "F", price_increase=True)
    assert out["n_evaluated"] == 0
    assert out["frac_correct_sign"] is None
    assert out["n_ambiguous_zero"] == 1


# --- substitution sign validity ---------------------------------------------


def test_substitution_sign_all_correct_under_hike():
    # Hike: competitors gain, focal loses, sum-zero so netting is a no-op.
    frame = _frame([("F", 100, -20), ("A", 50, 12), ("B", 50, 8)])
    out = substitution_sign_validity(frame, "F", price_increase=True)
    assert out["frac_redistribution_mass_correct"] == pytest.approx(1.0)


def test_substitution_sign_contraction_nets_to_zero():
    # Pure proportional contraction: residual ~0 → no substitution mass.
    rows = [(p, b, -30.0 * b / 200.0) for (p, b) in [("F", 100), ("A", 50), ("B", 50)]]
    out = substitution_sign_validity(_frame(rows), "F", price_increase=True)
    assert out["total_competitor_mass"] == pytest.approx(0.0, abs=1e-9)
    assert out["frac_redistribution_mass_correct"] is None


# --- own-elasticity range coverage ------------------------------------------


def test_range_coverage_counts_band_and_violations():
    eps = pd.DataFrame(np.diag([-1.8, -0.4, +0.5, -50.0]))  # band/band-edge/wrong/extreme
    out = own_elasticity_range_coverage(eps, band=(-5.0, -0.5), extreme_abs=8.0)
    assert out["frac_in_band"] == pytest.approx(1 / 4)  # only -1.8
    assert out["frac_wrong_sign"] == pytest.approx(1 / 4)  # +0.5
    assert out["frac_extreme"] == pytest.approx(1 / 4)  # -50
    assert out["frac_correct_sign"] == pytest.approx(3 / 4)


# --- monotonicity across a ± sweep ------------------------------------------


def test_monotonicity_flips_sign():
    up = _frame([("F", 100, -20), ("A", 50, 20)])
    dn = _frame([("F", 100, +18), ("A", 50, -18)])
    out = sweep_monotonicity(up, dn, "F")
    assert out["frac_consistent"] == pytest.approx(1.0)
    bad = sweep_monotonicity(up, up, "F")  # both negative -> inconsistent
    assert bad["frac_consistent"] == pytest.approx(0.0)


# --- context direction + bundle ---------------------------------------------


def test_price_direction_from_context():
    ctx = pd.DataFrame(
        {"intervention_id": ["s"] * 2, "product_id": ["F", "A"],
         "baseline_price": [10.0, 5.0], "intervention_price": [11.0, 5.0]}
    )
    assert price_direction_from_context(ctx, "s") == 1


def test_validity_scores_bundle_label_free():
    frame = _frame([("F", 100, -20), ("A", 50, 12), ("B", 50, 8)])
    eps = pd.DataFrame(np.diag([-1.8, -1.2, -2.0]))
    out = validity_scores(frame, "F", price_increase=True, eps_hat=eps)
    assert out["own_price_sign"]["frac_correct_sign"] == 1.0
    assert out["substitution_sign"]["frac_redistribution_mass_correct"] == 1.0
    assert out["own_elasticity_range"]["frac_in_band"] == 1.0
    # Cross plausibility + the gate verdict are bundled in the output.
    assert "cross_elasticity_plausibility" in out
    assert out["gate"]["verdict"] == "PASS"


# --- bootstrap CIs ----------------------------------------------------------


def _multiweek(rows_per_week):
    """rows_per_week: list of (week, rows) -> concatenated frame."""
    return pd.concat([_frame(rows, week=w) for (w, rows) in rows_per_week], ignore_index=True)


def test_own_sign_ci_brackets_point_and_saturates_when_unanimous():
    # 3 correct + 1 wrong -> point 0.75, CI a valid sub-interval of [0, 1].
    frame = _multiweek([
        (1, [("F", 100, -20)]), (2, [("F", 100, -18)]),
        (3, [("F", 100, -22)]), (4, [("F", 100, +5)]),
    ])
    out = own_price_sign_validity(frame, "F", price_increase=True, n_boot=300)
    assert out["frac_correct_sign"] == pytest.approx(0.75)
    ci = out["ci"]
    assert 0.0 <= ci["ci_low"] <= 0.75 <= ci["ci_high"] <= 1.0
    # Unanimous correct -> bootstrap of all ones is degenerate at 1.0.
    allc = _multiweek([(1, [("F", 100, -20)]), (2, [("F", 100, -10)])])
    assert own_price_sign_validity(allc, "F", price_increase=True, n_boot=100)["ci"]["ci_low"] == 1.0


def test_ci_skipped_when_n_boot_zero():
    frame = _frame([("F", 100, -20)])
    assert own_price_sign_validity(frame, "F", price_increase=True, n_boot=0)["ci"] is None


# --- unweighted count vs mass-weighted --------------------------------------


def test_substitution_reports_both_count_and_mass():
    # One big correct competitor hides two small wrong ones under mass-weighting, but the unweighted count exposes them. ΣΔq=0 so netting is a no-op.
    frame = _frame([("F", 100, -26), ("A", 50, 30), ("B", 50, -2), ("C", 50, -2)])
    out = substitution_sign_validity(frame, "F", price_increase=True, n_boot=200)
    assert out["frac_redistribution_mass_correct"] == pytest.approx(30 / 34)  # mass hides them
    assert out["frac_competitors_correct_count"] == pytest.approx(1 / 3)  # count reveals them
    assert out["n_competitor_observations_scored"] == 3
    assert out["mass_ci"] is not None and out["count_ci"] is not None


# --- complements flip the expected sign -------------------------------------


def test_complements_flip_expected_sign():
    # Hike: A,C are substitutes (gain), B is a complement (loses with focal). ΣΔq = -16 + 12 - 4 + 8 = 0.
    frame = _frame([("F", 100, -16), ("A", 50, 12), ("B", 50, -4), ("C", 50, 8)])
    naive = substitution_sign_validity(frame, "F", price_increase=True)
    assert naive["frac_redistribution_mass_correct"] == pytest.approx(20 / 24)  # B scored wrong
    with_comp = substitution_sign_validity(frame, "F", price_increase=True, complements=["B"])
    assert with_comp["frac_redistribution_mass_correct"] == pytest.approx(1.0)
    assert with_comp["frac_competitors_correct_count"] == pytest.approx(1.0)
    assert with_comp["n_complements_seen"] == 1


# --- cross-elasticity plausibility ------------------------------------------


def test_cross_elasticity_plausibility():
    eps = pd.DataFrame(
        [[-2.0, 0.3, 0.1], [0.5, -1.5, 9.0], [0.2, -0.4, -1.0]]
    )
    out = cross_elasticity_plausibility(eps, extreme_abs=8.0, expected_cross_sign=1)
    assert out["n_cross_entries"] == 6
    assert out["frac_cross_extreme"] == pytest.approx(1 / 6)  # the 9.0 entry
    assert out["frac_cross_exceeds_own"] == pytest.approx(1 / 6)  # 9.0 > own |−1.0|
    assert out["frac_cross_matches_prior"] == pytest.approx(5 / 6)  # one negative cross


# --- coherence gate verdict -------------------------------------------------


def test_coherence_gate_verdicts():
    good = {
        "own_price_sign": {"frac_correct_sign": 1.0},
        "substitution_sign": {"frac_redistribution_mass_correct": 0.9},
        "own_elasticity_range": {"frac_in_band": 1.0, "frac_wrong_sign": 0.0},
        "monotonicity": {"frac_consistent": 1.0},
    }
    assert coherence_gate(good)["verdict"] == "PASS"
    # Own-price sign below the floor is a hard fail (law of demand).
    bad = dict(good, own_price_sign={"frac_correct_sign": 0.2})
    assert coherence_gate(bad)["verdict"] == "FAIL"
    # Weak substitution is only a soft warning.
    weak = dict(good, substitution_sign={"frac_redistribution_mass_correct": 0.3})
    assert coherence_gate(weak)["verdict"] == "WARN"

