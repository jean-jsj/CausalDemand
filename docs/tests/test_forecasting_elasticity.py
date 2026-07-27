"""Unit tests for sales forecasting (demand WMAPE/WMPE) and elasticity recovery (elasticity) metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causaldemand.sales_forecasting import (
    build_demand_truth,
    demand_prediction_scores,
    revenue_weights,
)
from causaldemand.elasticity import (
    elasticity_scores,
    elasticity_truth_log_log,
)


# ---------------------------------------------------------------------------
# sales forecasting
# ---------------------------------------------------------------------------


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_id": ["P1", "P1", "P2", "P2"],
            "store_id": ["S1", "S2", "S1", "S2"],
            "week": [1, 1, 1, 1],
            "true_units": [100.0, 50.0, 20.0, 30.0],
        }
    )


def test_wmape_wmpe_hand_example():
    truth = _truth()
    predictions = truth.rename(columns={"true_units": "predicted_units"}).copy()
    predictions["predicted_units"] = [110.0, 40.0, 25.0, 30.0]  # errs +10, -10, +5, 0
    weights = pd.Series({"P1": 0.8, "P2": 0.2})
    scores = demand_prediction_scores(predictions, truth, weights)
    # per product: P1 abs_err 20, signed 0, abs_true 150; P2 abs_err 5, signed +5, abs_true 50.
    expected_wmape = (0.8 * 20 + 0.2 * 5) / (0.8 * 150 + 0.2 * 50)
    expected_wmpe = (0.8 * 0 + 0.2 * 5) / (0.8 * 150 + 0.2 * 50)
    assert scores["demand_wmape"] == pytest.approx(expected_wmape)
    assert scores["demand_wmpe"] == pytest.approx(expected_wmpe)
    assert scores["submission_complete"] is True


def test_missing_predictions_flagged_not_silently_dropped():
    truth = _truth()
    predictions = truth.head(3).rename(columns={"true_units": "predicted_units"})
    scores = demand_prediction_scores(predictions, truth, pd.Series({"P1": 0.5, "P2": 0.5}))
    assert scores["n_truth_rows_without_prediction"] == 1
    assert scores["submission_complete"] is False


def test_revenue_weights_sum_to_one():
    tx = pd.DataFrame({"product_id": ["P1", "P2", "P1"], "dollars": [30.0, 10.0, 60.0]})
    w = revenue_weights(tx)
    assert w.sum() == pytest.approx(1.0)
    assert w["P1"] == pytest.approx(0.9)


def test_build_demand_truth_is_observed_holdout_units():
    transactions_full = pd.DataFrame(
        {
            "product_id": ["P1", "P1", "P2"],
            "store_id": ["S1", "S1", "S1"],
            "week": [1, 2, 2],
            "units": [42.0, 17.0, 5.0],
            "dollars": [84.0, 34.0, 15.0],
        }
    )
    truth = build_demand_truth(transactions_full, [2])
    # Truth = observed holdout sales, not latent demand.
    assert sorted(truth["true_units"].tolist()) == [5.0, 17.0]
    assert set(truth["week"]) == {2}


# ---------------------------------------------------------------------------
# elasticity recovery
# ---------------------------------------------------------------------------


def _eps_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    products = ["P1", "P2", "P3"]
    star = pd.DataFrame(
        [
            [-2.0, 0.5, 0.01],
            [0.4, -1.5, -0.3],
            [0.02, 0.6, -2.5],
        ],
        index=pd.Index(products, name="affected_product_id"),
        columns=pd.Index(products, name="priced_product_id"),
    )
    hat = pd.DataFrame(
        [
            [-1.8, 0.45, 0.0],
            [0.5, -1.7, -0.25],
            [0.0, 0.55, 2.0],  # P3 own-price sign error
        ],
        index=star.index,
        columns=star.columns,
    )
    return hat, star


def test_elasticity_scores_own_block():
    hat, star = _eps_frames()
    weights = pd.Series({"P1": 0.5, "P2": 0.3, "P3": 0.2})
    scores = elasticity_scores(hat, star, weights)
    own = scores["own_price"]
    assert own["sign_accuracy"] == pytest.approx(2.0 / 3.0)
    # WMAPE: sum w|err| / sum w|true| with errs 0.2, 0.2, 4.5.
    expected = (0.5 * 0.2 + 0.3 * 0.2 + 0.2 * 4.5) / (0.5 * 2.0 + 0.3 * 1.5 + 0.2 * 2.5)
    assert own["wmape"] == pytest.approx(expected)
    assert own["rmse"] == pytest.approx(np.sqrt((0.2**2 + 0.2**2 + 4.5**2) / 3.0))


def test_elasticity_scores_cross_classes_and_ndcg():
    hat, star = _eps_frames()
    weights = pd.Series({"P1": 1 / 3, "P2": 1 / 3, "P3": 1 / 3})
    scores = elasticity_scores(hat, star, weights, unrelated_threshold_pct=0.20)
    cross = scores["cross_price"]
    # Threshold = 20th pct of |off-diag| = |{0.5,0.01,0.4,0.3,0.02,0.6}| -> 0.038
    assert cross["unrelated_abs_threshold"] == pytest.approx(
        float(np.quantile([0.5, 0.01, 0.4, 0.3, 0.02, 0.6], 0.2))
    )
    f1 = cross["f1_per_class"]
    assert set(f1.keys()) == {"substitute", "complement", "unrelated"}
    # All class counts present and consistent with 6 off-diagonal entries.
    assert sum(block["n_true"] for block in f1.values()) == 6
    assert cross["ndcg"] is not None and 0.0 < cross["ndcg"] <= 1.0
    # A perfect submission maxes the ranking metric.
    perfect = elasticity_scores(star, star, weights)
    assert perfect["cross_price"]["ndcg"] == pytest.approx(1.0)
    assert perfect["own_price"]["sign_accuracy"] == 1.0
    assert perfect["own_price"]["wmape"] == pytest.approx(0.0)


def test_missing_entries_scored_as_zero_and_counted():
    hat, star = _eps_frames()
    weights = pd.Series({"P1": 1 / 3, "P2": 1 / 3, "P3": 1 / 3})
    partial = hat.drop(columns=["P3"])
    scores = elasticity_scores(partial, star, weights)
    assert scores["n_matrix_entries_missing_in_submission"] == 3
    assert scores["submission_complete"] is False


def test_log_log_truth_closed_form():
    dgp = pd.DataFrame({"product_id": ["P1", "P2"], "own_elasticity": [-2.0, -1.5]})
    cross = np.array([[0.0, 0.1], [0.2, 0.0]])  # cross[priced j, affected i]
    eps = elasticity_truth_log_log(dgp, cross, ["P1", "P2"])
    # Own arc: (1.01^e - 1)/0.01.
    assert eps.loc["P1", "P1"] == pytest.approx((1.01**-2.0 - 1) / 0.01)
    # Cross: effect on P1 of P2's price = cross[1, 0] = 0.2.
    assert eps.loc["P1", "P2"] == pytest.approx((1.01**0.2 - 1) / 0.01)
    assert eps.loc["P2", "P1"] == pytest.approx((1.01**0.1 - 1) / 0.01)


# ---------------------------------------------------------------------------
# incidence: total truth, conditional-basis classes
# ---------------------------------------------------------------------------


def _incidence_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Conditional (switching) truth, per-priced-product incidence shift, total.

    ε_total(i,j) = ε_M(j) + ε_cond(i,j) exactly; the C↔A pairs have ZERO switching, so on totals they look like complements (the incidence shift), while on the conditional basis they are unrelated.
    """
    products = ["P1", "P2", "P3"]
    cond = pd.DataFrame(
        [
            [-2.00, 0.50, 0.00],
            [0.40, -1.80, 0.05],
            [0.00, 0.02, -1.50],
        ],
        index=pd.Index(products, name="affected_product_id"),
        columns=pd.Index(products, name="priced_product_id"),
    )
    incidence = pd.Series([-0.30, -0.25, -0.20], index=cond.columns)
    total = cond.add(incidence, axis="columns")
    return cond, incidence.to_frame().T, total


def test_classification_basis_is_conditional_when_supplied():
    cond, _, total = _incidence_frames()
    weights = pd.Series({"P1": 0.5, "P2": 0.3, "P3": 0.2})
    scores = elasticity_scores(total, total, weights, eps_star_conditional=cond)
    cross = scores["cross_price"]
    assert cross["classification_basis"] == "conditional_switching_incidence_netted_out"
    # Threshold computed on |conditional| off-diagonals, not totals.
    off = ~np.eye(3, dtype=bool)
    expected = float(np.quantile(np.abs(cond.to_numpy()[off]), 0.20))
    assert cross["unrelated_abs_threshold"] == pytest.approx(expected)
    # A submission with perfect totals earns perfect classes (its conditional implication ε̂ − ε*_M(j) reproduces the true switching matrix exactly).
    for block in cross["f1_per_class"].values():
        assert block["f1"] in (None, 1.0)
    # Magnitude blocks stay on totals: perfect totals ⇒ zero error.
    assert cross["all_pairs"]["wmape"] == pytest.approx(0.0)
    assert scores["truth_definition"] == "total_effect_incidence_plus_switching"


def test_classification_falls_back_to_total_without_conditional():
    cond, _, total = _incidence_frames()
    weights = pd.Series({"P1": 0.5, "P2": 0.3, "P3": 0.2})
    with_cond = elasticity_scores(total, total, weights, eps_star_conditional=cond)
    without = elasticity_scores(total, total, weights)
    assert without["cross_price"]["classification_basis"] == "total"
    # On the conditional basis no pair is a complement (the fixture has no true complementarity); on totals the common negative incidence shift manufactures apparent complements — the semantic muddying the conditional basis avoids.
    n_complement_cond = with_cond["cross_price"]["f1_per_class"]["complement"]["n_true"]
    n_complement_total = without["cross_price"]["f1_per_class"]["complement"]["n_true"]
    assert n_complement_cond == 0
    assert n_complement_total > 0
