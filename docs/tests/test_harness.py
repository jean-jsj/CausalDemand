"""Unit tests for the metrics/ harness helpers (leaderboard assembly)."""

from __future__ import annotations

import pytest

from causaldemand.leaderboard import aggregate_seeds, leaderboard_rows, to_markdown


def _scores(
    name: str,
    own_wmpe: float | None,
    sub_wape: float = 0.30,
    cell_slug: str = "complex_log_log_endogenous_seed001",
) -> dict:
    return {
        "schema_version": 2,
        "submission_name": name,
        "cell_dir": f"outputs/{cell_slug}",
        "cell_slug": cell_slug,
        "sales_forecasting": {"demand_wmape": 0.1, "demand_wmpe": -0.02},
        "elasticity_recovery": {
            "own_price": {"sign_accuracy": 0.9, "wmape": 0.3, "wmpe": 0.1}
        },
        "counterfactual_prediction": {
            "headline": {
                "scenario": "sweep_single_share_highest_plus10",
                "rank_metric": "own_price.abs_own_price_wmpe",
                "own_price": {
                    "own_price_wmpe": own_wmpe,
                    "n_store_weeks_focal_missing": 0,
                },
                "substitution": {
                    "substitution_wape": sub_wape,
                    "n_store_weeks_scored": 100,
                },
            }
        },
    }


def test_leaderboard_ranks_by_abs_own_wmpe_headline():
    # `weak` carries the larger |own-price WMPE| (0.30 vs 0.12) → the closer-to- zero identification bias (`strong`) must rank first.
    table = leaderboard_rows(
        [_scores("weak", -0.30, sub_wape=0.5), _scores("strong", -0.12, sub_wape=0.2)]
    )
    assert table.iloc[0]["submission"] == "strong"
    assert table.iloc[0]["rank"] == 1
    assert table.iloc[0]["own_price_bias"] == pytest.approx(-0.12)
    assert table.iloc[0]["substitution_error"] == pytest.approx(0.2)
    assert table.iloc[1]["submission"] == "weak"


def test_leaderboard_handles_not_submitted_tasks():
    table = leaderboard_rows(
        [
            {
                "submission_name": "l3_only",
                "cell_slug": "complex_log_log_endogenous_seed001",
                "sales_forecasting": {"status": "not_submitted"},
                "elasticity_recovery": {"status": "not_submitted"},
                "counterfactual_prediction": {
                    "headline": {
                        "own_price": {"own_price_wmpe": -0.15},
                        "substitution": {"substitution_wape": 0.7},
                    }
                },
            }
        ]
    )
    value = table.iloc[0]["forecast_error"]
    assert value is None or value != value  # None or NaN
    assert table.iloc[0]["own_price_bias"] == pytest.approx(-0.15)
    assert table.iloc[0]["substitution_error"] == pytest.approx(0.7)


def test_aggregate_seeds_means_and_spread():
    payloads = [
        _scores("m", -0.10, sub_wape=0.80, cell_slug="complex_log_log_endogenous_seed002"),
        _scores("m", -0.20, sub_wape=0.70, cell_slug="complex_log_log_endogenous_seed003"),
        _scores("m", -0.30, sub_wape=0.90, cell_slug="complex_log_log_endogenous_seed004"),
    ]
    table = aggregate_seeds(leaderboard_rows(payloads))
    assert len(table) == 1
    row = table.iloc[0]
    assert row["cell_type"] == "complex_log_log_endogenous"
    assert row["n_seeds"] == 3
    # own-price WMPE mean = (-0.10 - 0.20 - 0.30) / 3 = -0.20; substitution WAPE mean = (0.80 + 0.70 + 0.90) / 3 = 0.80.
    assert row["own_price_bias"] == pytest.approx(-0.20)
    assert row["substitution_error"] == pytest.approx(0.80)
    assert row["substitution_error_seed_sd"] == pytest.approx(0.0816496, abs=1e-5)


def test_markdown_render_includes_spread():
    payloads = [
        _scores("m", -0.10, sub_wape=0.80, cell_slug="complex_log_log_endogenous_seed002"),
        _scores("m", -0.20, sub_wape=0.90, cell_slug="complex_log_log_endogenous_seed004"),
    ]
    md = to_markdown(aggregate_seeds(leaderboard_rows(payloads)))
    assert md.startswith("| rank |")
    # substitution WAPE: mean (0.80 + 0.90) / 2 = 0.850, sd (ddof=0) = 0.050.
    assert "0.850 ± 0.050" in md


def test_read_submission_rejects_missing_columns(tmp_path):
    from causaldemand.evaluate_submission import (
        COUNTERFACTUAL_COLUMNS,
        SubmissionFormatError,
        _read_submission,
    )

    p = tmp_path / "counterfactual_deltas.csv"
    p.write_text("intervention_id,product_id\nx,P1\n")  # missing store_id/week/predicted_delta_units
    with pytest.raises(SubmissionFormatError) as e:
        _read_submission(p, COUNTERFACTUAL_COLUMNS, "counterfactual prediction")
    msg = str(e.value)
    assert "missing required column" in msg
    assert "predicted_delta_units" in msg
    assert "SUBMISSION_FORMAT.md" in msg


def test_read_submission_accepts_valid(tmp_path):
    from causaldemand.evaluate_submission import ELASTICITY_COLUMNS, _read_submission

    p = tmp_path / "elasticity_matrix.csv"
    p.write_text("affected_product_id,priced_product_id,elasticity\nP1,P2,0.3\n")
    df = _read_submission(p, ELASTICITY_COLUMNS, "elasticity recovery")
    assert len(df) == 1


def test_diagnostics_tables_flatten_scores():
    from causaldemand.diagnostics import build_tables

    scores = {
        "cell_slug": "complex_log_log_endogenous_seed001",
        "sales_forecasting": {"demand_wmape": 0.2, "demand_wmpe": -0.05},
        "elasticity_recovery": {
            "own_price": {"sign_accuracy": 1.0, "wmape": 0.1, "rmse": 0.3, "wmpe": 0.05, "mean_signed_error": 0.02},
            "cross_price": {
                "ndcg": 0.9,
                "ndcg_at_5": 0.95,
                "f1_per_class": {"substitute": {"f1": 0.8}, "complement": {"f1": None}, "unrelated": {"f1": 0.7}},
                "all_pairs": {"wmape": 0.5, "rmse": 0.2, "wmpe": 0.1, "mean_signed_error": 0.01},
                "by_true_class": {"substitute": {"wmape": 0.4, "rmse": 0.2, "wmpe": 0.1}},
                "unrelated_abs_threshold": 0.03,
            },
        },
        "counterfactual_prediction": {
            "interventions": [
                {
                    "intervention_id": "sweep_single_share_highest_plus10",
                    "substitution_wape": 0.72,
                },
                {
                    "intervention_id": "sweep_brand_leading_minus10",
                    "substitution_wape": 0.41,
                },
            ]
        },
    }
    l12, l3 = build_tables([scores])
    assert l12.iloc[0]["l1_demand_wmape"] == 0.2
    assert l12.iloc[0]["l2_cross_f1_substitute"] == 0.8
    assert l12.iloc[0]["l2_cross_wmape_substitute"] == 0.4
    assert "sweep_single_share_highest_plus10" in l3.columns
    assert l3.iloc[0]["sweep_single_share_highest_plus10"] == 0.72
    assert l3.iloc[0]["sweep_brand_leading_minus10"] == 0.41
    # column order follows the protocol table
    cols = list(l3.columns)
    assert cols.index("sweep_single_share_highest_plus10") < cols.index("sweep_brand_leading_minus10")
