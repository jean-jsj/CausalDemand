"""Unit tests for panel_scorer.py: a hand-computed example, row accounting, and the validity status."""
import json
import math

import numpy as np
import pandas as pd
import pytest

import panel_scorer as P


def _example():
    key = pd.DataFrame({
        "intervention_id": ["A", "A", "B", "B", "B"],
        "scope": ["product", "product", "brand", "brand", "brand"],
        "direction": ["plus10"] * 5,
        "product_id": ["P1", "P1", "P1", "P2", "P3"],
        "store_id": ["S1", "S2", "S3", "S3", "S3"],
        "week": [1, 1, 2, 2, 2],
        "q": [100.0, 50.0, 10.0, 20.0, 30.0],
        "q_cf": [90.0, 45.0, 11.0, 22.0, 30.0],
    })
    l1 = pd.DataFrame({"product_id": ["P1", "P1", "P1", "P2", "P3"], "store_id": ["S1", "S2", "S3", "S3", "S3"],
                       "week": [1, 1, 2, 2, 2], "predicted_units": [80.0, 60.0, 0.0, 25.0, 30.0]})
    dl = key[["intervention_id", "product_id", "store_id", "week"]].copy()
    dl["predicted_delta_units"] = [-16.0, -3.0, 1.0, 5.0, 0.0]
    return key, l1, dl


def test_hand_computed_example():
    key, l1, dl = _example()
    ps = P.score_scenarios(key, l1, dl).set_index("intervention_id")
    # A: q_tilde_cf = [100*0.8, 50*0.95] = [80, 47.5]
    assert ps.loc["A", "accuracy_cf_wmape"] == pytest.approx(12.5 / 135)
    assert ps.loc["A", "bias"] == pytest.approx(-7.5 / 15)
    # B: first row dropped (q_hat = 0); q_tilde_cf = [20*1.2, 30*1.0] = [24, 30]
    assert ps.loc["B", "n_dropped_qhat_nonpositive"] == 1 and ps.loc["B", "n_rows_scored"] == 2
    assert ps.loc["B", "accuracy_cf_wmape"] == pytest.approx(2 / 52)
    assert ps.loc["B", "bias"] == pytest.approx(2 / 2)
    sm = P.summarize(ps.reset_index())
    assert sm["mean_accuracy_cf_wmape"] == pytest.approx((12.5 / 135 + 2 / 52) / 2)
    assert sm["mean_abs_bias"] == pytest.approx(0.75)
    # A: true change -15, bias -0.5 -> over-reaction; B: true change +2, bias +1 -> over-reaction
    assert sm["share_overstated"] == pytest.approx(1.0)


def test_missing_prediction_is_counted_not_scored():
    key, l1, dl = _example()
    dl = dl.iloc[1:]                                     # drop the delta for A's first row
    ps = P.score_scenarios(key, l1, dl).set_index("intervention_id")
    assert ps.loc["A", "n_missing_prediction"] == 1 and ps.loc["A", "n_rows_scored"] == 1


def test_seed_interval_and_paired():
    r = P.seed_interval([1, 2, 3, 4, 5])
    assert r["mean"] == pytest.approx(3.0)
    assert r["half_width"] == pytest.approx(2.776445 * math.sqrt(2.5) / math.sqrt(5), rel=1e-5)
    pr = P.paired([2, 3, 4], [1, 1, 5])
    assert pr["n_positive"] == 2 and pr["mean"] == pytest.approx(2 / 3)


def test_forecast_wmape_whole_test_set():
    fkey = pd.DataFrame({"product_id": ["P1", "P1", "P2"], "store_id": ["S1", "S2", "S1"], "week": [1, 1, 1], "q": [100.0, 50.0, 10.0]})
    l1 = pd.DataFrame({"product_id": ["P1", "P1"], "store_id": ["S1", "S2"], "week": [1, 1], "predicted_units": [80.0, 60.0]})
    r = P.forecast_wmape(fkey, l1)
    assert r["forecast_wmape"] == pytest.approx(30 / 150)          # P2 row has no prediction: counted, not scored
    assert r["forecast_rows_missing_prediction"] == 1 and r["forecast_rows_scored"] == 2


# ---------------------------------------------------------------------------
# Validity status
# ---------------------------------------------------------------------------


def _full_panel(n_scenarios=P.N_SCENARIOS):
    """One row per scenario, each in its own store; the predictions are exact."""
    key = pd.DataFrame({
        "intervention_id": [f"I{i:02d}" for i in range(n_scenarios)],
        "scope": "product", "direction": "plus10", "product_id": "P1",
        "store_id": [f"S{i}" for i in range(n_scenarios)], "week": 1, "q": 10.0, "q_cf": 9.0,
    })
    fkey = key[P.KEY].assign(q=10.0)
    l1 = key[P.KEY].assign(predicted_units=10.0)
    dl = key[["intervention_id"] + P.KEY].assign(predicted_delta_units=-1.0)
    return key, fkey, l1, dl


def _summary(key, fkey, l1, dl):
    return P.run_summary(P.score_scenarios(key, l1, dl), fkey, l1)


def test_complete_run_is_valid():
    sm = _summary(*_full_panel())
    assert sm["status"] == "valid" and sm["invalid_reasons"] == ""
    assert sm["scorer_version"] == P.SCORER_VERSION
    assert sm["n_scenarios"] == P.N_SCENARIOS and sm["complete_panel"] is True
    assert sm["mean_accuracy_cf_wmape"] == pytest.approx(0.0) and sm["forecast_wmape"] == pytest.approx(0.0)


def test_status_fields_follow_the_scores():
    key, l1, dl = _example()
    ps = P.score_scenarios(key, l1, dl)
    fkey = key[P.KEY].drop_duplicates().assign(q=1.0)
    sm = P.run_summary(ps, fkey, l1, {"dataset": "d", "model": "m", "seed": 1})
    expected = {"dataset": "d", "model": "m", "seed": 1, **P.summarize(ps), **P.forecast_wmape(fkey, l1)}
    assert list(sm)[:len(expected)] == list(expected)
    assert {k: sm[k] for k in expected} == expected
    assert list(sm)[len(expected):] == ["scorer_version", "status", "invalid_reasons"]


@pytest.mark.parametrize("damage, reason", [
    ("missing_delta", "counterfactual ground-truth rows have no prediction"),
    ("zero_forecast", "excluded because predicted_units <= 1e-09"),
    ("missing_forecast_row", "forecast ground-truth rows have no prediction"),
    ("repeated_forecast_key", "repeat a (product_id, store_id, week) key"),
    ("fewer_scenarios", "31 of 32 scenarios are scored"),
    ("infinite_delta", "a summary score is not finite"),
])
def test_invalid_run_names_the_failed_condition(damage, reason):
    key, fkey, l1, dl = _full_panel()
    if damage == "missing_delta":
        dl = dl.iloc[1:]
    elif damage == "zero_forecast":
        l1.loc[0, "predicted_units"] = 0.0
    elif damage == "missing_forecast_row":
        fkey = pd.concat([fkey, pd.DataFrame({"product_id": ["P9"], "store_id": ["S0"], "week": [1], "q": [5.0]})])
    elif damage == "repeated_forecast_key":
        l1 = pd.concat([l1, l1.iloc[[3]]], ignore_index=True)
    elif damage == "fewer_scenarios":
        key, fkey, l1, dl = _full_panel(P.N_SCENARIOS - 1)
    elif damage == "infinite_delta":
        dl.loc[0, "predicted_delta_units"] = np.inf
    sm = _summary(key, fkey, l1, dl)
    assert sm["status"] == "invalid"
    assert reason in sm["invalid_reasons"]


def test_repeated_scenario_prediction_stops_scoring():
    key, fkey, l1, dl = _full_panel()
    with pytest.raises(pd.errors.MergeError):
        P.score_scenarios(key, l1, pd.concat([dl, dl.iloc[[0]]], ignore_index=True))


def test_per_scenario_columns():
    key, fkey, l1, dl = _full_panel()
    assert list(P.score_scenarios(key, l1, dl).columns) == [
        "intervention_id", "scope", "direction", "n_rows_key", "n_missing_prediction",
        "n_dropped_qhat_nonpositive", "n_rows_scored", "accuracy_cf_wmape", "bias", "true_change_sum"]


def _write_run(tmp_path, key, fkey, l1, dl):
    paths = {name: tmp_path / f"{name}.csv" for name in ("key", "fkey", "layer1", "deltas")}
    for name, frame in zip(paths, (key, fkey, l1, dl)):
        frame.to_csv(paths[name], index=False)
    return ["score"] + [arg for name, path in paths.items() for arg in (f"--{name}", str(path))]


def test_command_line_exit_status(tmp_path, capsys):
    key, fkey, l1, dl = _full_panel()
    args = _write_run(tmp_path, key, fkey, l1, dl)
    P.main(args + ["--out", str(tmp_path / "per_scenario.csv")])
    assert json.loads(capsys.readouterr().out)["status"] == "valid"
    assert len(pd.read_csv(tmp_path / "per_scenario.csv")) == P.N_SCENARIOS

    args = _write_run(tmp_path, key, fkey, l1, dl.iloc[1:])
    with pytest.raises(SystemExit) as exc:
        P.main(args)
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "invalid"
    assert "invalid score" in captured.err
