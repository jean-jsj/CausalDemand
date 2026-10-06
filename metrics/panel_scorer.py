"""Scores for the forecasting task and the counterfactual task of the benchmark.

A run is one method applied to one dataset. It is scored from four files:

  counterfactual ground truth   one row per (scenario, target product, store, week) in which the scenario
                                changes the price, with q = expected units at the actual price and
                                q_cf = expected units at the scenario price (columns intervention_id,
                                product_id, store_id, week, q, q_cf; also scope and direction)
  forecast ground truth         one row per (product, store, week) of the 16 evaluation weeks, for every
                                product-store pair with sales in the training window, with q = expected units
  forecast predictions          columns product_id, store_id, week, predicted_units                  (q_hat)
  scenario predictions          columns intervention_id, product_id, store_id, week,
                                predicted_delta_units: predicted units at the scenario price minus
                                predicted units at the actual price                                  (dq_hat)

Counterfactual error and bias, per scenario over its ground-truth rows R. The predicted proportional change is
applied to the true units, so an error in the level of sales cannot offset an error in the response:
  q_tilde_cf            = q * (q_hat + dq_hat) / q_hat
  counterfactual error  = sum_R |q_tilde_cf - q_cf| / sum_R q_cf
  counterfactual bias   = sum_R (q_tilde_cf - q_cf) / sum_R |q_cf - q|
                          (positive: the method predicts more units after the change than there are)
Rows with q_hat <= 1e-9 are excluded, because the ratio is undefined there. A brand scenario needs no special
handling: its ground truth lists the rows of every product of the brand whose price it changes.

Forecast error, one number per run, over every row of the forecast ground truth:
  forecast error        = sum |q_hat - q| / sum q

Summary of a run: the mean counterfactual error over the scenarios (the headline score), the mean absolute
counterfactual bias, the share of scenarios in which the method overstates the response (bias times the sign of
the true change is positive), and the forecast error. Across seeds, a score is reported as its mean with a 95%
t-interval.

Validity. A run's score is valid only if every ground-truth row has a prediction, no counterfactual row is
excluded, all 32 scenarios are scored, no key repeats in the forecast predictions and every summary score is
finite. Every run summary carries ``scorer_version``, ``status`` ("valid" or "invalid") and ``invalid_reasons``.
The scores of an invalid run are still written, for diagnosis, but are not benchmark results; the command line
then names the run on standard error and exits with status 1. A key that repeats in the scenario predictions
stops scoring with an error.

Command line:
  python panel_scorer.py score --key K --fkey F --layer1 L --deltas D [--out per_scenario.csv]
      K = counterfactual ground truth, F = forecast ground truth, L = forecast predictions,
      D = scenario predictions. Prints the run summary as JSON; --out writes the per-scenario table.
  python panel_scorer.py batch --spec spec.csv --out OUT
      spec.csv has one row per run, with columns dataset, model, seed, key, fkey, layer1, deltas.
      Writes OUT/per_run.csv and OUT/per_scenario.csv.
  python panel_scorer.py collect --runs-root RUNS --keys-root KEYS --out OUT
      Scores every folder RUNS/<dataset>/<model>/ that holds layer1_predictions.csv.gz (forecast predictions)
      and panel_deltas.csv.gz (scenario predictions), against KEYS/<dataset>.csv.gz (counterfactual ground
      truth) and KEYS/forecast/<dataset>.csv.gz (forecast ground truth). Writes scores/per_scenario.csv and
      scores/summary.json into each run folder, and OUT/per_run.csv and OUT/per_scenario.csv across runs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCORER_VERSION = "1.0.0"
KEY = ["product_id", "store_id", "week"]
QHAT_FLOOR = 1e-9
N_SCENARIOS = 32
SUMMARY_SCORES = ["mean_accuracy_cf_wmape", "mean_abs_bias", "share_overstated", "forecast_wmape"]


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["product_id"] = df["product_id"].astype(str)
    df["store_id"] = df["store_id"].astype(str)
    df["week"] = df["week"].astype(int)
    if "intervention_id" in df.columns:
        df["intervention_id"] = df["intervention_id"].astype(str)
    return df


def load_answer_key(path) -> pd.DataFrame:
    """Read a ground-truth file (counterfactual or forecast)."""
    return _norm(pd.read_csv(path))


def load_layer1(path) -> pd.DataFrame:
    """Read a forecast-predictions file."""
    return _norm(pd.read_csv(path, usecols=KEY + ["predicted_units"]))


def load_deltas(path) -> pd.DataFrame:
    """Read a scenario-predictions file."""
    return _norm(pd.read_csv(path, usecols=["intervention_id"] + KEY + ["predicted_delta_units"]))


def score_scenarios(key: pd.DataFrame, layer1: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    """One row per scenario: counterfactual error (``accuracy_cf_wmape``), bias, and row accounting."""
    l1 = layer1.drop_duplicates(KEY)
    m = key.merge(l1[KEY + ["predicted_units"]], on=KEY, how="left")
    m = m.merge(deltas[["intervention_id"] + KEY + ["predicted_delta_units"]],
                on=["intervention_id"] + KEY, how="left", validate="one_to_one")
    out = []
    for iid, g in m.groupby("intervention_id", sort=True):
        n_all = len(g)
        missing = g["predicted_units"].isna() | g["predicted_delta_units"].isna()
        g = g[~missing]
        nonpos = g["predicted_units"] <= QHAT_FLOOR
        g = g[~nonpos]
        q = g["q"].to_numpy(float); qcf = g["q_cf"].to_numpy(float)
        qhat = g["predicted_units"].to_numpy(float); dqhat = g["predicted_delta_units"].to_numpy(float)
        qtcf = q * (qhat + dqhat) / qhat
        dq = qcf - q
        out.append({
            "intervention_id": iid,
            "scope": g["scope"].iloc[0] if len(g) and "scope" in g else "",
            "direction": g["direction"].iloc[0] if len(g) and "direction" in g else "",
            "n_rows_key": n_all, "n_missing_prediction": int(missing.sum()),
            "n_dropped_qhat_nonpositive": int(nonpos.sum()), "n_rows_scored": int(len(g)),
            "accuracy_cf_wmape": float(np.abs(qtcf - qcf).sum() / qcf.sum()) if len(g) else np.nan,
            "bias": float((qtcf - qcf).sum() / np.abs(dq).sum()) if len(g) and np.abs(dq).sum() > 0 else np.nan,
            "true_change_sum": float(dq.sum()) if len(g) else np.nan,
        })
    return pd.DataFrame(out)


def forecast_wmape(fkey: pd.DataFrame, layer1: pd.DataFrame) -> dict:
    """Forecast error over every row of the forecast ground truth, and its row accounting."""
    m = fkey.merge(layer1.drop_duplicates(KEY)[KEY + ["predicted_units"]], on=KEY, how="left")
    miss = m["predicted_units"].isna()
    g = m[~miss]
    q = g["q"].to_numpy(float); qhat = g["predicted_units"].to_numpy(float)
    return {"forecast_wmape": float(np.abs(qhat - q).sum() / q.sum()),
            "forecast_rows_scored": int(len(g)), "forecast_rows_missing_prediction": int(miss.sum())}


def summarize(per_scenario: pd.DataFrame) -> dict:
    """Simple means over the scenarios whose counterfactual error and bias are both defined."""
    ps = per_scenario.dropna(subset=["accuracy_cf_wmape", "bias"])
    return {
        "n_scenarios": int(len(ps)),
        "complete_panel": bool(len(ps) == N_SCENARIOS),
        "mean_accuracy_cf_wmape": float(ps["accuracy_cf_wmape"].mean()),
        "mean_abs_bias": float(ps["bias"].abs().mean()),
        # overstated = predicted response larger than the true one, in the true change's direction
        # (price rise: true change < 0, over-reaction gives bias < 0; price cut: the reverse)
        "share_overstated": float((ps["bias"] * np.sign(ps["true_change_sum"]) > 0).mean()),
        "rows_missing_prediction": int(per_scenario["n_missing_prediction"].sum()),
        "rows_dropped_qhat_nonpositive": int(per_scenario["n_dropped_qhat_nonpositive"].sum()),
    }


def _t975(df: int) -> float:
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df))
    except ImportError:  # small fallback table
        return {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}[df]


def seed_interval(values) -> dict:
    """Mean and 95% t-interval over seeds: half width = t_0.975(n - 1) * sd / sqrt(n)."""
    v = np.asarray(list(values), float); n = len(v)
    mean = float(v.mean())
    if n < 2:
        return {"n": n, "mean": mean, "half_width": float("nan"), "lo": float("nan"), "hi": float("nan")}
    h = _t975(n - 1) * float(v.std(ddof=1)) / math.sqrt(n)
    return {"n": n, "mean": mean, "half_width": h, "lo": mean - h, "hi": mean + h}


def paired(a, b) -> dict:
    """Paired difference a - b over seeds, with the 95% interval and the count of seeds with a > b."""
    d = np.asarray(list(a), float) - np.asarray(list(b), float)
    r = seed_interval(d); r["n_positive"] = int((d > 0).sum()); r["diffs"] = d.tolist()
    return r


def run_status(summary: dict, forecast_rows_duplicate: int = 0) -> dict:
    """Validity of one run's score: ``scorer_version``, ``status`` and ``invalid_reasons``.

    ``summary`` holds the outputs of ``summarize`` and ``forecast_wmape``. The run is valid when every
    counterfactual and forecast ground-truth row has a prediction, no counterfactual row is excluded for
    q_hat <= QHAT_FLOOR, all N_SCENARIOS scenarios are scored, no key repeats in the forecast predictions
    (``forecast_rows_duplicate``) and every summary score is finite. Otherwise ``invalid_reasons`` lists
    each failed condition, separated by "; ".
    """
    reasons = []
    if summary["rows_missing_prediction"]:
        reasons.append(f"{summary['rows_missing_prediction']} counterfactual ground-truth rows have no prediction")
    if summary["rows_dropped_qhat_nonpositive"]:
        reasons.append(f"{summary['rows_dropped_qhat_nonpositive']} counterfactual rows are excluded "
                       f"because predicted_units <= {QHAT_FLOOR:g}")
    if summary["forecast_rows_missing_prediction"]:
        reasons.append(f"{summary['forecast_rows_missing_prediction']} forecast ground-truth rows have no prediction")
    if forecast_rows_duplicate:
        reasons.append(f"{forecast_rows_duplicate} forecast predictions repeat a (product_id, store_id, week) key")
    if summary["n_scenarios"] != N_SCENARIOS:
        reasons.append(f"{summary['n_scenarios']} of {N_SCENARIOS} scenarios are scored")
    if not all(math.isfinite(summary[k]) for k in SUMMARY_SCORES):
        reasons.append("a summary score is not finite")
    return {"scorer_version": SCORER_VERSION, "status": "invalid" if reasons else "valid",
            "invalid_reasons": "; ".join(reasons)}


def run_summary(per_scenario: pd.DataFrame, fkey: pd.DataFrame, layer1: pd.DataFrame, labels: dict | None = None) -> dict:
    """One run's summary: ``labels``, the scenario summary, the forecast error and the validity status."""
    summary = {**(labels or {}), **summarize(per_scenario), **forecast_wmape(fkey, layer1)}
    return {**summary, **run_status(summary, int(layer1.duplicated(KEY).sum()))}


def score_run(key_path, fkey_path, layer1_path, deltas_path):
    """Score one run from its four files; returns the per-scenario table and the run summary."""
    l1 = load_layer1(layer1_path)
    ps = score_scenarios(load_answer_key(key_path), l1, load_deltas(deltas_path))
    return ps, run_summary(ps, _norm(pd.read_csv(fkey_path)), l1)


def score_dir(run_dir, key: pd.DataFrame, fkey: pd.DataFrame, labels: dict) -> tuple[pd.DataFrame, dict]:
    """Score one standard run folder and write its per-scenario and aggregate files."""
    run_dir = Path(run_dir)
    l1 = load_layer1(run_dir / "layer1_predictions.csv.gz")
    ps = score_scenarios(key, l1, load_deltas(run_dir / "panel_deltas.csv.gz"))
    summary = run_summary(ps, fkey, l1, labels)
    (run_dir / "scores").mkdir(exist_ok=True)
    ps.to_csv(run_dir / "scores" / "per_scenario.csv", index=False)
    (run_dir / "scores" / "summary.json").write_text(json.dumps(summary, indent=1))
    for k, v in reversed(list(labels.items())):
        ps.insert(0, k, v)
    return ps, summary


def collect(runs_root, keys_root, out) -> list[dict]:
    """Score every run folder under ``runs_root`` (see the module docstring); returns the run summaries."""
    import re
    runs_root, keys_root, out = Path(runs_root), Path(keys_root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, allps = [], []
    for ds_dir in sorted(p for p in runs_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        key = load_answer_key(keys_root / f"{ds_dir.name}.csv.gz")
        fkey = _norm(pd.read_csv(keys_root / "forecast" / f"{ds_dir.name}.csv.gz"))
        seed = int(re.search(r"seed0*(\d+)", ds_dir.name).group(1))
        for run in sorted(p for p in ds_dir.iterdir() if (p / "panel_deltas.csv.gz").exists()):
            ps, sm = score_dir(run, key, fkey, {"dataset": ds_dir.name, "model": run.name, "seed": seed})
            allps.append(ps); rows.append(sm)
            print(json.dumps(sm), flush=True)
    pd.concat(allps).to_csv(out / "per_scenario.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "per_run.csv", index=False)
    return rows


def _exit_if_invalid(rows: list[dict]) -> None:
    """Name every invalid run on standard error and exit with status 1 if there is one."""
    invalid = [r for r in rows if r["status"] != "valid"]
    for r in invalid:
        name = "/".join(str(r[k]) for k in ("dataset", "model") if k in r) or "run"
        print(f"invalid score: {name}: {r['invalid_reasons']}", file=sys.stderr, flush=True)
    if invalid:
        print(f"{len(invalid)} of {len(rows)} runs have an invalid score; do not report their scores.",
              file=sys.stderr, flush=True)
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Score forecast and scenario predictions against the ground truth.")
    ap.add_argument("--version", action="version", version=SCORER_VERSION)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("score"); s1.add_argument("--key", required=True); s1.add_argument("--fkey", required=True); s1.add_argument("--layer1", required=True)
    s1.add_argument("--deltas", required=True); s1.add_argument("--out")
    s2 = sub.add_parser("batch"); s2.add_argument("--spec", required=True); s2.add_argument("--out", required=True)
    s3 = sub.add_parser("collect"); s3.add_argument("--runs-root", required=True); s3.add_argument("--keys-root", required=True); s3.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        _exit_if_invalid(collect(a.runs_root, a.keys_root, a.out)); return
    if a.cmd == "score":
        ps, sm = score_run(a.key, a.fkey, a.layer1, a.deltas)
        if a.out: ps.to_csv(a.out, index=False)
        print(json.dumps(sm, indent=1))
        _exit_if_invalid([sm])
    else:
        spec = pd.read_csv(a.spec); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        rows, allps = [], []
        keys, fkeys = {}, {}
        for r in spec.itertuples():
            if r.key not in keys: keys[r.key] = load_answer_key(r.key)
            if r.fkey not in fkeys: fkeys[r.fkey] = _norm(pd.read_csv(r.fkey))
            l1 = load_layer1(r.layer1)
            ps = score_scenarios(keys[r.key], l1, load_deltas(r.deltas))
            ps.insert(0, "seed", r.seed); ps.insert(0, "model", r.model); ps.insert(0, "dataset", r.dataset)
            allps.append(ps)
            rows.append(run_summary(ps, fkeys[r.fkey], l1, {"dataset": r.dataset, "model": r.model, "seed": r.seed}))
            print(r.dataset, r.model, r.seed, json.dumps(rows[-1]), flush=True)
        pd.concat(allps).to_csv(out / "per_scenario.csv", index=False)
        pd.DataFrame(rows).to_csv(out / "per_run.csv", index=False)
        _exit_if_invalid(rows)


if __name__ == "__main__":
    main()
