from __future__ import annotations

import itertools
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import stats

from causaldemand._code import panel_scorer
from causaldemand.names import TABLE_CONFIGURATIONS
from causaldemand.realdata import B, RNG_SEED, _draw_matrix, _quiet_numpy
from causaldemand.sources import SourceError
from causaldemand.tables import (ALL_MODELS, CPU_MODELS, DEFAULT_SET, DEFAULT_SET_NOTE, MODEL_GROUPS, MODEL_LABELS,
                                 SAMPLED_MODELS, SAMPLER_NOTE, SAMPLER_RUNS, SEEDS, CF, FC, Table, TableError,
                                 average_samplers, base_model, interval, is_sampler_run, scenario_scores,
                                 scenario_tables, select_runs)

KEY = ["product_id", "store_id", "week"]
METRICS = ("accuracy", "abs_bias", "share_overstated", "forecast")
TYPE_LABELS = {"discretechoice_confounding_on": "Probit confounded", "discretechoice_confounding_off": "Probit clean",
               "loglinear_confounding_on": "Log-log confounded", "loglinear_confounding_off": "Log-log clean"}
CONFIG_NAMES = {"discretechoice_confounding_on": "Discrete-choice, on",
                "discretechoice_confounding_off": "Discrete-choice, off",
                "loglinear_confounding_on": "Log-linear, on", "loglinear_confounding_off": "Log-linear, off"}
MODEL_SETS = (("8 models", ALL_MODELS), ("5 CPU models", CPU_MODELS))
SCENARIO_SUBSETS = ("all 32", "product", "brand", "+10%", "-10%")
STABILITY = "results/main/stability"
TOLERANCE, ABS_TOLERANCE, STORE_SUMS_TOLERANCE = 1e-12, 1e-15, 1e-12


def _frame(x, loader) -> pd.DataFrame:
    return x if isinstance(x, pd.DataFrame) else loader(x)


def store_sums(counterfactual_truth, forecast_truth, forecast, scenarios) -> dict:
    P = panel_scorer()
    key = _frame(counterfactual_truth, P.load_answer_key)
    fkey = _frame(forecast_truth, P.load_answer_key)
    l1 = _frame(forecast, P.load_layer1).drop_duplicates(KEY)
    dl = _frame(scenarios, P.load_deltas)
    stores = np.array(sorted(fkey.store_id.unique()))
    sidx = {s: i for i, s in enumerate(stores)}
    scen = np.array(sorted(key.intervention_id.unique()))
    meta = key.drop_duplicates("intervention_id").set_index("intervention_id").loc[scen, ["scope", "direction"]]
    keyrows = key[["intervention_id"] + KEY]
    dl = dl[["intervention_id"] + KEY + ["predicted_delta_units"]].merge(
        keyrows, on=["intervention_id"] + KEY, how="inner")
    m = key.merge(l1[KEY + ["predicted_units"]], on=KEY, how="left")
    m = m.merge(dl, on=["intervention_id"] + KEY, how="left", validate="one_to_one")
    ok = m.predicted_units.notna() & m.predicted_delta_units.notna() & (m.predicted_units > P.QHAT_FLOOR)
    m = m[ok]
    q, qcf = m.q.to_numpy(float), m.q_cf.to_numpy(float)
    qh, dqh = m.predicted_units.to_numpy(float), m.predicted_delta_units.to_numpy(float)
    qt = q * (qh + dqh) / qh
    si = m.store_id.map(sidx).to_numpy()
    ci = pd.Index(scen).get_indexer(m.intervention_id)
    terms = {"A": np.abs(qt - qcf), "Q": qcf, "Bn": qt - qcf, "D": np.abs(qcf - q), "S": qcf - q}
    sums = {}
    for k, v in terms.items():
        a = np.zeros((len(scen), len(stores)))
        np.add.at(a, (ci, si), v)
        sums[k] = a
    f = fkey.merge(l1[KEY + ["predicted_units"]], on=KEY, how="left")
    f = f[f.predicted_units.notna()]
    fs = f.store_id.map(sidx).to_numpy()
    F = np.bincount(fs, weights=np.abs(f.predicted_units.to_numpy(float) - f.q.to_numpy(float)), minlength=len(stores))
    Fq = np.bincount(fs, weights=f.q.to_numpy(float), minlength=len(stores))
    return {"scen": scen, "scope": meta.scope.to_numpy(), "direction": meta.direction.to_numpy(), "stores": stores,
            "F": F, "Fq": Fq, **sums}


def check_store_sums(sums: dict, per_scenario: pd.DataFrame, forecast_wmape: float) -> float:
    ps = per_scenario.set_index("intervention_id").loc[sums["scen"]]
    acc = sums["A"].sum(1) / sums["Q"].sum(1)
    bias = sums["Bn"].sum(1) / sums["D"].sum(1)
    diffs = (np.abs(acc - ps.accuracy_cf_wmape.to_numpy(float)).max(), np.abs(bias - ps.bias.to_numpy(float)).max(),
             abs(sums["F"].sum() / sums["Fq"].sum() - forecast_wmape))
    return max(float(d) if d == d else float("inf") for d in diffs)


def draw_metrics(z, W):
    with _quiet_numpy():
        acc = (W @ z["A"].T) / (W @ z["Q"].T)
        bias = (W @ z["Bn"].T) / (W @ z["D"].T)
        sign = np.sign(W @ z["S"].T)
        forecast = (W @ z["F"]) / (W @ z["Fq"])
    return {"accuracy": np.nanmean(acc, 1), "abs_bias": np.nanmean(np.abs(bias), 1),
            "share_overstated": np.nanmean(bias * sign > 0, 1), "forecast": forecast}


def kendall_w(R: np.ndarray) -> float:
    R = R[~np.isnan(R).any(1)]
    m, n = R.shape
    ranks = np.vstack([stats.rankdata(r) for r in R])
    S = ((ranks.sum(0) - ranks.sum(0).mean()) ** 2).sum()
    T = 0.0
    for r in R:
        c = np.unique(r, return_counts=True)[1]
        T += float((c ** 3 - c).sum())
    denom = m ** 2 * (n ** 3 - n) - m * T
    return float(12 * S / denom) if denom > 0 else float("nan")


def _order(key):
    dataset, setting, seed, model = key
    return (TABLE_CONFIGURATIONS.index(setting), int(seed),
            ALL_MODELS.index(model) if model in ALL_MODELS else len(ALL_MODELS), model, dataset)


def store_bootstrap_intervals(runs: Iterable, b: int = B, rng_seed: int = RNG_SEED) -> pd.DataFrame:
    W = ones = stores = None
    parts: dict = {}
    for (dataset, setting, seed, model), sums in runs:
        if W is None:
            stores = sums["stores"]
            W, ones = _draw_matrix(len(stores), b, rng_seed), np.ones((1, len(stores)))
        elif len(sums["stores"]) != len(stores) or not (sums["stores"] == stores).all():
            raise TableError(f"{dataset} {model}: the store order differs from the first run's")
        point, boot = draw_metrics(sums, ones), draw_metrics(sums, W)
        parts.setdefault((dataset, setting, int(seed), base_model(model)), []).append(
            (str(model), {k: point[k] for k in METRICS}, {k: boot[k] for k in METRICS}))
    rows = []
    for key in sorted(parts, key=_order):
        dataset, setting, seed, base = key
        runs_of = parts[key]
        names = [r[0] for r in runs_of]
        if base in SAMPLED_MODELS:
            complete = len(names) == len(set(names)) == SAMPLER_RUNS and all(map(is_sampler_run, names))
        else:
            complete = len(names) == 1
        if not complete:
            raise TableError(f"{dataset} {base}: runs {', '.join(names)}")
        for k in METRICS:
            p = np.mean([r[1][k] for r in runs_of], axis=0)
            d = np.mean([r[2][k] for r in runs_of], axis=0)
            rows.append({"dataset": dataset, "type": TYPE_LABELS[setting], "seed": seed, "model": base, "metric": k,
                         "point": float(p[0]), "lo": float(np.nanpercentile(d, 2.5)),
                         "hi": float(np.nanpercentile(d, 97.5))})
    return pd.DataFrame(rows, columns=["dataset", "type", "seed", "model", "metric", "point", "lo", "hi"])


def rank_agreement(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = scores.sort_values("intervention_id")
    groups = {k: g for k, g in d.groupby(["setting", "seed", "model"], sort=False)}

    def per_scenario(setting, seed, model, col):
        g = groups.get((setting, seed, model))
        if g is None or len(g) != 32:
            raise TableError(f"{setting} seed {seed} {model}: {0 if g is None else len(g)} scenarios, expected 32")
        return g[col].to_numpy(float)

    across_seeds = []
    for setting in TABLE_CONFIGURATIONS:
        for col, label in (("accuracy_cf_wmape", "accuracy"), ("abs_bias", "mean |bias|")):
            for set_name, ms in MODEL_SETS:
                X = np.array([[np.nanmean(per_scenario(setting, s, m, col)) for m in ms] for s in SEEDS])
                taus = [stats.kendalltau(X[i], X[j]).statistic for i, j in itertools.combinations(range(len(SEEDS)), 2)]
                across_seeds.append({"type": TYPE_LABELS[setting], "metric": label, "models": set_name,
                                     "tau_mean": float(np.mean(taus)), "tau_min": float(np.min(taus)),
                                     "W": kendall_w(X)})
    across_scenarios = []
    for setting in TABLE_CONFIGURATIONS:
        for s in SEEDS:
            g = groups.get((setting, s, ALL_MODELS[0]))
            if g is None:
                raise TableError(f"{setting} seed {s}: no scores of {ALL_MODELS[0]}")
            scope, direc = g["scope"].to_numpy(), g["direction"].to_numpy()
            subsets = {"all 32": np.ones(len(scope), bool), "product": scope == "product", "brand": scope == "brand",
                       "+10%": direc == "plus10", "-10%": direc == "minus10"}
            for col, label in (("accuracy_cf_wmape", "accuracy"), ("abs_bias", "|bias|")):
                for set_name, ms in MODEL_SETS:
                    X = np.column_stack([per_scenario(setting, s, m, col) for m in ms])
                    for sub, mask in subsets.items():
                        across_scenarios.append({"type": TYPE_LABELS[setting], "seed": s, "metric": label,
                                                 "models": set_name, "scenarios": sub, "W": kendall_w(X[mask])})
    return pd.DataFrame(across_seeds), pd.DataFrame(across_scenarios)


def _setting_of(frame: pd.DataFrame) -> pd.Series:
    if "setting" in frame:
        return frame["setting"]
    inverse = {v: k for k, v in TYPE_LABELS.items()}
    return frame["type"].map(inverse)


def check_s1_points(intervals: pd.DataFrame, per_run) -> None:
    runs = average_samplers(select_runs(per_run, DEFAULT_SET, category="tissue"))
    b = intervals.assign(setting=_setting_of(intervals))
    for metric, col in (("accuracy", CF), ("forecast", FC)):
        bm = b[b["metric"] == metric]
        m = bm.merge(runs, on=["setting", "seed", "model"])
        if len(bm) != len(runs) or len(m) != len(runs) or not np.allclose(m["point"], m[col], rtol=0, atol=1e-9):
            raise TableError(f"store bootstrap: the point values of {metric} differ from the per-run scores")


def s1_tables(intervals: pd.DataFrame, per_run=None) -> list[Table]:
    if per_run is not None:
        check_s1_points(intervals, per_run)
    b = intervals.assign(setting=_setting_of(intervals))
    out = []
    for metric, what in (("accuracy", "Counterfactual WMAPE"), ("forecast", "Forecast WMAPE")):
        bm = b[b["metric"] == metric]
        body = []
        for i, setting in enumerate(TABLE_CONFIGURATIONS):
            if i:
                body.append([])
            body.append([f"({'abcd'[i]}) {CONFIG_NAMES[setting]}"])
            for j, (_, models) in enumerate(MODEL_GROUPS):
                if j:
                    body.append([])
                for m in models:
                    cells = []
                    for sd in SEEDS:
                        r = bm[(bm["setting"] == setting) & (bm["model"] == m) & (bm["seed"] == sd)]
                        if len(r) != 1:
                            raise TableError(f"store bootstrap: {setting} {m} seed {sd}: {len(r)} rows")
                        r = r.iloc[0]
                        cells.append(f"{r.point:.3f} [{r.lo:.3f}, {r.hi:.3f}]")
                    body.append([MODEL_LABELS[m]] + cells)
        notes = [f"Score of each data seed [95% interval from {B:,} bootstrap draws of the stores, random number "
                 f"generator seed {RNG_SEED}].", DEFAULT_SET_NOTE, SAMPLER_NOTE]
        out.append(Table(f"{what} by seed, facial tissue datasets", ["Model"] + [f"Seed {s}" for s in SEEDS], body,
                         notes))
    return out


def s2_table(across_seeds: pd.DataFrame, across_scenarios: pd.DataFrame) -> Table:
    rs = across_seeds.assign(setting=_setting_of(across_seeds))
    rs = rs[(rs["metric"] == "accuracy") & (rs["models"] == "8 models")].set_index("setting")
    sc = across_scenarios.assign(setting=_setting_of(across_scenarios))
    sc = sc[(sc["metric"] == "accuracy") & (sc["models"] == "8 models")]
    labels = ("All 32", "Product", "Brand", "+10%", "-10%")
    body = []
    for setting in TABLE_CONFIGURATIONS:
        r = rs.loc[setting]
        cells = [f"{r.tau_mean:.2f}", f"{r.tau_min:.2f}", f"{r.W:.2f}"]
        for key in SCENARIO_SUBSETS:
            w = sc[(sc["setting"] == setting) & (sc["scenarios"] == key)].set_index("seed")["W"]
            if sorted(w.index) != list(SEEDS):
                raise TableError(f"rank agreement: {setting} {key}: seeds {sorted(w.index)}")
            m, h = interval(w.loc[list(SEEDS)].to_numpy(float))
            cells.append(f"{m:.2f} ± {h:.2f}")
        body.append([CONFIG_NAMES[setting]] + cells)
    columns = [[""] + ["Across seeds"] * 3 + ["Across scenarios: Kendall's W"] * len(labels),
               ["Configuration", "τ mean", "τ min", "W"] + list(labels)]
    notes = ["The eight models of the main results, ranked by counterfactual WMAPE. Across seeds: Kendall's τ-b "
             "between the rankings of every pair of the five seeds (mean and minimum) and Kendall's W over the "
             "seeds. Across scenarios: Kendall's W over the 32 scenarios of one seed, mean ± half-width of the 95% "
             "interval over the seeds; also over the product scenarios, the brand scenarios, the price rises (+10%) "
             "and the price cuts (-10%)."]
    return Table("Agreement of the model rankings across seeds and scenarios, facial tissue datasets", columns, body,
                 notes)


class StoreBootstrap:
    def __init__(self, runs: Mapping[str, tuple]):
        self.runs = {k: tuple(v) for k, v in runs.items()}
        self.pending: dict = {}
        self.frames: list[pd.DataFrame] = []
        self.errors: list[str] = []
        self.max_difference = 0.0

    def __call__(self, ctx: dict) -> None:
        if ctx["model"] not in self.runs.get(ctx["run_set"], ()):
            return
        ds = ctx["dataset"]
        if self.pending and ds not in self.pending:
            self.flush()
        try:
            sums = store_sums(ctx["key"], ctx["forecast_truth"], ctx["forecast"], ctx["scenarios"])
            diff = check_store_sums(sums, ctx["per_scenario"], ctx["summary"]["forecast_wmape"])
        except (KeyError, ValueError, TypeError) as exc:
            self.errors.append(f"{ctx['run']}: the store sums cannot be computed: {type(exc).__name__}: {exc}")
            return
        self.max_difference = max(self.max_difference, diff)
        self.pending.setdefault(ds, []).append(((ds.generator_name, ds.setting, ds.seed, ctx["model"]), sums))

    def flush(self) -> None:
        for ds, items in self.pending.items():
            try:
                self.frames.append(store_bootstrap_intervals(items))
            except (TableError, ValueError) as exc:
                self.errors.append(f"{ds.id}: the store-bootstrap intervals cannot be computed: {exc}")
        self.pending = {}

    def intervals(self) -> pd.DataFrame:
        self.flush()
        if not self.frames:
            return pd.DataFrame(columns=["dataset", "type", "seed", "model", "metric", "point", "lo", "hi"])
        return pd.concat(self.frames, ignore_index=True)


def compare(new: pd.DataFrame, old: pd.DataFrame, keys: list[str], values: list[str]) -> str | None:
    a, b = new[keys + values].copy(), old[keys + values].copy()
    for k in keys:
        a[k], b[k] = a[k].astype(str), b[k].astype(str)
    if a.duplicated(keys).any() or b.duplicated(keys).any():
        return "repeated rows"
    m = a.merge(b, on=keys, how="outer", suffixes=("", "_archived"), indicator=True)
    if (m["_merge"] != "both").any():
        return f"{int((m['_merge'] != 'both').sum())} rows without a counterpart"
    for v in values:
        x, y = m[v].astype(float).to_numpy(), m[v + "_archived"].astype(float).to_numpy()
        same = (np.isnan(x) & np.isnan(y)) | np.isclose(x, y, rtol=TOLERANCE, atol=ABS_TOLERANCE)
        if not same.all():
            i = int(np.argmin(same))
            return f"{', '.join(str(m.iloc[i][k]) for k in keys)}: {v} {x[i]!r} rebuilt, {y[i]!r} archived"
    return None


def appendix_tables(result: dict, boot: StoreBootstrap, source, sums) -> tuple[dict, list[str]]:
    from causaldemand.rescore import _listed, run_set_frames

    per_run = run_set_frames(result, "main")
    per_scenario = {k: v for k, v in result["per_scenario_by_run_set"].items() if k.startswith("main/")}
    tables, problems = {}, []
    archived = {}
    files = [f"{STABILITY}/{f}" for f in ("bootstrap_intervals.csv", "rank_across_seeds.csv",
                                          "rank_across_scenarios.csv")]
    try:
        root = source.fetch(files)
        for f in files:
            bad = _listed(sums, f, root / f)
            if bad:
                problems.append(bad)
            archived[f.rsplit("/", 1)[1]] = pd.read_csv(root / f)
    except (SourceError, OSError) as exc:
        problems.append(f"{STABILITY}: the archived tables cannot be read: {exc}")

    def check(name: str, new: pd.DataFrame, keys: list[str], values: list[str]) -> None:
        if name in archived:
            why = compare(new, archived[name], keys, values)
            if why:
                problems.append(f"{STABILITY}/{name}: differs from the rebuilt table ({why})")

    try:
        scores = scenario_scores(per_scenario, DEFAULT_SET, per_run)
        seeds, scen = rank_agreement(scores)
        check("rank_across_seeds.csv", seeds, ["type", "metric", "models"], ["tau_mean", "tau_min", "W"])
        check("rank_across_scenarios.csv", scen, ["type", "seed", "metric", "models", "scenarios"], ["W"])
        tables["rank"] = [s2_table(seeds, scen)]
    except (TableError, ValueError, KeyError) as exc:
        problems.append(f"rank agreement: {exc}")
    intervals = boot.intervals()
    problems += boot.errors
    if boot.max_difference > STORE_SUMS_TOLERANCE:
        problems.append(f"store sums: a score and its sum over the stores differ by {boot.max_difference:.2g}")
    try:
        check("bootstrap_intervals.csv", intervals, ["dataset", "type", "seed", "model", "metric"],
              ["point", "lo", "hi"])
        tables["seeds"] = s1_tables(intervals, per_run)
    except (TableError, ValueError, KeyError) as exc:
        problems.append(f"store bootstrap: {exc}")
    try:
        scen_tables = scenario_tables(per_scenario, per_run)
        tables["scenario_error"] = [t for k, t in scen_tables.items() if k.startswith("scen-cf-")]
        tables["scenario_bias"] = [t for k, t in scen_tables.items() if k.startswith("scen-bias-")]
    except (TableError, ValueError, KeyError) as exc:
        problems.append(f"scenario tables: {exc}")
    return tables, problems
