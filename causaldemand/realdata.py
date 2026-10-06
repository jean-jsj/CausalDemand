from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from causaldemand._code import file_sha256, output_folder, panel_scorer

P = panel_scorer()
KEY = list(P.KEY)
SKEY = ["intervention_id"] + KEY
B, RNG_SEED = 1000, 20260930
CHUNK = 2_000_000
ARMS = {"cpu_default": ["lgbm", "xgb", "rf", "dml", "hier"], "cpu_tuned": ["lgbm", "xgb", "rf", "dml"]}
FM_ARM = "foundation_models"
FM_MODELS = ["tabpfn", "tabfm", "chronos2"]
SAMPLED = ("tabpfn", "tabfm")
SAMPLER_SEEDS = (0, 1, 2, 3, 4)
ARM_LABEL = {"cpu_default": "default", "cpu_tuned": "forecast-tuned", FM_ARM: "default"}
NAMES = {"lgbm": "LightGBM", "xgb": "XGBoost", "rf": "Random forest", "dml": "Double ML (IV)",
         "hier": "Hierarchical (IV)", "tabpfn": "TabPFN", "tabfm": "TabFM", "chronos2": "Chronos-2"}
HEADLINE = [("cpu_tuned", "lgbm"), ("cpu_tuned", "xgb"), ("cpu_tuned", "rf"), ("cpu_tuned", "dml"),
            ("cpu_default", "hier"), (FM_ARM, "tabpfn"), (FM_ARM, "tabfm"), (FM_ARM, "chronos2")]
REFERENCE = ("cpu_tuned", "lgbm")
EXPECTED_TRIALS = {"lgbm": 20, "xgb": 20, "rf": 10, "dml": 20}
IV_MODELS = ("dml", "hier")
COUNTS = ["N", "NZ", "C", "Z", "M", "QZ", "ZQZ"]

TRAIN_KEYS = ("inputs/transactions_train_public.csv", "public/transactions_train_public.csv", "train")
PANEL_KEYS = ("inputs/counterfactual_sweep_context_panel.csv", "panel context")
FORECAST_TRUTH = ("ground_truth/ground_truth_forecast.csv", "hidden/answer_key_forecast.csv.gz")
INPUTS = "public"
INPUT_FILES = ("transactions_train_public.csv", "transactions_holdout_context_public.csv",
               "counterfactual_sweep_context_panel.csv")
REAL_BOOTSTRAP = {
    "cereal": {"archived_label": "dff_cer_p60", "n_stores": 81,
               "draw_matrix_sha256": "2a23c80d45d9ca778c01130332412ccce668731d96de3c39f8444f84e9a162f4",
               "train_sha256": "bc1246d91a545c2d6b76a48587acfbe9edef0c131a2db507444cf3fd15d92afd",
               "panel_sha256": "8a51520894fd2cee36609258019b3729fc9f270f82bcfc1539c5c2c9b352025d",
               "forecast_truth_csv_sha256": "f6f5128e5192739483da63c678015d179e24b3ef80b032231b6c2c96c6c9b960"},
    "snack_crackers": {"archived_label": "dff_sna_p60", "n_stores": 82,
                       "draw_matrix_sha256": "b1b907ba82dc398e9246c4f13b4a10f5a8fb9154175d8bba0a41d4dd92d237b0",
                       "train_sha256": "bfcf9c29ec31c326e799bcaef9e0f8895b4dee7dadbf91f84a83c8517001e315",
                       "panel_sha256": "c0154314e0336fda590ea3f7f3fa95c4936d71e1468079def2f0d35741ac336a",
                       "forecast_truth_csv_sha256":
                           "3dc7280f601bd81f5641e5107ca2f434068cdf712ac78210e17b5f10b5d21765"},
}
REFERENCE_KEYS = ("B", "seed", "n_stores", "store_order", "draw_matrix_sha256", "dataset_sha256")
TABLE_LABELS = {("cpu_tuned", "lgbm"): "LightGBM (tuned)", ("cpu_tuned", "xgb"): "XGBoost (tuned)",
                ("cpu_tuned", "rf"): "Random forest (tuned)", ("cpu_tuned", "dml"): "Double ML (IV) (tuned)",
                ("cpu_default", "hier"): "Hierarchical linear (IV)", (FM_ARM, "tabpfn"): "TabPFN", (FM_ARM, "tabfm"): "TabFM",
                (FM_ARM, "chronos2"): "Chronos-2"}
FM_MEAN_RUN = "mean over sampler-seed runs"


class RealDataError(ValueError):
    pass


def text_sha256(path: Path) -> str:
    path = Path(path)
    if path.suffix != ".gz":
        return file_sha256(path)
    h = hashlib.sha256()
    with gzip.open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def direction_label(intervention_id: str) -> int:
    if intervention_id.endswith("plus10"):
        return 1
    if intervention_id.endswith("minus10"):
        return -1
    raise RealDataError(f"scenario {intervention_id}: the id ends in neither plus10 nor minus10")


def fm_runs(model: str) -> list:
    return [(f"{model}__sampler{k}", k) for k in SAMPLER_SEEDS] if model in SAMPLED else [(model, None)]


def _draw_matrix(n_stores: int, b: int = B, seed: int = RNG_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = [np.bincount(rng.integers(0, n_stores, n_stores), minlength=n_stores) for _ in range(b)]
    return np.vstack(draws).astype(float)


def _matrix_sha256(W: np.ndarray) -> str:
    return hashlib.sha256(W.astype(np.int64).tobytes()).hexdigest()


@contextmanager
def _quiet_numpy():
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def dataset_paths(dataset_dir: Path) -> tuple[Path, Path]:
    dataset_dir = Path(dataset_dir)
    inputs = dataset_dir / INPUTS
    if not dataset_dir.is_dir():
        raise RealDataError(f"{dataset_dir}: no such dataset folder")
    if not inputs.is_dir():
        raise RealDataError(f"{dataset_dir} holds no folder {INPUTS}/ (the dataset's input files)")
    absent = [n for n in INPUT_FILES if not (inputs / n).is_file()]
    if absent:
        raise RealDataError(f"{inputs} lacks {', '.join(absent)}; download the dataset again with "
                            "causaldemand download <category>")
    for name in FORECAST_TRUTH:
        if (dataset_dir / name).is_file():
            return inputs, dataset_dir / name
    raise RealDataError(f"{dataset_dir} holds no forecast ground truth ({FORECAST_TRUTH[0]}); causaldemand score "
                        "and rescore build it from the Kilts Center files")


def load_dataset(inputs: Path, forecast_truth: Path) -> dict:
    inputs, forecast_truth = Path(inputs), Path(forecast_truth)
    fkey = P.load_answer_key(forecast_truth)
    dup = int(fkey.duplicated(KEY).sum())
    if dup:
        raise RealDataError(f"the forecast ground truth repeats {dup} (product_id, store_id, week) keys")
    stores = np.array(sorted(fkey["store_id"].unique()))
    sidx = pd.Series(np.arange(len(stores)), index=stores)

    order, parts, tiny = [], [], 0
    for ch in pd.read_csv(inputs / "counterfactual_sweep_context_panel.csv",
                          usecols=SKEY + ["baseline_price", "intervention_price"],
                          dtype={"intervention_id": str, "product_id": str, "store_id": str}, chunksize=CHUNK):
        order += [i for i in pd.unique(ch["intervention_id"]) if i not in order]
        b, i = ch["baseline_price"].to_numpy(float), ch["intervention_price"].to_numpy(float)
        tiny += int(((b != i) & np.isclose(b, i)).sum())
        ch = ch[i != b]
        parts.append(ch.assign(week=ch["week"].astype(int)))
    if tiny:
        raise RealDataError(f"{tiny} rows change price by less than numpy.isclose's tolerance; "
                            "models/foundation/fm_runner.py does not count them as moved")
    moved = pd.concat(parts, ignore_index=True)
    scen = pd.Index(order, name="intervention_id")
    moved["scen"] = scen.get_indexer(moved["intervention_id"])
    moved["store"] = moved["store_id"].map(sidx)
    if moved["store"].isna().any():
        raise RealDataError(f"{int(moved['store'].isna().sum())} moved rows are in stores outside the forecast "
                            "ground truth")
    moved["store"] = moved["store"].astype(int)
    dp = moved["intervention_price"].to_numpy(float) - moved["baseline_price"].to_numpy(float)
    moved["expected"] = -np.sign(dp).astype(int)
    label = np.array([direction_label(i) for i in order])
    moved["against_label"] = np.sign(dp).astype(int) != label[moved["scen"].to_numpy()]
    hold = pd.read_csv(inputs / "transactions_holdout_context_public.csv", usecols=KEY,
                       dtype={"product_id": str, "store_id": str})
    last = hold.groupby(["product_id", "store_id"])["week"].max()
    series_last = moved.merge(last.rename("last_week"), left_on=["product_id", "store_id"], right_index=True,
                              how="left").groupby("scen")["last_week"].max().reindex(range(len(order)))
    full = (series_last == series_last.max()).to_numpy()
    sel = {"up": label == 1, "down": label == -1, "both": np.ones(len(order), bool),
           "up_full_window": (label == 1) & full, "down_full_window": (label == -1) & full, "both_full_window": full}
    meta = pd.DataFrame({"intervention_id": order,
                         "rule": [i.replace("sweep_", "").rsplit("_promo_", 1)[0] for i in order],
                         "scope": ["brand" if "_brand" in i else "product" for i in order],
                         "direction": np.where(label == 1, "plus10", "minus10")})
    meta["moved_series_last_test_week"] = series_last.to_numpy()
    sha = {"train": file_sha256(inputs / "transactions_train_public.csv"),
           "panel": file_sha256(inputs / "counterfactual_sweep_context_panel.csv"),
           "forecast_truth_csv": text_sha256(forecast_truth)}
    if forecast_truth.suffix == ".gz":
        sha["forecast_key"] = file_sha256(forecast_truth)
    return {"fkey": fkey, "stores": stores, "sidx": sidx, "moved": moved, "scen": scen, "sel": sel, "meta": meta, "sha": sha}


def _spec_dataset_sha256(sha: dict) -> dict:
    third = "forecast_key" if "forecast_key" in sha else "forecast_truth_csv"
    return {"train": sha["train"], "panel": sha["panel"], third: sha[third]}


def read_moved_deltas(path: Path, moved: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    keys = moved[SKEY]
    parts = []
    for ch in pd.read_csv(path, usecols=SKEY + ["predicted_delta_units"],
                          dtype={"intervention_id": str, "product_id": str, "store_id": str}, chunksize=CHUNK):
        ch["week"] = ch["week"].astype(int)
        parts.append(ch.merge(keys, on=SKEY, how="inner"))
    d = pd.concat(parts, ignore_index=True)
    dup = int(d.duplicated(SKEY).sum())
    d = d.drop_duplicates(SKEY)
    return moved[SKEY].merge(d, on=SKEY, how="left", validate="one_to_one"), dup


def count_matrices(moved: pd.DataFrame, dq: np.ndarray, qhat: np.ndarray, n_scen: int, S: int) -> dict:
    valid = np.isfinite(dq)
    nz = valid & (dq != 0)
    flags = {"N": valid, "NZ": nz, "C": nz & (np.sign(dq) == moved["expected"].to_numpy()),
             "Z": valid & (dq == 0), "M": ~valid, "QZ": qhat == 0, "ZQZ": valid & (dq == 0) & (qhat == 0)}
    flat = moved["scen"].to_numpy() * S + moved["store"].to_numpy()
    return {k: np.bincount(flat, weights=v.astype(float), minlength=n_scen * S).reshape(n_scen, S)
            for k, v in flags.items()}


def metrics(z: dict, Wm: np.ndarray, sel: dict) -> dict:
    C, NZ, N, Z = (Wm @ z[k].T for k in ("C", "NZ", "N", "Z"))
    with _quiet_numpy():
        out = {"forecast_wmape": (Wm @ z["F"]) / (Wm @ z["Fq"])}
        for d, s in sel.items():
            c, nz, n, zz = C[:, s], NZ[:, s], N[:, s], Z[:, s]
            out[f"sign_{d}_excl"] = c.sum(1) / nz.sum(1)
            out[f"sign_{d}_wrong"] = c.sum(1) / n.sum(1)
            out[f"sign_{d}_excl_mean"] = np.nanmean(c / nz, 1)
            out[f"sign_{d}_wrong_mean"] = np.nanmean(c / n, 1)
            out[f"zero_share_{d}"] = zz.sum(1) / n.sum(1)
    return out


def manifest_checks(run: Path, arm: str, model: str, data: dict,
                    sampler_seed: int | None = None) -> tuple[list, list, dict]:
    invalid, issues, rec = [], [], {}
    fm = arm == FM_ARM
    mp = run / "manifest.json"
    if not mp.exists():
        issues.append("no manifest.json")
    else:
        man = json.loads(mp.read_text(encoding="utf-8"))
        env = man.get("env", {})
        if not fm or "env" in man:
            rec["env"] = ";".join(f"{k}={v}" for k, v in sorted(env.items()))
        rec["code_sha256"] = json.dumps(man.get("code_sha256", {}), sort_keys=True)
        sha = man.get("data_sha256", {})
        for label, keys, ref in (("training file", TRAIN_KEYS, "train"), ("scenario file", PANEL_KEYS, "panel")):
            k = next((k for k in keys if k in sha), None)
            if k is None:
                issues.append(f"manifest records no checksum of the {label}")
            elif sha[k] != data["sha"][ref]:
                invalid.append(f"the run used another {label} than the dataset's")
        model_field = "model" if fm else "model_key"
        if man.get(model_field, model) != model:
            invalid.append(f"manifest {model_field} {man.get(model_field)} != folder {model}")
        if fm and model in SAMPLED:
            ctx = man.get("context") or {}
            if "sampler_seed" not in ctx:
                issues.append("manifest records no sampler seed")
            elif ctx["sampler_seed"] != sampler_seed:
                invalid.append(f"manifest sampler seed {ctx['sampler_seed']} != folder {sampler_seed}")
        if "CARD_KEEP_PROMO_COST" in env:
            issues.append(f"CARD_KEEP_PROMO_COST={env['CARD_KEEP_PROMO_COST']} (must be unset)")
        if not fm and model in IV_MODELS and env.get("CARD_PRICE_ENDING_CENT") != "0":
            issues.append(f"CARD_PRICE_ENDING_CENT={env.get('CARD_PRICE_ENDING_CENT', 'unset')} (must be \"0\")")
    if arm == "cpu_tuned":
        tp = run / "tuning.json"
        if not tp.exists():
            issues.append("no tuning.json")
        else:
            tun = json.loads(tp.read_text(encoding="utf-8"))
            rec["n_trials"] = tun.get("n_trials")
            rec["best_trial"] = tun.get("best_trial")
            if tun.get("n_trials") != EXPECTED_TRIALS[model] or len(tun.get("trials", [])) != EXPECTED_TRIALS[model]:
                issues.append(f"{tun.get('n_trials')} trials recorded, {len(tun.get('trials', []))} run "
                              f"(expected {EXPECTED_TRIALS[model]})")
    return invalid, issues, rec


def score_run(run: Path, arm: str, model: str, data: dict, sampler_seed: int | None = None, *,
              forecast: Path | None = None, scenarios: Path | None = None,
              check_manifest: bool = True) -> tuple[dict, dict, pd.DataFrame]:
    fkey, moved, S, n_scen = data["fkey"], data["moved"], len(data["stores"]), len(data["scen"])
    invalid, issues, rec = manifest_checks(run, arm, model, data, sampler_seed) if check_manifest else ([], [], {})

    l1 = P.load_layer1(forecast or run / "layer1_predictions.csv.gz")
    fw = P.forecast_wmape(fkey, l1)
    dup_l1 = int(l1.duplicated(KEY).sum())
    l1u = l1.drop_duplicates(KEY)
    f = fkey.merge(l1u[KEY + ["predicted_units"]], on=KEY, how="left")
    f = f[f["predicted_units"].notna()]
    fs = f["store_id"].map(data["sidx"]).to_numpy()
    F = np.bincount(fs, weights=np.abs(f["predicted_units"].to_numpy(float) - f["q"].to_numpy(float)), minlength=S)
    Fq = np.bincount(fs, weights=f["q"].to_numpy(float), minlength=S)
    check = abs(F.sum() / Fq.sum() - fw["forecast_wmape"])
    if np.isfinite(fw["forecast_wmape"]) and not check <= 1e-12:
        raise RuntimeError(f"{run}: per-store forecast sums differ from the scorer by {check:.3g}")

    d, dup_d = read_moved_deltas(scenarios or run / "panel_deltas.csv.gz", moved)
    dq = d["predicted_delta_units"].to_numpy(float)
    qhat = moved[KEY].merge(l1u[KEY + ["predicted_units"]], on=KEY, how="left")["predicted_units"].to_numpy(float)
    z = count_matrices(moved, dq, qhat, n_scen, S)
    z["F"], z["Fq"] = F, Fq

    if fw["forecast_rows_missing_prediction"]:
        invalid.append(f"{fw['forecast_rows_missing_prediction']} forecast-key rows have no prediction")
    if dup_l1:
        invalid.append(f"{dup_l1} forecast predictions repeat a (product_id, store_id, week) key")
    if not np.isfinite(fw["forecast_wmape"]):
        invalid.append("the forecast WMAPE is not finite")
    n_miss = int(z["M"].sum())
    if n_miss:
        invalid.append(f"{n_miss} moved rows have no finite predicted change")
    if dup_d:
        invalid.append(f"{dup_d} scenario predictions repeat an (intervention_id, product_id, store_id, week) key")

    row = {"arm": arm, "settings": ARM_LABEL.get(arm, ""), "model": model, "model_name": NAMES.get(model, model),
           "status": "invalid" if invalid else "valid", "invalid_reasons": "; ".join(invalid),
           "protocol_issues": "; ".join(issues), "scorer_version": P.SCORER_VERSION,
           "forecast_rows_scored": fw["forecast_rows_scored"],
           "forecast_rows_missing_prediction": fw["forecast_rows_missing_prediction"],
           "forecast_rows_duplicate": dup_l1, "forecast_point_check_abs_diff": check,
           "scenario_prediction_keys_duplicate": dup_d,
           "moved_rows_baseline_prediction_missing": int(np.isnan(qhat).sum()), **rec}
    for dname, s in data["sel"].items():
        tot = {k: int(z[k][s].sum()) for k in COUNTS}
        row.update({f"n_moved_{dname}": tot["N"] + tot["M"], f"n_with_prediction_{dname}": tot["N"],
                    f"n_missing_prediction_{dname}": tot["M"], f"n_nonzero_{dname}": tot["NZ"],
                    f"n_correct_{dname}": tot["C"], f"n_wrong_sign_{dname}": tot["NZ"] - tot["C"],
                    f"n_zero_change_{dname}": tot["Z"], f"n_baseline_pred_zero_{dname}": tot["QZ"],
                    f"n_zero_change_and_baseline_pred_zero_{dname}": tot["ZQZ"],
                    f"n_scenarios_{dname}": int(s.sum()),
                    f"n_scenarios_without_nonzero_change_{dname}": int((z["NZ"][s].sum(1) == 0).sum())})

    ps = data["meta"].copy()
    tot = {k: z[k].sum(1) for k in COUNTS}
    with np.errstate(invalid="ignore", divide="ignore"):
        ps = ps.assign(n_moved=(tot["N"] + tot["M"]).astype(int), n_with_prediction=tot["N"].astype(int),
                       n_missing_prediction=tot["M"].astype(int), n_nonzero=tot["NZ"].astype(int),
                       n_correct=tot["C"].astype(int), n_wrong_sign=(tot["NZ"] - tot["C"]).astype(int),
                       n_zero_change=tot["Z"].astype(int), n_baseline_pred_zero=tot["QZ"].astype(int),
                       n_moved_against_label=np.bincount(data["moved"]["scen"],
                                                         weights=data["moved"]["against_label"].astype(float),
                                                         minlength=n_scen).astype(int),
                       sign_excl=tot["C"] / tot["NZ"], sign_wrong=tot["C"] / tot["N"],
                       zero_share=tot["Z"] / tot["N"])
    ps.insert(0, "model", model)
    ps.insert(0, "arm", arm)
    return row, z, ps


def add_estimates(row: dict, point: dict, boot: dict) -> None:
    for k in point:
        row[k] = float(point[k][0])
        if not k.endswith("_mean"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                row[f"{k}_lo"], row[f"{k}_hi"] = (float(v) for v in np.nanpercentile(boot[k], [2.5, 97.5]))


def mean_over_runs(model: str, scored: list) -> tuple[dict, dict]:
    point = {k: np.mean([p[k] for _, p, _ in scored], axis=0) for k in scored[0][1]}
    boot = {k: np.mean([b[k] for _, _, b in scored], axis=0) for k in scored[0][2]}
    invalid = [f"{r['run']}: {r['invalid_reasons']}" for r, _, _ in scored if r["status"] != "valid"]
    issues = [f"{r['run']}: {r['protocol_issues']}" for r, _, _ in scored if r["protocol_issues"]]
    absent = [n for n, _ in fm_runs(model) if n not in {r["run"] for r, _, _ in scored}]
    if absent:
        invalid.append(f"{len(scored)} of {len(SAMPLER_SEEDS)} sampler-seed runs scored "
                       f"(not scored: {', '.join(absent)})")
    row = {"arm": FM_ARM, "settings": ARM_LABEL[FM_ARM], "model": model, "model_name": NAMES[model],
           "status": "invalid" if invalid else "valid", "invalid_reasons": "; ".join(invalid),
           "protocol_issues": "; ".join(issues), "scorer_version": P.SCORER_VERSION}
    add_estimates(row, point, boot)
    row.update(run_dir="; ".join(r["run_dir"] for r, _, _ in scored), run=FM_MEAN_RUN,
               n_sampler_runs_averaged=len(scored), sampler_runs_averaged="; ".join(r["run"] for r, _, _ in scored))
    return row, boot


def write_rows(path: Path, groups: list) -> None:
    groups = [g for g in groups if g]
    cols = list(dict.fromkeys(k for g in groups for r in g for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        for i, g in enumerate(groups):
            pd.DataFrame(g).reindex(columns=cols).to_csv(fh, index=False, header=i == 0, lineterminator="\n")


def fmt(p) -> str:
    return "n/a" if p is None or not np.isfinite(p) else f"{p:.3f}"


def score_reference_runs(dataset_dir: Path, runs_root: Path, out: Path | None = None, *,
                         dataset_folder: str | None = None, label: str | None = None, quiet: bool = False) -> dict:
    dataset_dir, runs_root = Path(dataset_dir), Path(runs_root)
    say = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
    inputs, forecast_truth = dataset_paths(dataset_dir)
    dataset_folder = dataset_folder or Path(os.path.abspath(dataset_dir)).name
    name = label or dataset_folder

    data = load_dataset(inputs, forecast_truth)
    S = len(data["stores"])
    W = _draw_matrix(S)
    ones = np.ones((1, S))
    say(f"{name}: {len(data['fkey']):,} forecast-key rows, {S} stores, {len(data['scen'])} scenarios, "
        f"{len(data['moved']):,} moved rows")

    def score(arm: str, model: str, run_name: str, sampler_seed: int | None = None):
        run = runs_root / arm / dataset_folder / run_name
        absent = [f for f in ("layer1_predictions.csv.gz", "panel_deltas.csv.gz") if not (run / f).exists()]
        if absent:
            missing.append(f"{arm}/{run_name} ({'no run folder' if not run.is_dir() else 'no ' + ', '.join(absent)})")
            if not run.is_dir() and arm != FM_ARM and (arm, model) not in HEADLINE:
                not_in_table.append(missing[-1])
            return None
        try:
            with _quiet_numpy():
                row, z, ps = score_run(run, arm, model, data, sampler_seed)
        except Exception as e:
            missing.append(f"{arm}/{run_name} (unreadable: {type(e).__name__}: {e})")
            return None
        point, boot = metrics(z, ones, data["sel"]), metrics(z, W, data["sel"])
        add_estimates(row, point, boot)
        row["run_dir"] = run.relative_to(runs_root).as_posix()
        if arm == FM_ARM:
            row["run"] = ps["run"] = run_name
        per_scen.append(ps)
        say(f"  {arm}/{run_name}: {row['status']}; forecast WMAPE {fmt(row['forecast_wmape'])}; sign accuracy, "
            f"price up, zeros left out {fmt(row['sign_up_excl'])}")
        return row, point, boot

    rows, runs, fm_means, per_scen, missing, not_in_table = {}, {}, [], [], [], []
    for arm, models in ARMS.items():
        for m in models:
            res = score(arm, m, m)
            if res is not None:
                rows[(arm, m)] = runs[(arm, m)] = res[0]
    for m in FM_MODELS:
        scored = []
        for run_name, k in fm_runs(m):
            res = score(FM_ARM, m, run_name, k)
            if res is not None:
                scored.append(res)
                runs[(FM_ARM, run_name)] = res[0]
        if not scored:
            continue
        if m in SAMPLED:
            row, _ = mean_over_runs(m, scored)
            fm_means.append(row)
        else:
            row = scored[0][0]
        rows[(FM_ARM, m)] = row
    if not_in_table:
        say("  no run folder (not needed for the real-data table): "
            + ", ".join(x.split(" (", 1)[0] for x in not_in_table))
    for x in missing:
        if x not in not_in_table:
            say(f"  not scored: {x}")
    if not rows:
        raise RealDataError(f"{runs_root}: no run of {dataset_folder} to score")

    groups = [[x for r, x in runs.items() if r[0] != FM_ARM], [x for r, x in runs.items() if r[0] == FM_ARM],
              fm_means]
    per_scenario = pd.concat(per_scen, ignore_index=True)
    spec = {"B": B, "seed": RNG_SEED, "rng": "numpy.random.default_rng (PCG64)",
            "draw": "W[b] = numpy.bincount(rng.integers(0, S, S), minlength=S), b = 0..B-1, drawn in order",
            "interval": "numpy.nanpercentile(draws, [2.5, 97.5]) (linear interpolation)",
            "statistic": "ratio of store-weighted sums (W @ per-store sums), per metric",
            "sampler_seed_mean": "TabPFN, TabFM: in each draw (and for the point estimate), the mean over the "
                                 "model's sampler-seed runs of each run's metric (a mean of ratios)",
            "n_stores": S, "store_order": data["stores"].tolist(),
            "draw_matrix_sha256": _matrix_sha256(W),
            "numpy_version": np.__version__, "pandas_version": pd.__version__, "python": platform.python_version(),
            "scorer_version": P.SCORER_VERSION, "dataset": name,
            "dataset_sha256": _spec_dataset_sha256(data["sha"])}
    if out is not None:
        out = output_folder(Path(out))
        write_rows(out / "per_run.csv", groups)
        per_scenario.to_csv(out / "per_scenario.csv", index=False, lineterminator="\n")
        with open(out / "bootstrap_spec.json", "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(spec, indent=1))
    bad = [f"{r[0]}/{r[1]}" for r, x in runs.items() if x["status"] != "valid"]
    say(f"{name}: {len(runs)} runs scored" + (f"; invalid: {', '.join(bad)}" if bad else "")
        + (f"; written to {out}" if out is not None else ""))
    flat = [x for g in groups for x in g]
    per_run = pd.DataFrame(flat).reindex(columns=list(dict.fromkeys(k for x in flat for k in x)))
    return {"rows": rows, "runs": runs, "fm_means": fm_means, "per_run": per_run, "per_scenario": per_scenario,
            "missing": missing, "spec": spec}


def read_reference(reference) -> dict:
    if isinstance(reference, dict):
        ref = reference
    else:
        try:
            ref = json.loads(Path(reference).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RealDataError(f"{reference}: the reference bootstrap cannot be read ({exc})") from exc
    if not isinstance(ref, dict):
        raise RealDataError("the reference bootstrap is not a JSON object")
    absent = [k for k in REFERENCE_KEYS if k not in ref]
    if not absent:
        if isinstance(ref["dataset_sha256"], dict):
            absent = [f"dataset_sha256.{k}" for k in ("train", "panel") if k not in ref["dataset_sha256"]]
        else:
            absent = ["dataset_sha256 (an object of checksums)"]
    if absent:
        raise RealDataError(f"the reference bootstrap lacks {', '.join(absent)}")
    return ref


def released_dataset(sha: dict) -> str | None:
    for name, v in REAL_BOOTSTRAP.items():
        if sha.get("train") == v["train_sha256"] or sha.get("panel") == v["panel_sha256"] \
                or sha.get("forecast_truth_csv") == v["forecast_truth_csv_sha256"]:
            return name
    return None


def check_released(data: dict, W: np.ndarray) -> str | None:
    name = released_dataset(data["sha"])
    if name is None:
        return None
    v = REAL_BOOTSTRAP[name]
    for k, ref, what in (("train", "train_sha256", "training file"), ("panel", "panel_sha256", "scenario file"),
                         ("forecast_truth_csv", "forecast_truth_csv_sha256", "forecast ground truth")):
        if data["sha"][k] != v[ref]:
            raise RealDataError(f"the {what} differs from that of the released {name} dataset")
    if len(data["stores"]) != v["n_stores"]:
        raise RealDataError(f"the number of stores differs from that of the released {name} dataset")
    if _matrix_sha256(W) != v["draw_matrix_sha256"]:
        raise RealDataError(f"this version of numpy draws other stores than the bootstrap of the released {name} "
                            "scores")
    return name


def check_reference(data: dict, W: np.ndarray, reference) -> None:
    ref = read_reference(reference)
    if (ref["B"], ref["seed"], ref["n_stores"]) != (B, RNG_SEED, len(data["stores"])) \
            or ref["store_order"] != data["stores"].tolist():
        raise RealDataError("the store order or the bootstrap size differs from the reference bootstrap")
    if _matrix_sha256(W) != ref["draw_matrix_sha256"]:
        raise RealDataError("this version of numpy draws other stores than the reference bootstrap")
    rsha = ref["dataset_sha256"]
    for k, what in (("train", "training file"), ("panel", "scenario file")):
        if rsha[k] != data["sha"][k]:
            raise RealDataError(f"the {what} differs from the one the reference runs were scored on")
    expected = rsha.get("forecast_truth_csv")
    if expected is None:
        expected = next((v["forecast_truth_csv_sha256"] for v in REAL_BOOTSTRAP.values()
                         if (v["train_sha256"], v["panel_sha256"]) == (rsha["train"], rsha["panel"])), None)
    if expected is not None:
        same = expected == data["sha"]["forecast_truth_csv"]
    elif "forecast_key" in rsha and "forecast_key" in data["sha"]:
        same = rsha["forecast_key"] == data["sha"]["forecast_key"]
    else:
        raise RealDataError(
            "the forecast ground truth cannot be checked: the reference bootstrap records "
            + ("the checksum of a compressed file, and the dataset folder holds the CSV file"
               if "forecast_key" in rsha else "no checksum of it"))
    if not same:
        raise RealDataError("the forecast ground truth differs from the one the reference runs were scored on")


def submission_scores(dataset_dir: Path, forecast: Path, scenarios: Path, *, reference=None,
                      label: str = "submission") -> tuple[dict, pd.DataFrame]:
    from causaldemand.scoring import ScoringInputError

    for p, what in ((forecast, "forecast predictions"), (scenarios, "scenario predictions")):
        if not Path(p).is_file():
            raise ScoringInputError(f"{p}: no such file ({what})")
    inputs, forecast_truth = dataset_paths(dataset_dir)
    data = load_dataset(inputs, forecast_truth)
    S = len(data["stores"])
    W = _draw_matrix(S)
    if reference is not None:
        check_reference(data, W, reference)
    else:
        check_released(data, W)
    try:
        with _quiet_numpy():
            row, z, ps = score_run(Path(forecast).parent, "submission", label, data,
                                   forecast=Path(forecast), scenarios=Path(scenarios), check_manifest=False)
    except RealDataError:
        raise
    except ValueError as exc:
        text = str(exc)
        if "Usecols" in text or "usecols" in text or "columns expected" in text:
            raise ScoringInputError(
                "a prediction file lacks a column. Forecast predictions need product_id, store_id, week, "
                "predicted_units; scenario predictions need intervention_id, product_id, store_id, week, "
                f"predicted_delta_units ({text})") from exc
        raise ScoringInputError(text) from exc
    add_estimates(row, metrics(z, np.ones((1, S)), data["sel"]), metrics(z, W, data["sel"]))
    return row, ps


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _text(x) -> str:
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else str(x)


def _decimals3(x) -> str:
    return "{:.3f}".format(float(x)) if _finite(x) else "n/a"


def table_cells(row) -> tuple[str, str]:
    wmape = (f"{_decimals3(row['forecast_wmape'])} [{_decimals3(row.get('forecast_wmape_lo'))}, "
             f"{_decimals3(row.get('forecast_wmape_hi'))}]") if _finite(row["forecast_wmape"]) else "n/a"
    sign = (f"{_decimals3(row['sign_up_excl'])} [{_decimals3(row.get('sign_up_excl_lo'))}, "
            f"{_decimals3(row.get('sign_up_excl_hi'))}]") if _finite(row["sign_up_excl"]) else "n/a"
    return wmape, sign


def real_table_rows(per_run: pd.DataFrame) -> list[tuple[str, str, str]]:
    run = per_run["run"].map(_text) if "run" in per_run else pd.Series("", index=per_run.index)
    out = []
    for arm, model in HEADLINE:
        hit = (per_run["arm"] == arm) & (per_run["model"] == model)
        if arm == FM_ARM and model in SAMPLED:
            hit &= run == FM_MEAN_RUN
        elif arm == FM_ARM:
            hit &= run != FM_MEAN_RUN
        else:
            hit &= run == ""
        label = TABLE_LABELS[(arm, model)]
        if not hit.any():
            out.append((label, "missing", "missing"))
            continue
        if int(hit.sum()) > 1:
            raise RealDataError(f"per_run holds {int(hit.sum())} rows of {arm}/{model}")
        r = per_run[hit].iloc[0]
        if model in SAMPLED:
            n = int(float(r["n_sampler_runs_averaged"])) if _finite(r.get("n_sampler_runs_averaged")) else 0
            full = len(SAMPLER_SEEDS)
            label += "" if n == full else f" ({n} of {full} seeds)"
        label += (" (invalid)" if _text(r["status"]) != "valid" else "") \
            + (" (not run as specified)" if _text(r.get("protocol_issues")) else "")
        out.append((label, *table_cells(r)))
    return out
