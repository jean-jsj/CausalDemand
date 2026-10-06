import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from card_eval_lib import FeatureSpace, load_cell
import card_eval_models as M
import run_panel_model as RPM

SEED = 20261001
VAL_WEEKS = 16

DEFAULTS = {
    "lgbm": dict(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=20,
                 subsample=1.0, colsample_bytree=1.0, reg_lambda=0.0),
    "xgb": dict(n_estimators=300, learning_rate=0.05, max_depth=8, min_child_weight=1.0,
                subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0),
    "rf": dict(n_estimators=200, max_depth=18, min_samples_leaf=5, max_features=1.0),
    "dml": dict(n_estimators=200, learning_rate=0.05, num_leaves=63, min_child_samples=20),
}


def draw(model: str, rng: np.random.Generator) -> dict:
    lu = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    li = lambda lo, hi: int(round(lu(lo, hi)))
    if model == "lgbm":
        return dict(n_estimators=li(100, 1000), learning_rate=lu(0.01, 0.2), num_leaves=li(15, 255),
                    min_child_samples=li(5, 200), subsample=float(rng.uniform(0.5, 1.0)),
                    colsample_bytree=float(rng.uniform(0.5, 1.0)), reg_lambda=lu(1e-3, 10.0))
    if model == "xgb":
        return dict(n_estimators=li(100, 1000), learning_rate=lu(0.01, 0.2), max_depth=int(rng.integers(3, 13)),
                    min_child_weight=lu(1.0, 50.0), subsample=float(rng.uniform(0.5, 1.0)),
                    colsample_bytree=float(rng.uniform(0.5, 1.0)), reg_lambda=lu(1e-3, 10.0))
    if model == "rf":
        return dict(n_estimators=li(100, 400), max_depth=int(rng.integers(8, 31)),
                    min_samples_leaf=li(1, 50), max_features=float(rng.uniform(0.3, 1.0)))
    if model == "dml":
        return dict(n_estimators=li(100, 600), learning_rate=lu(0.02, 0.2), num_leaves=li(15, 127),
                    min_child_samples=li(5, 100))
    raise ValueError(model)


def make(model: str, p: dict):
    if model == "lgbm":
        from lightgbm import LGBMRegressor

        def build(fs):
            m = M.LightGBMModel(fs)
            m.model = LGBMRegressor(**p, subsample_freq=1 if p["subsample"] < 1 else 0, max_depth=-1,
                                    random_state=42, n_jobs=-1, verbose=-1)
            return m
    elif model == "xgb":
        from xgboost import XGBRegressor

        def build(fs):
            m = M.XGBoostModel(fs)
            m.model = XGBRegressor(**p, random_state=42, n_jobs=-1, verbosity=0, tree_method="hist",
                                   enable_categorical=True)
            return m
    elif model == "rf":
        from sklearn.ensemble import RandomForestRegressor

        def build(fs):
            m = M.RandomForestModel(fs, max_train=None)
            m.model = RandomForestRegressor(**p, n_jobs=-1, random_state=42)
            return m
    elif model == "dml":
        from lightgbm import LGBMRegressor

        class TunedDML(M.DoubleMLDepthIV):
            def _learner(self):
                return LGBMRegressor(**p, n_jobs=-1, random_state=self.random_state, verbose=-1,
                                     force_row_wise=True, deterministic=True)

        def build(fs):
            return TunedDML(fs, max_train=None)
    else:
        raise ValueError(model)
    return build


def main() -> None:
    cell_dir, panel_dir, out_root, model = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    n_trials, trials_path = int(sys.argv[5]), Path(sys.argv[6])
    cell = load_cell(cell_dir, load_sweep=False)
    weeks = sorted(int(w) for w in cell.train["week"].unique())
    val_weeks = weeks[-VAL_WEEKS:]
    tr = cell.train[~cell.train["week"].isin(val_weeks)].reset_index(drop=True)
    va = cell.train[cell.train["week"].isin(val_weeks)].reset_index(drop=True)
    ctx_cols = [c for c in cell.holdout_context.columns if c in va.columns]
    tune_cell = replace(cell, train=tr, holdout_context=va[ctx_cols].copy(), eval_weeks=val_weeks)

    rng = np.random.default_rng(SEED + ["lgbm", "xgb", "rf", "dml"].index(model))
    configs = [DEFAULTS[model]] + [draw(model, rng) for _ in range(n_trials - 1)]
    done = {}
    if trials_path.exists():
        for line in trials_path.read_text().splitlines():
            r = json.loads(line)
            done[r["trial"]] = r
    for i, p in enumerate(configs):
        if i in done:
            continue
        t0 = time.time()
        fs = FeatureSpace(tune_cell)
        m = make(model, p)(fs).fit(tr)
        q = np.asarray(m.predict_units(va[ctx_cols]), float)
        y = va["units"].to_numpy(float)
        wmape = float(np.abs(q - y).sum() / y.sum())
        rec = {"trial": i, "params": p, "val_forecast_wmape": wmape, "seconds": round(time.time() - t0, 1)}
        with open(trials_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        done[i] = rec
        print(json.dumps(rec), flush=True)
    best = min(done.values(), key=lambda r: r["val_forecast_wmape"])

    RPM.MODELS[model] = make(model, best["params"])
    sys.argv = ["run_panel_model.py", str(cell_dir), str(panel_dir), str(out_root), model]
    RPM.main()
    record = {
        "objective": "forecast WMAPE on the last 16 public training weeks (realized units, public rows)",
        "train_weeks": [weeks[0], weeks[-VAL_WEEKS - 1]], "validation_weeks": [val_weeks[0], val_weeks[-1]],
        "search": "random search, log-uniform where stated; trial 0 = fixed default settings",
        "seed": SEED, "n_trials": n_trials, "best_trial": best["trial"], "best_params": best["params"],
        "default_val_forecast_wmape": done[0]["val_forecast_wmape"],
        "best_val_forecast_wmape": best["val_forecast_wmape"],
        "trials": [done[i] for i in sorted(done)],
    }
    (out_root / cell_dir.name / model / "tuning.json").write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
