import hashlib, json, os, platform, sys, time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from card_eval_lib import FeatureSpace, load_cell
import card_eval_models as M
import card_eval_run as R

MODELS = {
    "lgbm": lambda fs: M.LightGBMModel(fs),
    "xgb": lambda fs: M.XGBoostModel(fs),
    "rf": lambda fs: M.RandomForestModel(fs, max_train=None),
    "dml": lambda fs: M.DoubleMLDepthIV(fs, max_train=None),
    "hier": lambda fs: M.HierarchicalDepthCF(fs, max_train=None),
}
KEY = ["product_id", "store_id", "week"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    cell_dir, panel_dir, out_root, mkey = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    out = out_root / cell_dir.name / mkey
    out.mkdir(parents=True, exist_ok=True)
    t = {}
    t0 = time.time()
    cell = load_cell(cell_dir, load_sweep=False)
    ctx = pd.read_csv(panel_dir / "public" / "counterfactual_sweep_context_panel.csv")
    ctx["product_id"] = ctx.product_id.astype(str); ctx["store_id"] = ctx.store_id.astype(str); ctx["week"] = ctx.week.astype(int)
    cell = replace(cell, sweep_context=ctx)
    t["load_s"] = round(time.time() - t0, 1)

    fs = FeatureSpace(cell)
    t0 = time.time(); model = MODELS[mkey](fs).fit(cell.train); t["fit_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    l1 = R.build_layer1(model, cell)[KEY + ["predicted_units"]]
    l1.to_csv(out / "layer1_predictions.csv.gz", index=False, float_format="%.6g", compression={"method": "gzip", "compresslevel": 1})
    t["layer1_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    l3 = R.build_layer3(model, cell)[["intervention_id"] + KEY + ["predicted_delta_units"]]
    l3.to_csv(out / "panel_deltas.csv.gz", index=False, float_format="%.6g", compression={"method": "gzip", "compresslevel": 1})
    t["deltas_s"] = round(time.time() - t0, 1)

    def params(m):
        inner = getattr(m, "model", None)
        try:
            return {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in inner.get_params().items()}
        except Exception:
            return {k: str(v) for k, v in vars(m).items() if isinstance(v, (int, float, str, bool, tuple, list)) and not k.startswith("_")}

    pkgs = {}
    for mod in ["numpy", "pandas", "sklearn", "lightgbm", "xgboost", "statsmodels", "scipy"]:
        try:
            pkgs[mod] = __import__(mod).__version__
        except Exception:
            pass
    manifest = {
        "dataset": cell_dir.name, "cell_dir": str(cell_dir), "panel_dir": str(panel_dir),
        "data_sha256": {"public/transactions_train_public.csv": sha256(cell_dir / "public" / "transactions_train_public.csv"),
                        "panel context": sha256(panel_dir / "public" / "counterfactual_sweep_context_panel.csv")},
        "model_key": mkey, "model_class": type(model).__name__, "model_name": getattr(model, "name", ""),
        "model_params": params(model),
        "features_numeric": list(fs.NUMERIC), "instrument_information_in_features": "log_promo_cost" in fs.NUMERIC,
        "training_rows": int(len(cell.train)), "training_data": "all rows (no sampler)", "random_state": 42,
        "rows_written": {"layer1": int(len(l1)), "panel_deltas": int(len(l3))},
        "code_sha256": {f: sha256(HERE / f) for f in ["card_eval_lib.py", "card_eval_models.py", "card_eval_run.py", "run_panel_model.py"]},
        "packages": pkgs, "python": platform.python_version(),
        "hardware": {"machine": platform.machine(), "processor": platform.processor(), "platform": platform.platform(), "cpu_count": os.cpu_count()},
        "env": {k: os.environ[k] for k in ["CARD_KEEP_PROMO_COST", "CARD_PRICE_ENDING_CENT"] if k in os.environ},
        "timings_s": t, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"dataset": cell_dir.name, "model": mkey, **t}))


if __name__ == "__main__":
    main()
