import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent
sys.path.insert(0, str(HARNESS))
from card_eval_lib import FeatureSpace, load_cell, subsample_store_weeks
import card_eval_models as M
import card_eval_run as R

KEY = ["product_id", "store_id", "week"]
SIZES = {"tabpfn": 6400, "tabfm": 400, "chronos2": None}


class Chronos2LogMissing(M.Chronos2Model):
    name = "Chronos-2 (log units, no-sale weeks missing)"

    def __init__(self, fs, **kw):
        super().__init__(fs, point="median", **kw)

    def fit(self, train: pd.DataFrame) -> "Chronos2LogMissing":
        grid = self._carried_grid(train)
        g = self.fs.raw_features(grid)
        g = g.sort_values(["product_id", "store_id", "week"]).reset_index(drop=True)
        self.last_train_week_ = int(g["week"].max())
        starts = g.groupby(["product_id", "store_id"], sort=False).indices
        present = g["present"].to_numpy(dtype=bool)
        target = np.where(present, M._log_units(g["units"].to_numpy()), np.nan).astype(np.float32)
        cov = self._covariate_matrix(g)
        self.series_keys_, self.series_pos_, self.targets_, self.past_cov_ = [], {}, [], []
        last_cov = []
        for pos, (key, idx) in enumerate(starts.items()):
            self.series_keys_.append(key)
            self.series_pos_[key] = pos
            self.targets_.append(np.ascontiguousarray(target[idx]))
            block = cov[idx]
            self.past_cov_.append({name: np.ascontiguousarray(block[:, k])
                                   for k, name in enumerate(self.covariates)})
            last_cov.append(block[-1])
        self.last_cov_ = np.asarray(last_cov, dtype=np.float32)
        self.context_rows_ = int(len(g))
        self.context_products_ = int(g["product_id"].nunique())
        self.n_series_ = len(self.series_keys_)
        self.missing_row_share_ = float(1.0 - present.mean())
        self._load_pipeline()
        return self

    def _predict_units_raw(self, f: pd.DataFrame) -> np.ndarray:
        log_median = super()._predict_units_raw(f)
        if getattr(self, "unseen_series_", None):
            raise RuntimeError(f"{len(self.unseen_series_)} series without training history")
        return np.exp(np.clip(log_median, -20.0, 20.0))


def make_model(key: str, fs: FeatureSpace):
    if key == "tabpfn":
        return M.TabPFNModel(fs)
    if key == "tabfm":
        return M.TabFMModel(fs, max_train=None, predict_chunk=2048)
    if key == "chronos2":
        return Chronos2LogMissing(fs, batch_size=256, predict_chunk=4096)
    raise ValueError(key)


def _predict(model, f: pd.DataFrame) -> np.ndarray:
    return np.clip(np.asarray(model._predict_units_raw(f), dtype=float), 0.0, None)


def panel_context(cell) -> pd.DataFrame:
    sw = cell.sweep_context.copy()
    flags = cell.holdout_context[KEY + ["promo_flag"]].drop_duplicates(KEY)
    sw = sw.merge(flags, on=KEY, how="left", validate="many_to_one")
    fallback = (pd.to_numeric(sw["promo_cost"], errors="coerce").fillna(0.0) != 0.0).astype(int)
    sw["promo_flag"] = pd.to_numeric(sw["promo_flag"], errors="coerce").fillna(fallback).astype(int)
    sw["_moved"] = ~np.isclose(sw["baseline_price"].astype(float), sw["intervention_price"].astype(float))
    return sw


def needed_rows(f: pd.DataFrame, moved: np.ndarray, whole_series: bool) -> np.ndarray:
    if not whole_series:
        return moved
    pairs = set(zip(f["product_id"].to_numpy()[moved], f["store_id"].to_numpy()[moved]))
    return np.fromiter(((p, s) in pairs for p, s in zip(f["product_id"].to_numpy(), f["store_id"].to_numpy())),
                       dtype=bool, count=len(f))


def build_deltas(model, fs: FeatureSpace, sw: pd.DataFrame, whole_series: bool, log=print) -> pd.DataFrame:
    base = sw.drop_duplicates(KEY).rename(columns={"baseline_price": "price"}).reset_index(drop=True)
    any_moved = base[KEY].merge(sw.loc[sw["_moved"], KEY].drop_duplicates(), on=KEY, how="left",
                                indicator=True)["_merge"].eq("both").to_numpy()
    fb = fs.raw_features(base)
    sel = needed_rows(fb, any_moved, whole_series)
    q_base = pd.Series(_predict(model, fb[sel]), index=pd.MultiIndex.from_frame(fb.loc[sel, KEY]))
    log(f"  baseline: {int(sel.sum()):,} rows predicted")
    out = []
    for iid, grp in sw.groupby("intervention_id", sort=False):
        grp = grp.reset_index(drop=True)
        moved = grp["_moved"].to_numpy()
        fcf = fs.raw_features(grp.rename(columns={"intervention_price": "price"}))
        sel = needed_rows(fcf, moved, whole_series)
        q = pd.Series(_predict(model, fcf[sel]), index=pd.MultiIndex.from_frame(fcf.loc[sel, KEY]))
        keys = pd.MultiIndex.from_frame(fcf.loc[moved, KEY])
        d = grp.loc[moved, ["intervention_id"] + KEY].copy()
        d["predicted_delta_units"] = q.reindex(keys).to_numpy() - q_base.reindex(keys).to_numpy()
        out.append(d)
    deltas = pd.concat(out, ignore_index=True)
    if deltas["predicted_delta_units"].isna().any():
        raise RuntimeError("missing predictions on moved rows")
    return deltas


def _swap_caches(model, caches=None):
    if not hasattr(model, "_pinned"):
        return None
    old = (model._pinned, model._rolling)
    model._pinned, model._rolling = caches if caches is not None else ({}, {})
    return old


def invariance_check(model, fs: FeatureSpace, sw: pd.DataFrame, whole_series: bool, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    iid = sw.loc[sw["_moved"]].groupby("intervention_id").size().idxmax()
    grp = sw[sw["intervention_id"] == iid].reset_index(drop=True)
    fcf = fs.raw_features(grp.rename(columns={"intervention_price": "price"}))
    sel = needed_rows(fcf, grp["_moved"].to_numpy(), whole_series)
    if whole_series:
        pairs = list(zip(fcf["product_id"], fcf["store_id"]))
        own = {p for p, s in zip(pairs, sel) if s}
        others = sorted({p for p in pairs if p not in own})
        pick = {others[i] for i in rng.choice(len(others), size=min(2000, len(others)), replace=False)}
        extra = np.fromiter((p in pick for p in pairs), dtype=bool, count=len(pairs))
    else:
        pool = np.flatnonzero(~sel)
        extra = np.zeros(len(fcf), dtype=bool)
        extra[rng.choice(pool, size=min(50_000, len(pool)), replace=False)] = True
    order = rng.permutation(np.flatnonzero(sel | extra))
    saved = _swap_caches(model)
    t0 = time.time()
    a = _predict(model, fcf[sel])
    _swap_caches(model)
    b_all = _predict(model, fcf.iloc[order])
    _swap_caches(model, saved)
    b = pd.Series(b_all, index=order).reindex(np.flatnonzero(sel)).to_numpy()
    diff = np.abs(a - b)
    return {"scenario": iid, "rows_checked": int(sel.sum()), "extra_rows": int(extra.sum()),
            "max_abs_diff": float(diff.max()), "max_rel_diff": float((diff / np.maximum(np.abs(a), 1e-9)).max()),
            "mean_units": float(a.mean()), "seconds": round(time.time() - t0, 1)}


def layout_diagnostic(model, fs: FeatureSpace, sw: pd.DataFrame, iids: list) -> pd.DataFrame:
    base = sw.drop_duplicates(KEY).rename(columns={"baseline_price": "price"}).reset_index(drop=True)
    fb = fs.raw_features(base)
    idx = pd.MultiIndex.from_frame(fb[KEY])
    base_full = pd.Series(_predict(model, fb), index=idx)
    any_moved = idx.isin(pd.MultiIndex.from_frame(sw.loc[sw["_moved"], KEY]))
    base_r = pd.Series(_predict(model, fb[any_moved]), index=idx[any_moved])
    out = []
    for iid in iids:
        grp = sw[sw["intervention_id"] == iid].reset_index(drop=True)
        moved = grp["_moved"].to_numpy()
        fcf = fs.raw_features(grp.rename(columns={"intervention_price": "price"}))
        keys = pd.MultiIndex.from_frame(fcf.loc[moved, KEY])
        d = grp.loc[moved, ["intervention_id"] + KEY].copy()
        d["base_full"] = base_full.reindex(keys).to_numpy()
        d["base_restricted"] = base_r.reindex(keys).to_numpy()
        d["cf_full"] = _predict(model, fcf)[moved]
        d["cf_restricted"] = _predict(model, fcf[moved])
        d["cf_restricted_again"] = _predict(model, fcf[moved])
        out.append(d)
    return pd.concat(out, ignore_index=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def versions() -> dict:
    import importlib.metadata as md
    out = {"python": platform.python_version()}
    for p in ["numpy", "pandas", "scikit-learn", "scipy", "torch", "tabpfn", "tabfm", "chronos-forecasting",
              "transformers", "huggingface-hub"]:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    return out


def hardware() -> dict:
    hw = {"platform": platform.platform(), "cpu_count": os.cpu_count()}
    try:
        import torch
        if torch.cuda.is_available():
            hw["gpu"] = torch.cuda.get_device_name(0)
            hw["cuda"] = torch.version.cuda
    except ImportError:
        pass
    return hw


def load(cell_dir: Path, panel_dir: Path):
    cell = load_cell(cell_dir, load_sweep=False)
    ctx = pd.read_csv(panel_dir / "public" / "counterfactual_sweep_context_panel.csv")
    ctx["product_id"] = ctx.product_id.astype(str)
    ctx["store_id"] = ctx.store_id.astype(str)
    ctx["week"] = ctx.week.astype(int)
    return replace(cell, sweep_context=ctx)


def run(cell_dir: Path, panel_dir: Path, out: Path, key: str, sampler_seed: int = 0, check: bool = False,
        log=print, store_weeks="model default") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    t = {}
    t0 = time.time()
    cell = load(cell_dir, panel_dir)
    fs = FeatureSpace(cell)
    size = SIZES[key] if store_weeks == "model default" else store_weeks
    sub = subsample_store_weeks(cell, size, sampler_seed) if size is not None else cell
    sw = panel_context(cell)
    t["load_s"] = round(time.time() - t0, 1)

    model = make_model(key, fs)
    t0 = time.time(); model.fit(sub.train); t["fit_s"] = round(time.time() - t0, 1)
    log(f"  fit {t['fit_s']}s on {len(sub.train):,} rows")

    t0 = time.time()
    l1 = R.build_layer1(model, cell)[KEY + ["predicted_units"]]
    l1.to_csv(out / "layer1_predictions.csv.gz", index=False, float_format="%.6g",
              compression={"method": "gzip", "compresslevel": 1})
    t["layer1_s"] = round(time.time() - t0, 1)
    log(f"  layer1 {t['layer1_s']}s, {len(l1):,} rows")

    inv = None
    if check:
        inv = invariance_check(model, fs, sw, whole_series=(key == "chronos2"))
        log(f"  invariance check: {inv}")

    t0 = time.time()
    deltas = build_deltas(model, fs, sw, whole_series=(key == "chronos2"), log=log)
    deltas.to_csv(out / "panel_deltas.csv.gz", index=False, float_format="%.6g",
                  compression={"method": "gzip", "compresslevel": 1})
    t["deltas_s"] = round(time.time() - t0, 1)
    log(f"  deltas {t['deltas_s']}s, {len(deltas):,} moved rows")

    public = cell_dir / "public"
    settings = {k: v for k, v in vars(model).items()
                if isinstance(v, (int, float, str, bool, type(None))) and not k.startswith("_")}
    settings.update(getattr(model, "settings_", {}) or {})
    if key == "chronos2":
        settings.update(covariates=list(model.covariates), cross_learning=model.cross_learning,
                        target="log units (shared floor), weeks without a public row = missing",
                        point_forecast="exp(median of the log forecast)", cache_stats=model.cache_stats_)
    man = {
        "dataset": cell_dir.name, "cell_dir": str(cell_dir), "panel_dir": str(panel_dir),
        "data_sha256": {"train": sha256(public / "transactions_train_public.csv"),
                        "panel context": sha256(panel_dir / "public" / "counterfactual_sweep_context_panel.csv")},
        "model": key, "model_class": type(model).__name__, "model_name": model.name, "settings": settings,
        "features": list(fs.NUMERIC) + (["product_code", "store_code", "brand_code_id"]
                                        if key != "chronos2" else ["(one series per product x store)"]),
        "instrument_information_in_features": any(c in fs.NUMERIC for c in ("log_promo_cost", "promo_cost",
                                                                              "supply_cost_proxy")),
        "context": {"store_weeks": size, "sampler_seed": sampler_seed if size is not None else None,
                    "rows": int(len(sub.train)), "stores": int(sub.train["store_id"].nunique())},
        "random_state": 42,
        "rows_written": {"layer1": int(len(l1)), "panel_deltas": int(len(deltas))},
        "panel_deltas_rows": "moved rows only: predictions are written for the rows whose price changes",
        "code_sha256": {f: sha256(HARNESS / f) for f in ["card_eval_lib.py", "card_eval_models.py",
                                                         "card_eval_run.py"]} | {"fm_runner.py": sha256(Path(__file__))},
        "versions": versions(), "hardware": hardware(), "timings_s": t, "invariance_check": inv,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if man["instrument_information_in_features"]:
        raise RuntimeError("instrument information in the feature set")
    (out / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    return man


if __name__ == "__main__":
    a = sys.argv[1:]
    run(Path(a[0]), Path(a[1]), Path(a[2]), a[3], int(a[4]) if len(a) > 4 else 0, check="check" in a[5:])
