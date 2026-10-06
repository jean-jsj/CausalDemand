from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import os

import numpy as np
import pandas as pd


SWEEP_CONTEXT_FILE = "counterfactual_sweep_context_public.csv"
HEADLINE_INTERVENTION = "sweep_single_share_highest_plus10"


@dataclass
class Cell:
    cell_dir: Path
    name: str
    family: str
    eval_weeks: list[int]
    train: pd.DataFrame
    holdout_context: pd.DataFrame
    products: pd.DataFrame
    stores: pd.DataFrame
    seasonality: pd.DataFrame
    sweep_context: pd.DataFrame
    product_ids: list[str] = field(default_factory=list)


def _read_run_config(cell_dir: Path) -> dict[str, Any]:
    params_path = cell_dir / "release" / "scoring_params.json"
    if params_path.exists():
        return json.loads(params_path.read_text())
    return json.loads((cell_dir / "reports" / "run_config_resolved.json").read_text())["config"]


def load_cell(cell_dir: str | Path, load_sweep: bool = True) -> Cell:
    cell_dir = Path(cell_dir)
    public = cell_dir / "public"
    if not public.is_dir():
        raise FileNotFoundError(f"{cell_dir} has no public/ directory")

    cfg = _read_run_config(cell_dir)
    family = cfg["benchmark_family"]["active_cell"]["model_family"]
    n_eval = int(cfg["simulation"]["counterfactual_eval_weeks"])

    train = pd.read_csv(public / "transactions_train_public.csv")
    holdout = pd.read_csv(public / "transactions_holdout_context_public.csv")
    products = pd.read_csv(public / "products_public.csv")
    stores = pd.read_csv(public / "stores_public.csv")
    seasonality_path = public / "seasonality_index_public.csv"
    if seasonality_path.exists():
        seasonality = pd.read_csv(seasonality_path)
    else:
        seasonality = pd.DataFrame({"week": pd.Series(dtype=int),
                                    "seasonality_index": pd.Series(dtype=float)})
    if load_sweep:
        sweep = pd.read_csv(public / SWEEP_CONTEXT_FILE)
    else:
        sweep = pd.DataFrame(columns=["intervention_id", "product_id", "store_id",
                                      "week", "baseline_price", "intervention_price",
                                      "promo_cost"])

    eval_weeks = sorted(int(w) for w in holdout["week"].unique())[-n_eval:]

    for df in (train, holdout, sweep):
        df["product_id"] = df["product_id"].astype(str)
        df["store_id"] = df["store_id"].astype(str)
    products["product_id"] = products["product_id"].astype(str)

    return Cell(
        cell_dir=cell_dir,
        name=cell_dir.name,
        family=family,
        eval_weeks=eval_weeks,
        train=train,
        holdout_context=holdout,
        products=products,
        stores=stores,
        seasonality=seasonality,
        sweep_context=sweep,
        product_ids=sorted(products["product_id"].astype(str).unique()),
    )


def subsample_stores(cell: Cell, n_stores: int | None, seed: int = 0) -> Cell:
    if n_stores is None:
        return cell
    rng = np.random.default_rng(seed)
    all_stores = cell.train["store_id"].unique()
    if n_stores >= len(all_stores):
        return cell
    keep = set(rng.choice(all_stores, size=n_stores, replace=False).tolist())

    def flt(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["store_id"].isin(keep)].reset_index(drop=True)

    return Cell(
        cell_dir=cell.cell_dir,
        name=cell.name,
        family=cell.family,
        eval_weeks=cell.eval_weeks,
        train=flt(cell.train),
        holdout_context=flt(cell.holdout_context),
        products=cell.products,
        stores=cell.stores[cell.stores["store_id"].astype(str).isin(keep)].reset_index(drop=True),
        seasonality=cell.seasonality,
        sweep_context=flt(cell.sweep_context),
        product_ids=cell.product_ids,
    )


def subsample_store_weeks(cell: Cell, n_store_weeks: int | None, seed: int = 0) -> Cell:
    if n_store_weeks is None:
        return cell
    train = cell.train
    keys = (train[["store_id", "week"]].drop_duplicates()
            .sort_values(["store_id", "week"]).reset_index(drop=True))
    if n_store_weeks >= len(keys):
        return cell
    order = np.random.default_rng(seed).permutation(len(keys))
    chosen = keys.iloc[order[:n_store_weeks]]
    keep = set(zip(chosen["store_id"].astype(str), chosen["week"].astype(int)))
    mask = [(s, w) in keep for s, w in zip(train["store_id"].astype(str), train["week"].astype(int))]
    sub = train[np.asarray(mask)].reset_index(drop=True)
    return Cell(cell_dir=cell.cell_dir, name=cell.name, family=cell.family, eval_weeks=cell.eval_weeks,
                train=sub, holdout_context=cell.holdout_context, products=cell.products, stores=cell.stores,
                seasonality=cell.seasonality, sweep_context=cell.sweep_context, product_ids=cell.product_ids)


def competitor_log_price(frame: pd.DataFrame, price_col: str = "price") -> pd.Series:
    logp = np.log(frame[price_col].to_numpy(dtype=float))
    work = pd.DataFrame({"store_id": frame["store_id"].values,
                         "week": frame["week"].values, "logp": logp})
    grp = work.groupby(["store_id", "week"])["logp"]
    sw_sum = grp.transform("sum")
    sw_cnt = grp.transform("count")
    denom = (sw_cnt - 1).replace(0, np.nan)
    comp = (sw_sum - work["logp"]) / denom
    comp = comp.fillna(work["logp"])
    return pd.Series(comp.to_numpy(), index=frame.index)


class FeatureSpace:
    NUMERIC = ["logprice", "comp_logprice", "promo_flag", "log_promo_cost",
               "seasonality", "log_household", "week_idx"]
    if os.environ.get("CARD_KEEP_PROMO_COST", "0") != "1":
        NUMERIC = [c for c in NUMERIC if c != "log_promo_cost"]

    def __init__(self, cell: Cell):
        self.cell = cell
        self.week0 = int(cell.train["week"].min())
        self.season_map = dict(zip(cell.seasonality["week"].astype(int),
                                   cell.seasonality["seasonality_index"].astype(float)))
        hh = cell.stores.copy()
        hh["store_id"] = hh["store_id"].astype(str)
        self.household_map = dict(zip(hh["store_id"], hh["household_count"].astype(float)))
        brand = cell.products.copy()
        brand["product_id"] = brand["product_id"].astype(str)
        self.brand_map = dict(zip(brand["product_id"], brand["brand_code"].astype(str)))
        self.product_vocab = {p: i for i, p in enumerate(sorted(self.brand_map))}
        self.brand_vocab = {b: i for i, b in enumerate(sorted(set(self.brand_map.values())))}
        stores_sorted = sorted(cell.train["store_id"].astype(str).unique())
        self.store_vocab = {s: i for i, s in enumerate(stores_sorted)}

    def raw_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        f = frame.copy()
        f["product_id"] = f["product_id"].astype(str)
        f["store_id"] = f["store_id"].astype(str)
        f["week"] = f["week"].astype(int)
        f["logprice"] = np.log(f["price"].astype(float))
        f["comp_logprice"] = competitor_log_price(f, "price")
        f["promo_flag"] = pd.to_numeric(f.get("promo_flag", 0), errors="coerce").fillna(0.0)
        f["log_promo_cost"] = np.log1p(pd.to_numeric(
            f.get("promo_cost", 0.0), errors="coerce").fillna(0.0).clip(lower=0))
        f["seasonality"] = f["week"].map(self.season_map).astype(float).fillna(1.0)
        f["log_household"] = np.log1p(
            f["store_id"].map(self.household_map).astype(float).fillna(0.0))
        f["week_idx"] = f["week"].astype(int) - self.week0
        f["product_code"] = f["product_id"].map(self.product_vocab).fillna(-1).astype(int)
        f["store_code"] = f["store_id"].map(self.store_vocab).fillna(-1).astype(int)
        f["brand_code_id"] = (f["product_id"].map(self.brand_map)
                              .map(self.brand_vocab).fillna(-1).astype(int))
        return f

    def tree_matrix(self, f: pd.DataFrame) -> pd.DataFrame:
        cols = self.NUMERIC + ["product_code", "store_code", "brand_code_id"]
        return f[cols].astype(float)

    def linear_design(self, f: pd.DataFrame, product_fe: bool = True) -> pd.DataFrame:
        base = f[list(self.NUMERIC)].astype(float).reset_index(drop=True)
        if product_fe:
            codes = pd.Categorical(f["product_code"].astype(int),
                                   categories=list(range(len(self.product_vocab))))
            pd_dum = pd.get_dummies(codes, prefix="prod", drop_first=True).astype(float)
            base = pd.concat([base, pd_dum.reset_index(drop=True)], axis=1)
        return base.astype(float)


class BaseUnitsModel:
    name = "base"

    def __init__(self, fs: FeatureSpace):
        self.fs = fs

    def fit(self, train: pd.DataFrame) -> "BaseUnitsModel":
        raise NotImplementedError

    def _predict_units_raw(self, f: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_units(self, frame: pd.DataFrame) -> np.ndarray:
        f = self.fs.raw_features(frame)
        pred = np.asarray(self._predict_units_raw(f), dtype=float)
        return np.clip(pred, 0.0, None)
