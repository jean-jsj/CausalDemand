from __future__ import annotations

import hashlib
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from card_eval_lib import BaseUnitsModel, FeatureSpace


LOG_UNITS_FLOOR = 1e-6

PINNED_WEIGHTS = {
    "tabpfn": ("Prior-Labs/tabpfn_3", "24a16a89d245878b846555110985634aa2e656d7",
               "tabpfn-v3-regressor-v3_default.ckpt",
               "311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301"),
    "chronos2": ("amazon/chronos-2", "29ec3766d36d6f73f0696f85560a422f50e8498c", "model.safetensors",
                 "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42"),
}


def pinned_weights(key: str) -> Path:
    repo_id, revision, filename, sha256 = PINNED_WEIGHTS[key]
    from huggingface_hub import hf_hub_download, snapshot_download

    if key == "chronos2":
        path = Path(snapshot_download(repo_id=repo_id, revision=revision)) / filename
    else:
        path = Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision))
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    if h.hexdigest() != sha256:
        raise RuntimeError(f"{repo_id}@{revision[:12]}/{filename}: SHA-256 {h.hexdigest()[:16]}... is not the "
                           f"{sha256[:16]}... of the weights the released runs used")
    return path


def _log_units(units: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(units, dtype=float), LOG_UNITS_FLOOR, None))


def _drop_collinear(design: pd.DataFrame, protect: list[str] | None = None,
                    tol: float = 1e-9) -> pd.DataFrame:
    protect = protect or []
    ordered = [c for c in protect if c in design.columns] + \
              [c for c in design.columns if c not in protect]
    A = design[ordered].to_numpy(dtype=float)
    _, r = np.linalg.qr(A)
    diag = np.abs(np.diag(r))
    thresh = tol * (diag.max() if diag.size else 1.0)
    keep_mask = diag > thresh
    n_protect = sum(1 for c in protect if c in design.columns)
    keep_mask[:n_protect] = True
    kept = [c for c, k in zip(ordered, keep_mask) if k]
    return design[kept]


class LightGBMModel(BaseUnitsModel):
    name = "LightGBM"

    def __init__(self, fs: FeatureSpace, n_estimators: int = 300, max_depth: int = -1,
                 learning_rate: float = 0.05, num_leaves: int = 63, random_state: int = 42):
        super().__init__(fs)
        from lightgbm import LGBMRegressor
        self.model = LGBMRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            num_leaves=num_leaves, random_state=random_state, n_jobs=-1, verbose=-1)
        self.cat_features = ["product_code", "store_code", "brand_code_id"]

    def fit(self, train: pd.DataFrame) -> "LightGBMModel":
        f = self.fs.raw_features(train)
        X = self.fs.tree_matrix(f)
        y = _log_units(train["units"].to_numpy())
        cat_idx = [X.columns.get_loc(c) for c in self.cat_features]
        self.model.fit(X, y, categorical_feature=cat_idx)
        return self

    def _predict_units_raw(self, f: pd.DataFrame) -> np.ndarray:
        X = self.fs.tree_matrix(f)
        return np.exp(self.model.predict(X))


class RandomForestModel(BaseUnitsModel):
    name = "Random Forest"

    def __init__(self, fs: FeatureSpace, n_estimators: int = 200, max_depth: int | None = 18,
                 max_train: int | None = 300_000, random_state: int = 42):
        super().__init__(fs)
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1,
            random_state=random_state, min_samples_leaf=5)
        self.max_train = max_train
        self.random_state = random_state

    def fit(self, train: pd.DataFrame) -> "RandomForestModel":
        train = train.reset_index(drop=True)
        f = self.fs.raw_features(train)
        if self.max_train is not None and len(train) > self.max_train:
            idx = train.sample(
                self.max_train, random_state=self.random_state
            ).index
            train = train.loc[idx].reset_index(drop=True)
            f = f.loc[idx].reset_index(drop=True)
        X = self.fs.tree_matrix(f)
        y = _log_units(train["units"].to_numpy())
        self.model.fit(X, y)
        return self

    def _predict_units_raw(self, f: pd.DataFrame) -> np.ndarray:
        return np.exp(self.model.predict(self.fs.tree_matrix(f)))


class XGBoostModel(BaseUnitsModel):
    name = "XGBoost"

    def __init__(self, fs: FeatureSpace, n_estimators: int = 300, max_depth: int = 8,
                 learning_rate: float = 0.05, random_state: int = 42):
        super().__init__(fs)
        from xgboost import XGBRegressor
        self.model = XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            random_state=random_state, n_jobs=-1, verbosity=0,
            tree_method="hist", enable_categorical=True)
        self.cat_features = ["product_code", "store_code", "brand_code_id"]

    def _matrix(self, f: pd.DataFrame) -> pd.DataFrame:
        X = self.fs.tree_matrix(f)
        sizes = {"product_code": len(self.fs.product_vocab),
                 "store_code": len(self.fs.store_vocab),
                 "brand_code_id": len(self.fs.brand_vocab)}
        for c, n in sizes.items():
            X[c] = pd.Categorical(X[c].astype(int), categories=[-1] + list(range(n)))
        return X

    def fit(self, train: pd.DataFrame) -> "XGBoostModel":
        f = self.fs.raw_features(train)
        y = _log_units(train["units"].to_numpy())
        self.model.fit(self._matrix(f), y)
        return self

    def _predict_units_raw(self, f: pd.DataFrame) -> np.ndarray:
        return np.exp(self.model.predict(self._matrix(f)))


class TabFMModel(BaseUnitsModel):
    name = "TabFM 1.0.0 (PyTorch)"
    checkpoint = "google/tabfm-1.0.0-pytorch"

    def __init__(
        self,
        fs: FeatureSpace,
        max_train: int = 512,
        n_estimators: int = 1,
        random_state: int = 42,
        device: str | None = None,
        predict_chunk: int | None = 256,
    ):
        super().__init__(fs)
        self.max_train = max_train
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.device = device
        self.predict_chunk = predict_chunk
        self.backend = "tabfm-pytorch"
        self.model = None

    @staticmethod
    def _context_indices(
        frame: pd.DataFrame,
        max_rows: int | None,
        random_state: int,
    ) -> np.ndarray:
        if max_rows is None or len(frame) <= max_rows:
            return np.arange(len(frame), dtype=int)
        if max_rows <= 0:
            raise ValueError("max_train must be positive or None")

        rng = np.random.default_rng(random_state)
        groups = {
            product: indices.to_numpy(dtype=int)
            for product, indices in frame.groupby("product_id", sort=True).groups.items()
        }
        products = list(groups)
        if max_rows < len(products):
            products = list(rng.choice(products, size=max_rows, replace=False))

        quota = max(1, max_rows // len(products))
        selected: list[int] = []
        for product in products:
            indices = groups[product]
            take = min(quota, len(indices))
            selected.extend(rng.choice(indices, size=take, replace=False).tolist())

        if len(selected) < max_rows:
            remaining = np.setdiff1d(
                np.arange(len(frame), dtype=int),
                np.asarray(selected, dtype=int),
                assume_unique=False,
            )
            take = min(max_rows - len(selected), len(remaining))
            selected.extend(rng.choice(remaining, size=take, replace=False).tolist())
        return np.asarray(selected[:max_rows], dtype=int)

    def _feature_frame(self, f: pd.DataFrame) -> pd.DataFrame:
        X = f[self.fs.NUMERIC].astype(float).copy()
        X["product_id"] = f["product_id"].astype(str).to_numpy()
        X["store_id"] = f["store_id"].astype(str).to_numpy()
        X["brand_code"] = (
            f["product_id"].map(self.fs.brand_map).fillna("UNKNOWN").astype(str).to_numpy()
        )
        return X

    def fit(self, train: pd.DataFrame) -> "TabFMModel":
        try:
            from tabfm import TabFMRegressor, tabfm_v1_0_0_pytorch
        except ImportError as exc:
            raise ImportError(
                "TabFM is required. Install the PyTorch backend with "
                "`python -m pip install \"tabfm[pytorch]==1.0.1\"`."
            ) from exc
        import torch

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        use_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        foundation = tabfm_v1_0_0_pytorch.load(
            model_type="regression",
            device=device,
            dtype=dtype,
            use_cache=True,
        )
        self.dtype_ = str(dtype).replace("torch.", "")
        train = train.reset_index(drop=True)
        f = self.fs.raw_features(train)
        idx = self._context_indices(train, self.max_train, self.random_state)
        context = train.iloc[idx].reset_index(drop=True)
        context_features = f.iloc[idx].reset_index(drop=True)
        X = self._feature_frame(context_features)
        y = _log_units(context["units"].to_numpy())
        self.model = TabFMRegressor(
            model=foundation,
            n_estimators=self.n_estimators,
            max_num_features=None,
            max_num_rows=None,
            batch_size=1,
            random_state=self.random_state,
        )
        self.model.fit(X, y)
        self.device_ = device
        self.context_rows_ = len(context)
        self.context_products_ = context["product_id"].nunique()
        return self

    def _predict_units_raw(self, f: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("TabFMModel must be fitted before prediction")
        X = self._feature_frame(f)
        cs = self.predict_chunk
        if cs is None or len(X) <= cs:
            log_pred = self.model.predict(X)
        else:
            starts = range(0, len(X), cs)
            try:
                from tqdm.auto import tqdm
                starts = tqdm(starts, desc=f"TabFM predict ({len(X):,} rows)",
                              unit="chunk", leave=False)
            except ImportError:
                pass
            parts = [self.model.predict(X.iloc[i:i + cs]) for i in starts]
            log_pred = np.concatenate(parts)
        return np.exp(np.clip(log_pred, -20.0, 20.0))


class _InContextTabularModel(BaseUnitsModel):
    cat_features = ["product_code", "store_code", "brand_code_id"]

    def __init__(self, fs: FeatureSpace, random_state: int = 42, device: str | None = None,
                 predict_chunk: int | None = None):
        super().__init__(fs)
        self.random_state = random_state
        self.device = device
        self.predict_chunk = predict_chunk
        self.model = None

    def _feature_frame(self, f: pd.DataFrame) -> pd.DataFrame:
        X = self.fs.tree_matrix(f)
        for c in self.cat_features:
            X[c] = X[c].astype(int)
        return X

    def _cat_indices(self, X: pd.DataFrame) -> list[int]:
        return [X.columns.get_loc(c) for c in self.cat_features]

    def _resolve_device(self) -> str:
        import torch
        return self.device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _predict_log(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def _predict_units_raw(self, f: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"{type(self).__name__} must be fitted before prediction")
        X = self._feature_frame(f)
        cs = self.predict_chunk
        if cs is None or len(X) <= cs:
            log_pred = np.asarray(self._predict_log(X), dtype=float)
        else:
            parts = [np.asarray(self._predict_log(X.iloc[i:i + cs]), dtype=float)
                     for i in range(0, len(X), cs)]
            log_pred = np.concatenate(parts)
        return np.exp(np.clip(log_pred, -20.0, 20.0))


class TabPFNModel(_InContextTabularModel):
    name = "TabPFN"

    def __init__(self, fs: FeatureSpace, random_state: int = 42, device: str | None = None,
                 predict_chunk: int | None = 8192, fit_mode: str = "fit_with_cache",
                 model_version: str | None = None):
        super().__init__(fs, random_state=random_state, device=device, predict_chunk=predict_chunk)
        self.fit_mode = fit_mode
        self.model_version = model_version

    def fit(self, train: pd.DataFrame) -> "TabPFNModel":
        try:
            import tabpfn
            from tabpfn import TabPFNRegressor
        except ImportError as exc:
            raise ImportError("TabPFN is required: `python -m pip install -r "
                              "causaldemand/models/requirements-fm.txt`") from exc
        train = train.reset_index(drop=True)
        X = self._feature_frame(self.fs.raw_features(train))
        y = _log_units(train["units"].to_numpy())
        device = self._resolve_device()
        kw = dict(random_state=self.random_state, device=device, fit_mode=self.fit_mode,
                  categorical_features_indices=self._cat_indices(X))
        if self.model_version is None:
            self.model = TabPFNRegressor(model_path=str(pinned_weights("tabpfn")), **kw)
        else:
            from tabpfn.constants import ModelVersion
            self.model = TabPFNRegressor.create_default_for_version(ModelVersion(self.model_version), **kw)
        self.model.fit(X, y)
        self.device_ = device
        self.dtype_ = str(getattr(self.model, "forced_inference_dtype_", None) or "auto")
        self.context_rows_ = int(len(X))
        self.context_products_ = int(train["product_id"].nunique())
        self.settings_ = {
            "package": "tabpfn", "version": tabpfn.__version__,
            "model_version": self.model_version or "package default",
            "model_path": ("{}@{}/{}".format(*PINNED_WEIGHTS["tabpfn"][:3]) if self.model_version is None
                           else str(getattr(self.model, "model_path", "auto"))),
            "n_estimators": int(getattr(self.model, "n_estimators_", 0)) or None,
            "fit_mode": self.fit_mode, "inference_precision": str(self.model.inference_precision),
            "use_autocast": bool(getattr(self.model, "use_autocast_", False)),
            "feature_modalities": [f.modality.value for f in
                                   getattr(self.model, "inferred_feature_schema_").features]
            if getattr(self.model, "inferred_feature_schema_", None) is not None else None,
        }
        return self

    def _predict_log(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


class TabICLModel(_InContextTabularModel):
    name = "TabICLv2"

    def __init__(self, fs: FeatureSpace, random_state: int = 42, device: str | None = None,
                 predict_chunk: int | None = 16_384, kv_cache: bool = True):
        super().__init__(fs, random_state=random_state, device=device, predict_chunk=predict_chunk)
        self.kv_cache = kv_cache

    def fit(self, train: pd.DataFrame) -> "TabICLModel":
        try:
            from tabicl import TabICLRegressor
        except ImportError as exc:
            raise ImportError("TabICL is required: `python -m pip install tabicl`") from exc
        import importlib.metadata
        train = train.reset_index(drop=True)
        X = self._feature_frame(self.fs.raw_features(train))
        y = _log_units(train["units"].to_numpy())
        device = self._resolve_device()
        self.model = TabICLRegressor(random_state=self.random_state, device=device, kv_cache=self.kv_cache)
        self.model.fit(X, y)
        self.device_ = device
        self.dtype_ = f"amp={self.model.use_amp}"
        self.context_rows_ = int(len(X))
        self.context_products_ = int(train["product_id"].nunique())
        self.settings_ = {
            "package": "tabicl", "version": importlib.metadata.version("tabicl"),
            "checkpoint": self.model.checkpoint_version,
            "model_path": str(getattr(self.model, "model_path_", None)),
            "n_estimators": int(self.model.n_estimators), "batch_size": self.model.batch_size,
            "kv_cache": self.kv_cache, "offload_mode": str(self.model.offload_mode),
        }
        return self

    def _predict_log(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


def _nearest_week_fill(frame: pd.DataFrame, value_col: str, by: list) -> pd.Series:
    grouped = frame.groupby(by, sort=False)[value_col]
    ff = grouped.ffill()
    bf = grouped.bfill()
    notna = frame[value_col].notna()
    idx = np.arange(len(frame))
    last_obs = pd.Series(np.where(notna, idx, np.nan), index=frame.index)
    next_obs = pd.Series(np.where(notna, idx, np.nan), index=frame.index)
    last_obs = last_obs.groupby([frame[c] for c in by], sort=False).ffill()
    next_obs = next_obs.groupby([frame[c] for c in by], sort=False).bfill()
    dist_prev = idx - last_obs.to_numpy()
    dist_next = next_obs.to_numpy() - idx
    use_next = (np.isnan(dist_prev)) | (
        ~np.isnan(dist_next) & (dist_next < dist_prev)
    )
    return pd.Series(np.where(use_next, bf.to_numpy(), ff.to_numpy()), index=frame.index)


class Chronos2Model(BaseUnitsModel):
    name = "Chronos-2 (covariate-native)"
    checkpoint = "amazon/chronos-2"

    def __init__(
        self,
        fs: FeatureSpace,
        covariates: tuple[str, ...] | None = None,
        point: str = "qmean",
        batch_size: int = 256,
        predict_chunk: int | None = 4096,
        context_length: int | None = None,
        cross_learning: bool = False,
        device: str | None = None,
        dtype: str | None = None,
        max_cache: int = 300_000,
        random_state: int = 42,
        pipeline=None,
    ):
        super().__init__(fs)
        self.covariates = tuple(covariates) if covariates else tuple(fs.NUMERIC)
        if point not in ("qmean", "median"):
            raise ValueError("point must be 'qmean' or 'median'")
        self.point = point
        self.batch_size = batch_size
        self.predict_chunk = predict_chunk
        self.context_length = context_length
        self.cross_learning = cross_learning
        self.device = device
        self.dtype = dtype
        self.max_cache = max_cache
        self.random_state = random_state
        self.pipeline = pipeline
        self._pinned: dict[bytes, np.ndarray] = {}
        self._rolling: dict[bytes, np.ndarray] = {}
        self.cache_stats_ = {"hits": 0, "misses": 0, "evictions": 0}


    def _carried_grid(self, train: pd.DataFrame) -> pd.DataFrame:
        tx = train.copy()
        tx["product_id"] = tx["product_id"].astype(str)
        tx["store_id"] = tx["store_id"].astype(str)
        tx["week"] = tx["week"].astype(int)

        stores = self.fs.cell.stores.copy()
        stores["store_id"] = stores["store_id"].astype(str)
        products = self.fs.cell.products.copy()
        products["product_id"] = products["product_id"].astype(str)

        weeks = np.arange(tx["week"].min(), tx["week"].max() + 1)
        pairs = tx[["product_id", "store_id"]].drop_duplicates()
        grid = (
            pairs.merge(pd.DataFrame({"week": weeks}), how="cross")
            .merge(tx, on=["product_id", "store_id", "week"], how="left")
            .merge(stores[["store_id", "chain"]], on="store_id", how="left")
            .merge(products[["product_id", "brand_code"]], on="product_id", how="left")
        )
        grid["present"] = grid["units"].notna()
        grid["units"] = grid["units"].fillna(0.0)

        obs = grid[grid["present"]]
        flag_cbw = (obs.groupby(["chain", "brand_code", "week"])["promo_flag"]
                    .first().rename("flag_cbw"))
        pcost_cbw = (obs[obs["promo_flag"] == 1]
                     .groupby(["chain", "brand_code", "week"])["promo_cost"]
                     .first().rename("pcost_cbw"))
        grid = grid.join(flag_cbw, on=["chain", "brand_code", "week"])
        grid = grid.join(pcost_cbw, on=["chain", "brand_code", "week"])

        grid = grid.sort_values(
            ["chain", "brand_code", "store_id", "product_id", "week"]
        ).reset_index(drop=True)
        by = ["chain", "brand_code", "store_id", "product_id"]

        grid["promo_flag"] = grid["promo_flag"].fillna(grid["flag_cbw"])
        still = grid["promo_flag"].isna()
        if still.any():
            grid.loc[still, "promo_flag"] = _nearest_week_fill(grid, "promo_flag", by)[still]
        grid["promo_flag"] = grid["promo_flag"].fillna(0.0).astype(int)

        grid["promo_cost"] = np.where(
            grid["promo_flag"] == 0, 0.0, grid["promo_cost"].fillna(grid["pcost_cbw"]))
        missing_pc = (grid["promo_flag"] == 1) & pd.isna(grid["promo_cost"])
        if missing_pc.any():
            grid["promo_cost"] = grid["promo_cost"].where(~missing_pc)
            filled = _nearest_week_fill(grid, "promo_cost", by)
            grid.loc[missing_pc, "promo_cost"] = filled[missing_pc]
            grid["promo_cost"] = grid["promo_cost"].fillna(0.0)

        if grid["price"].isna().any():
            grid["price"] = _nearest_week_fill(grid, "price", ["store_id", "product_id"])
        return grid.drop(columns=["flag_cbw", "pcost_cbw"])


    def _covariate_matrix(self, f: pd.DataFrame) -> np.ndarray:
        return f[list(self.covariates)].to_numpy(dtype=np.float32)

    def fit(self, train: pd.DataFrame) -> "Chronos2Model":
        grid = self._carried_grid(train)
        g = self.fs.raw_features(grid)
        g = g.sort_values(["product_id", "store_id", "week"]).reset_index(drop=True)

        self.last_train_week_ = int(g["week"].max())
        starts = g.groupby(["product_id", "store_id"], sort=False).indices

        units = g["units"].to_numpy(dtype=np.float32)
        cov = self._covariate_matrix(g)

        self.series_keys_ = []
        self.series_pos_ = {}
        self.targets_ = []
        self.past_cov_ = []
        last_cov = []
        for pos, (key, idx) in enumerate(starts.items()):
            self.series_keys_.append(key)
            self.series_pos_[key] = pos
            self.targets_.append(np.ascontiguousarray(units[idx]))
            block = cov[idx]
            self.past_cov_.append(
                {name: np.ascontiguousarray(block[:, k])
                 for k, name in enumerate(self.covariates)})
            last_cov.append(block[-1])
        self.last_cov_ = np.asarray(last_cov, dtype=np.float32)

        self.context_rows_ = int(len(g))
        self.context_products_ = int(g["product_id"].nunique())
        self.n_series_ = len(self.series_keys_)
        self.zero_row_share_ = float(1.0 - g["present"].mean())

        self._load_pipeline()
        return self

    def _load_pipeline(self) -> None:
        if self.pipeline is None:
            try:
                from chronos import Chronos2Pipeline
            except ImportError as exc:
                raise ImportError(
                    "Chronos-2 is required. Install it with "
                    '`python -m pip install "chronos-forecasting==2.3.1"`.'
                ) from exc
            import torch

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = self.dtype or "float32"
            self.pipeline = Chronos2Pipeline.from_pretrained(
                str(pinned_weights("chronos2").parent), device_map=device, torch_dtype=dtype)
            torch.manual_seed(self.random_state)
            self.device_ = device
            self.dtype_ = dtype
        else:
            self.device_ = self.device or str(
                getattr(getattr(self.pipeline, "model", None), "device", "injected"))
            self.dtype_ = self.dtype or "float32"
        self.model_context_length_ = int(self.pipeline.model_context_length)
        q = np.asarray(self.pipeline.quantiles, dtype=np.float64)
        self.quantile_levels_ = q.tolist()
        bounds = np.concatenate([[0.0], q, [1.0]])
        mass = (bounds[2:] - bounds[:-2]) / 2.0
        self._q_mass = (mass / mass.sum()).astype(np.float32)
        self._median_idx = int(np.argmin(np.abs(q - 0.5)))


    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        detach = getattr(x, "detach", None)
        if detach is not None:
            x = detach().to("cpu")
        return np.asarray(x, dtype=np.float32)

    def _future_covariates(self, f: pd.DataFrame) -> tuple[np.ndarray, np.ndarray,
                                                           np.ndarray, np.ndarray]:
        weeks = f["week"].to_numpy(dtype=int)
        if weeks.min() <= self.last_train_week_:
            raise ValueError(
                f"Chronos2Model forecasts strictly after the training window "
                f"(last train week {self.last_train_week_}); frame contains "
                f"week {weeks.min()}")
        horizon = int(weeks.max() - self.last_train_week_)
        step = weeks - (self.last_train_week_ + 1)

        pairs = list(zip(f["product_id"].to_numpy(), f["store_id"].to_numpy()))
        local: dict[tuple, int] = {}
        row_series = np.full(len(f), -1, dtype=np.int64)
        positions: list[int] = []
        unseen: set[tuple] = set()
        for i, key in enumerate(pairs):
            slot = local.get(key)
            if slot is None:
                pos = self.series_pos_.get(key)
                if pos is None:
                    unseen.add(key)
                    continue
                slot = len(positions)
                local[key] = slot
                positions.append(pos)
            row_series[i] = slot
        if unseen:
            self.unseen_series_ = getattr(self, "unseen_series_", set()) | unseen
        pos_arr = np.asarray(positions, dtype=np.int64)

        seen_rows = row_series >= 0
        future = np.repeat(self.last_cov_[pos_arr][:, None, :], horizon, axis=1)
        future[row_series[seen_rows], step[seen_rows], :] = (
            self._covariate_matrix(f)[seen_rows]
        )
        return pos_arr, row_series, step, future

    def _predict_units_raw(self, f: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Chronos2Model must be fitted before prediction")
        pos_arr, row_series, step, future = self._future_covariates(f)
        horizon = future.shape[1]

        pin = not self._pinned
        keys: list[bytes] = []
        todo: list[int] = []
        for i, pos in enumerate(pos_arr):
            h = hashlib.blake2b(future[i].tobytes(), digest_size=16)
            h.update(str(pos).encode())
            key = h.digest()
            keys.append(key)
            if key not in self._pinned and key not in self._rolling:
                todo.append(i)
        self.cache_stats_["hits"] += len(pos_arr) - len(todo)
        self.cache_stats_["misses"] += len(todo)
        store = self._pinned if pin else self._rolling

        if todo:
            chunk = self.predict_chunk or len(todo)
            spans = range(0, len(todo), chunk)
            try:
                from tqdm.auto import tqdm
                spans = tqdm(spans, desc=f"Chronos-2 ({len(todo):,} series)",
                             unit="chunk", leave=False)
            except ImportError:
                pass
            for start in spans:
                sel = todo[start:start + chunk]
                inputs = [{
                    "target": self.targets_[pos_arr[i]],
                    "past_covariates": self.past_cov_[pos_arr[i]],
                    "future_covariates": {name: future[i, :, k]
                                          for k, name in enumerate(self.covariates)},
                } for i in sel]
                preds = self.pipeline.predict(
                    inputs, prediction_length=horizon, batch_size=self.batch_size,
                    context_length=self.context_length, limit_prediction_length=False,
                    cross_learning=self.cross_learning)
                for i, pred in zip(sel, preds):
                    q = self._to_numpy(pred)[0]
                    store[keys[i]] = np.stack([
                        (self._q_mass[:, None] * q).sum(axis=0),
                        q[self._median_idx],
                    ])

        slot = 0 if self.point == "qmean" else 1
        out = np.zeros(len(row_series), dtype=float)
        if len(keys):
            curve = np.stack([(self._pinned.get(k) if k in self._pinned
                               else self._rolling[k])[slot] for k in keys])
            seen_rows = row_series >= 0
            out[seen_rows] = curve[row_series[seen_rows], step[seen_rows]]
        if len(self._rolling) > self.max_cache:
            self._rolling.clear()
            self.cache_stats_["evictions"] += 1
        return out.astype(float)


class DoubleMLLogLog(BaseUnitsModel):
    name = "Log-log DoubleML"

    def __init__(self, fs: FeatureSpace, n_folds: int = 2, random_state: int = 42,
                 max_train: int | None = 400_000):
        super().__init__(fs)
        self.n_folds = n_folds
        self.random_state = random_state
        self.max_train = max_train
        self.theta = None

    def _controls(self, f: pd.DataFrame) -> np.ndarray:
        cols = ["promo_flag", "log_promo_cost", "seasonality", "log_household",
                "week_idx", "product_code", "store_code", "brand_code_id"]
        return f[cols].to_numpy(dtype=float)

    def _rival_supply(self, train: pd.DataFrame, f: pd.DataFrame,
                      supply: np.ndarray) -> np.ndarray:
        sw = pd.DataFrame({"store_id": train["store_id"].values,
                           "week": train["week"].values, "s": supply})
        g = sw.groupby(["store_id", "week"])["s"]
        return ((g.transform("sum") - sw["s"]) /
                (g.transform("count") - 1).replace(0, np.nan)).fillna(sw["s"]).to_numpy()

    def _learner(self):
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=63,
                             n_jobs=-1, random_state=self.random_state, verbose=-1)

    def fit(self, train: pd.DataFrame) -> "DoubleMLLogLog":
        from sklearn.model_selection import KFold

        train = train.reset_index(drop=True)
        f = self.fs.raw_features(train)

        supply = np.log(np.clip(train["supply_cost_proxy"].to_numpy(dtype=float), 1e-9, None))
        comp_supply = self._rival_supply(train, f, supply)
        if self.max_train is not None and len(train) > self.max_train:
            idx = train.sample(
                self.max_train, random_state=self.random_state
            ).index.to_numpy()
            train = train.iloc[idx].reset_index(drop=True)
            f = f.iloc[idx].reset_index(drop=True)
            supply = supply[idx]
            comp_supply = comp_supply[idx]

        Y = _log_units(train["units"].to_numpy())
        D = np.column_stack([f["logprice"].to_numpy(), f["comp_logprice"].to_numpy()])
        Z = np.column_stack([supply, comp_supply])
        X = self._controls(f)

        n = len(train)
        Y_res = np.zeros(n)
        D_res = np.zeros((n, 2))
        Z_res = np.zeros((n, 2))
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        for tr_idx, te_idx in kf.split(X):
            lY = self._learner().fit(X[tr_idx], Y[tr_idx])
            Y_res[te_idx] = Y[te_idx] - lY.predict(X[te_idx])
            for k in range(2):
                mD = self._learner().fit(X[tr_idx], D[tr_idx, k])
                D_res[te_idx, k] = D[te_idx, k] - mD.predict(X[te_idx])
                rZ = self._learner().fit(X[tr_idx], Z[tr_idx, k])
                Z_res[te_idx, k] = Z[te_idx, k] - rZ.predict(X[te_idx])

        ZtD = Z_res.T @ D_res
        ZtY = Z_res.T @ Y_res
        self.iv_condition_number = float(np.linalg.cond(ZtD))
        if not np.isfinite(self.iv_condition_number) or self.iv_condition_number > 1e10:
            raise np.linalg.LinAlgError(
                "DoubleML residual IV system is ill-conditioned "
                f"(condition number={self.iv_condition_number:.3g})"
            )
        self.theta = np.linalg.solve(ZtD, ZtY)
        self.first_stage_partial_r2 = []
        for k in range(D_res.shape[1]):
            coef = np.linalg.lstsq(Z_res, D_res[:, k], rcond=None)[0]
            fitted = Z_res @ coef
            denominator = float(np.sum((D_res[:, k] - D_res[:, k].mean()) ** 2))
            r2 = 1.0 - float(np.sum((D_res[:, k] - fitted) ** 2)) / denominator
            self.first_stage_partial_r2.append(r2 if denominator > 0 else np.nan)

        self.g_hat = self._learner().fit(X, Y)
        self.m_hat = [self._learner().fit(X, D[:, k]) for k in range(2)]
        self.own_elasticity = float(self.theta[0])
        self.cross_channel = float(self.theta[1])
        return self

    def _predict_units_raw(self, f: pd.DataFrame) -> np.ndarray:
        X = self._controls(f)
        D = np.column_stack([f["logprice"].to_numpy(), f["comp_logprice"].to_numpy()])
        m = np.column_stack([mk.predict(X) for mk in self.m_hat])
        log_q = self.g_hat.predict(X) + ((D - m) @ self.theta)
        return np.exp(log_q)


COST_INSTRUMENTS = ("supply_cost_proxy", "promo_cost")


def _loo_mean(store_id: np.ndarray, week: np.ndarray, values: np.ndarray) -> np.ndarray:
    sw = pd.DataFrame({"store_id": store_id, "week": week, "v": values})
    g = sw.groupby(["store_id", "week"])["v"]
    return ((g.transform("sum") - sw["v"]) /
            (g.transform("count") - 1).replace(0, np.nan)).fillna(sw["v"]).to_numpy()


class DoubleMLDepthIV(BaseUnitsModel):
    name = "Log-log DoubleML depth-IV"

    CONTROLS = ["promo_flag", "seasonality", "log_household",
                "week_idx", "product_code", "store_code", "brand_code_id"]
    TREATMENTS = ["log_regular", "depth", "comp_log_regular", "comp_depth"]
    DEPTH_CAP = 0.90
    RECENTRE = float(os.environ.get("CARD_PRICE_ENDING_CENT") or "0.01")
    MOVE_TOL = 1e-6

    def __init__(self, fs: FeatureSpace, n_folds: int = 2, random_state: int = 42,
                 max_train: int | None = 400_000,
                 instruments: tuple[str, str] = COST_INSTRUMENTS):
        super().__init__(fs)
        self.n_folds = n_folds
        self.random_state = random_state
        self.max_train = max_train
        self.instruments = tuple(instruments)
        self.theta = None
        self.route_counts = {"baseline": 0, "regular": 0, "depth": 0}

    def _learner(self):
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=63,
                             n_jobs=-1, random_state=self.random_state, verbose=-1,
                             force_row_wise=True, deterministic=True)

    def _controls(self, f: pd.DataFrame) -> np.ndarray:
        return f[self.CONTROLS].to_numpy(dtype=float)


    def _holdout_reference(self, grid: pd.DataFrame) -> pd.DataFrame:
        ctx = self.fs.cell.holdout_context.copy()
        ctx["product_id"] = ctx["product_id"].astype(str)
        ctx["store_id"] = ctx["store_id"].astype(str)
        ctx["week"] = ctx["week"].astype(int)
        last_reg = (grid.sort_values("week")
                    .groupby(["store_id", "product_id"])["regular_price"].last())
        off = pd.to_numeric(ctx["promo_flag"], errors="coerce").fillna(0) == 0
        ctx["regular_price"] = np.where(off, ctx["price"].astype(float) + self.RECENTRE, np.nan)
        ctx = ctx.sort_values(["store_id", "product_id", "week"]).reset_index(drop=True)
        grouped = ctx.groupby(["store_id", "product_id"], sort=False)["regular_price"]
        ctx["regular_price"] = grouped.ffill().fillna(grouped.bfill())
        ctx = ctx.join(last_reg.rename("last_regular"), on=["store_id", "product_id"])
        ctx["regular_price"] = ctx["regular_price"].fillna(ctx["last_regular"])
        prod_med = ctx.groupby("product_id")["regular_price"].transform("median")
        ctx["regular_price"] = ctx["regular_price"].fillna(prod_med)
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = 1.0 - (ctx["price"].astype(float).to_numpy() + self.RECENTRE) \
                / ctx["regular_price"].to_numpy()
        on = pd.to_numeric(ctx["promo_flag"], errors="coerce").fillna(0).to_numpy() == 1
        ctx["depth"] = np.where(on, np.clip(depth, 0.0, self.DEPTH_CAP), 0.0)
        out = ctx[["product_id", "store_id", "week", "regular_price", "depth"]].copy()
        out["base_price"] = ctx["price"].astype(float)
        return out

    def _build_reference(self, grid: pd.DataFrame) -> None:
        pres = grid.loc[grid["present"],
                        ["product_id", "store_id", "week", "price", "regular_price", "depth"]]
        train_ref = pres.rename(columns={"price": "base_price"})
        hold_ref = self._holdout_reference(grid)
        ref = pd.concat([train_ref[["product_id", "store_id", "week",
                                    "base_price", "regular_price", "depth"]],
                         hold_ref[["product_id", "store_id", "week",
                                   "base_price", "regular_price", "depth"]]],
                        ignore_index=True)
        self.base_ref = ref.rename(columns={"regular_price": "ref_regular",
                                            "depth": "ref_depth",
                                            "base_price": "ref_price"})
        self.prod_median_regular = grid.groupby("product_id")["regular_price"].median()

    def _route_channels(self, f: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        key = ["product_id", "store_id", "week"]
        work = f[key + ["price", "promo_flag"]].copy()
        work["week"] = work["week"].astype(int)
        work = work.merge(self.base_ref, on=key, how="left", validate="many_to_one")
        price = work["price"].astype(float).to_numpy()
        on = pd.to_numeric(work["promo_flag"], errors="coerce").fillna(0).to_numpy() == 1

        regular = work["ref_regular"].to_numpy(dtype=float)
        depth = work["ref_depth"].to_numpy(dtype=float)
        base_price = work["ref_price"].to_numpy(dtype=float)
        found = np.isfinite(regular) & np.isfinite(base_price)

        if (~found).any():
            fb_reg = work["product_id"].map(self.prod_median_regular).to_numpy(dtype=float)
            fb_reg = np.where(np.isfinite(fb_reg), fb_reg, price + self.RECENTRE)
            reg_fb = np.where(on, fb_reg, price + self.RECENTRE)
            with np.errstate(divide="ignore", invalid="ignore"):
                dep_fb = np.where(on, np.clip(1.0 - (price + self.RECENTRE) / reg_fb,
                                              0.0, self.DEPTH_CAP), 0.0)
            regular = np.where(found, regular, reg_fb)
            depth = np.where(found, depth, dep_fb)
            base_price = np.where(found, base_price, price)

        moved = found & (np.abs(price - base_price) > self.MOVE_TOL)
        if moved.any():
            if bool(on[moved].all()):
                self.route_counts["depth"] += 1
                dd = (base_price + self.RECENTRE - price) / regular
                depth = np.where(moved,
                                 np.clip(depth + dd, 0.0, self.DEPTH_CAP),
                                 depth)
            else:
                self.route_counts["regular"] += 1
                with np.errstate(divide="ignore", invalid="ignore"):
                    dlogp = np.log(price / base_price)
                dlogp = np.where(moved, np.nan_to_num(dlogp, nan=0.0,
                                                      posinf=0.0, neginf=0.0), 0.0)
                regular = regular * np.exp(dlogp)
        else:
            self.route_counts["baseline"] += 1
        log_regular = np.log(np.clip(regular, 0.01, None))
        return log_regular, depth

    def _treatments(self, f: pd.DataFrame) -> np.ndarray:
        log_regular, depth = self._route_channels(f)
        comp_lr = _loo_mean(f["store_id"].to_numpy(), f["week"].to_numpy(), log_regular)
        comp_dep = _loo_mean(f["store_id"].to_numpy(), f["week"].to_numpy(), depth)
        return np.column_stack([log_regular, depth, comp_lr, comp_dep])


    def fit(self, train: pd.DataFrame) -> "DoubleMLDepthIV":
        from sklearn.model_selection import KFold

        from baselines.grid_reconstruction import reconstruct_grid

        cell = self.fs.cell
        grid = reconstruct_grid(_public_for_grid(cell))
        self._build_reference(grid)

        train = train.reset_index(drop=True)
        f = self.fs.raw_features(train)
        key = ["product_id", "store_id", "week"]
        chan = train[key].copy()
        chan["week"] = chan["week"].astype(int)
        chan = chan.merge(
            grid.loc[grid["present"], key + ["regular_price", "depth"]],
            on=key, how="left", validate="one_to_one")
        reg = chan["regular_price"].astype(float)
        self.n_regular_fallback_rows = int(reg.isna().sum())
        reg = reg.fillna(train["price"].astype(float) + self.RECENTRE)
        log_regular = np.log(reg.clip(lower=0.01).to_numpy())
        depth = chan["depth"].fillna(0.0).to_numpy(dtype=float)
        del grid, chan

        store = train["store_id"].to_numpy()
        week = train["week"].to_numpy()
        comp_lr = _loo_mean(store, week, log_regular)
        comp_dep = _loo_mean(store, week, depth)
        D = np.column_stack([log_regular, depth, comp_lr, comp_dep])

        Z = _instrument_block(train, store, week, self.instruments)

        Y = _log_units(train["units"].to_numpy())
        X = self._controls(f)

        if self.max_train is not None and len(train) > self.max_train:
            idx = train.sample(self.max_train, random_state=self.random_state).index.to_numpy()
            Y, D, Z, X = Y[idx], D[idx], Z[idx], X[idx]

        n = len(Y)
        kD, kZ = D.shape[1], Z.shape[1]
        Y_res = np.zeros(n)
        D_res = np.zeros((n, kD))
        Z_res = np.zeros((n, kZ))
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        for tr_idx, te_idx in kf.split(X):
            lY = self._learner().fit(X[tr_idx], Y[tr_idx])
            Y_res[te_idx] = Y[te_idx] - lY.predict(X[te_idx])
            for k in range(kD):
                mD = self._learner().fit(X[tr_idx], D[tr_idx, k])
                D_res[te_idx, k] = D[te_idx, k] - mD.predict(X[te_idx])
            for k in range(kZ):
                rZ = self._learner().fit(X[tr_idx], Z[tr_idx, k])
                Z_res[te_idx, k] = Z[te_idx, k] - rZ.predict(X[te_idx])

        ZZ = Z_res.T @ Z_res
        self.iv_condition_number = float(np.linalg.cond(ZZ))
        if not np.isfinite(self.iv_condition_number) or self.iv_condition_number > 1e12:
            raise np.linalg.LinAlgError(
                "depth-IV instrument gram matrix is ill-conditioned "
                f"(condition number={self.iv_condition_number:.3g})")
        ZD = Z_res.T @ D_res
        ZY = Z_res.T @ Y_res
        A = ZD.T @ np.linalg.solve(ZZ, ZD)
        b = ZD.T @ np.linalg.solve(ZZ, ZY)
        self.iv_projected_condition_number = float(np.linalg.cond(A))
        if not np.isfinite(self.iv_projected_condition_number) \
                or self.iv_projected_condition_number > 1e10:
            raise np.linalg.LinAlgError(
                "depth-IV projected treatment system is ill-conditioned "
                f"(condition number={self.iv_projected_condition_number:.3g})")
        self.theta = np.linalg.solve(A, b)

        self.first_stage_partial_r2 = []
        for k in range(kD):
            coef = np.linalg.lstsq(Z_res, D_res[:, k], rcond=None)[0]
            fitted = Z_res @ coef
            denom = float(np.sum((D_res[:, k] - D_res[:, k].mean()) ** 2))
            r2 = 1.0 - float(np.sum((D_res[:, k] - fitted) ** 2)) / denom if denom > 0 else np.nan
            self.first_stage_partial_r2.append(r2)

        self.g_hat = self._learner().fit(X, Y)
        self.m_hat = [self._learner().fit(X, D[:, k]) for k in range(kD)]
        self.own_elasticity = float(self.theta[0])
        self.depth_response = float(self.theta[1])
        self.cross_channel = float(self.theta[2])
        self.cross_depth = float(self.theta[3])
        return self

    def _predict_units_raw(self, f: pd.DataFrame) -> np.ndarray:
        X = self._controls(f)
        D = self._treatments(f)
        m = np.column_stack([mk.predict(X) for mk in self.m_hat])
        log_q = self.g_hat.predict(X) + ((D - m) @ self.theta)
        return np.exp(np.clip(log_q, -20.0, 20.0))


class HierarchicalLogLog(BaseUnitsModel):
    name = "Hierarchical log-log"

    FIXED = ["logprice", "comp_logprice", "promo_flag", "log_promo_cost",
             "seasonality", "log_household", "week_idx"]

    def __init__(self, fs: FeatureSpace, max_train: int | None = 150_000, random_state: int = 42):
        super().__init__(fs)
        self.max_train = max_train
        self.random_state = random_state

    def fit(self, train: pd.DataFrame) -> "HierarchicalLogLog":
        import statsmodels.formula.api as smf

        train = train.reset_index(drop=True)
        f = self.fs.raw_features(train).reset_index(drop=True)
        f["log_units"] = _log_units(train["units"].to_numpy())
        if self.max_train is not None and len(train) > self.max_train:
            idx = train.sample(
                self.max_train, random_state=self.random_state
            ).index
            f = f.loc[idx].reset_index(drop=True)
        f["grp"] = f["product_code"].astype(int)
        formula = "log_units ~ " + " + ".join(self.FIXED)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            md = smf.mixedlm(formula, f, groups=f["grp"], re_formula="~logprice")
            self.result = md.fit(method="lbfgs", maxiter=200)
        self.fe = self.result.fe_params
        self.re = {int(k): v for k, v in self.result.random_effects.items()}
        self.own_elasticity = float(self.fe.get("logprice", np.nan))
        self.cross_channel = float(self.fe.get("comp_logprice", np.nan))
        return self

    def _predict_units_raw(self, f: pd.DataFrame) -> np.ndarray:
        fe = self.fe
        base = np.full(len(f), float(fe.get("Intercept", 0.0)))
        for col in self.FIXED:
            base = base + float(fe.get(col, 0.0)) * f[col].to_numpy(dtype=float)
        codes = f["product_code"].astype(int).to_numpy()
        logp = f["logprice"].to_numpy(dtype=float)
        re_int = np.zeros(len(f))
        re_slope = np.zeros(len(f))
        for i, c in enumerate(codes):
            eff = self.re.get(int(c))
            if eff is None:
                continue
            re_int[i] = float(eff.get("Group", 0.0))
            re_slope[i] = float(eff.get("logprice", 0.0))
        log_q = base + re_int + re_slope * logp
        return np.exp(log_q)


def _instrument_block(train: pd.DataFrame, store: np.ndarray, week: np.ndarray,
                      instruments: tuple[str, str] = COST_INSTRUMENTS) -> np.ndarray:
    reg_col, dep_col = instruments
    supply = np.log(np.clip(train[reg_col].to_numpy(dtype=float), 1e-9, None))
    pc = pd.to_numeric(train[dep_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    comp_supply = _loo_mean(store, week, supply)
    npc = np.column_stack([_loo_mean(store, week, pc ** p) for p in (1, 2, 3)])
    return np.column_stack([supply, pc, pc ** 2, pc ** 3, comp_supply, npc])


def _cost_instrument_block(train: pd.DataFrame, store: np.ndarray,
                           week: np.ndarray) -> np.ndarray:
    return _instrument_block(train, store, week, COST_INSTRUMENTS)


def _public_for_grid(cell) -> dict:
    tx = cell.train
    missing = [c for c in ("promo_cost", "supply_cost_proxy") if c not in tx.columns]
    if missing:
        tx = tx.copy()
        if "promo_cost" in missing:
            tx["promo_cost"] = 0.0
        if "supply_cost_proxy" in missing:
            tx["supply_cost_proxy"] = np.nan
    return {"transactions": tx, "products": cell.products, "stores": cell.stores}


class HierarchicalCFBase(BaseUnitsModel):
    EXOG = ["promo_flag", "seasonality", "log_household", "week_idx"]
    PRICE_COLS: list[str] = []

    def __init__(self, fs: FeatureSpace, max_train: int | None = 150_000,
                 random_state: int = 42):
        super().__init__(fs)
        self.max_train = max_train
        self.random_state = random_state

    def _price_matrix(self, f: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def _prepare_channels(self, train: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fit(self, train: pd.DataFrame) -> "HierarchicalCFBase":
        import statsmodels.formula.api as smf

        train = train.reset_index(drop=True)
        f = self.fs.raw_features(train)
        prices = self._prepare_channels(train, f)

        store = train["store_id"].to_numpy()
        week = train["week"].to_numpy()
        Z = _cost_instrument_block(train, store, week)

        frame = f[self.EXOG].copy().reset_index(drop=True)
        for c in self.PRICE_COLS:
            frame[c] = prices[c].to_numpy(dtype=float)
        frame["log_units"] = _log_units(train["units"].to_numpy())
        frame["grp"] = f["product_code"].astype(int).to_numpy()

        prod_dum = pd.get_dummies(
            pd.Categorical(f["product_code"].astype(int),
                           categories=list(range(len(self.fs.product_vocab)))),
            prefix="fsprod", drop_first=True).astype(float).to_numpy()

        if self.max_train is not None and len(train) > self.max_train:
            idx = train.sample(self.max_train, random_state=self.random_state
                               ).index.to_numpy()
            frame = frame.iloc[idx].reset_index(drop=True)
            Z = Z[idx]
            prod_dum = prod_dum[idx]

        self.exog_used = [c for c in self.EXOG
                          if float(np.ptp(frame[c].to_numpy(dtype=float))) > 0]
        exog_fs = np.column_stack([
            np.ones(len(frame)),
            frame[self.exog_used].to_numpy(dtype=float),
            prod_dum,
        ])
        design = np.column_stack([Z, exog_fs])
        self.cf_cols = [f"cf_{c}" for c in self.PRICE_COLS]
        self.first_stage_r2 = {}
        for c, cf_col in zip(self.PRICE_COLS, self.cf_cols):
            y = frame[c].to_numpy(dtype=float)
            coef = np.linalg.lstsq(design, y, rcond=None)[0]
            resid = y - design @ coef
            frame[cf_col] = resid
            denom = float(np.sum((y - y.mean()) ** 2))
            self.first_stage_r2[c] = (
                1.0 - float(resid @ resid) / denom if denom > 0 else np.nan)

        formula = "log_units ~ " + " + ".join(
            self.PRICE_COLS + self.exog_used + self.cf_cols)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            md = smf.mixedlm(formula, frame, groups=frame["grp"],
                             re_formula=f"~{self.PRICE_COLS[0]}")
            self.result = md.fit(method="lbfgs", maxiter=200)
        self.fe = self.result.fe_params
        self.re = {int(k): v for k, v in self.result.random_effects.items()}
        self.own_elasticity = float(self.fe.get(self.PRICE_COLS[0], np.nan))
        self.cf_coefs = {c: float(self.fe.get(c, np.nan)) for c in self.cf_cols}
        return self

    def _predict_units_raw(self, f: pd.DataFrame) -> np.ndarray:
        P = self._price_matrix(f)
        fe = self.fe
        base = np.full(len(f), float(fe.get("Intercept", 0.0)))
        for i, col in enumerate(self.PRICE_COLS):
            base = base + float(fe.get(col, 0.0)) * P[:, i]
        for col in self.EXOG:
            base = base + float(fe.get(col, 0.0)) * f[col].to_numpy(dtype=float)
        codes = f["product_code"].astype(int).to_numpy()
        re_int = np.zeros(len(f))
        re_slope = np.zeros(len(f))
        for i, c in enumerate(codes):
            eff = self.re.get(int(c))
            if eff is None:
                continue
            re_int[i] = float(eff.get("Group", 0.0))
            re_slope[i] = float(eff.get(self.PRICE_COLS[0], 0.0))
        log_q = base + re_int + re_slope * P[:, 0]
        return np.exp(np.clip(log_q, -20.0, 20.0))


class HierarchicalBlendedCF(HierarchicalCFBase):
    name = "Hierarchical log-log blended-CF"

    PRICE_COLS = ["logprice", "comp_logprice"]

    def _prepare_channels(self, train: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
        return f[self.PRICE_COLS].reset_index(drop=True)

    def _price_matrix(self, f: pd.DataFrame) -> np.ndarray:
        return f[self.PRICE_COLS].to_numpy(dtype=float)


class HierarchicalDepthCF(DoubleMLDepthIV):
    name = "Hierarchical log-log depth-CF"

    FIXED = ["log_regular", "depth", "comp_log_regular", "comp_depth",
             "promo_flag", "seasonality", "log_household", "week_idx"]
    CF_COLS = ["cf_log_regular", "cf_depth", "cf_comp_log_regular", "cf_comp_depth"]

    def __init__(self, fs: FeatureSpace, max_train: int | None = 150_000,
                 random_state: int = 42,
                 instruments: tuple[str, str] = COST_INSTRUMENTS):
        super().__init__(fs, random_state=random_state, max_train=max_train,
                         instruments=instruments)

    def fit(self, train: pd.DataFrame) -> "HierarchicalDepthCF":
        import statsmodels.formula.api as smf

        from baselines.grid_reconstruction import reconstruct_grid

        cell = self.fs.cell
        grid = reconstruct_grid(_public_for_grid(cell))
        self._build_reference(grid)

        train = train.reset_index(drop=True)
        f = self.fs.raw_features(train)
        key = ["product_id", "store_id", "week"]
        chan = train[key].copy()
        chan["week"] = chan["week"].astype(int)
        chan = chan.merge(
            grid.loc[grid["present"], key + ["regular_price", "depth"]],
            on=key, how="left", validate="one_to_one")
        reg = chan["regular_price"].astype(float)
        self.n_regular_fallback_rows = int(reg.isna().sum())
        reg = reg.fillna(train["price"].astype(float) + self.RECENTRE)
        log_regular = np.log(reg.clip(lower=0.01).to_numpy())
        depth = chan["depth"].fillna(0.0).to_numpy(dtype=float)
        del grid, chan

        store = train["store_id"].to_numpy()
        week = train["week"].to_numpy()
        comp_lr = _loo_mean(store, week, log_regular)
        comp_dep = _loo_mean(store, week, depth)

        Z = _instrument_block(train, store, week, self.instruments)

        frame = f[["promo_flag", "seasonality", "log_household", "week_idx"]].copy()
        frame["log_regular"] = log_regular
        frame["depth"] = depth
        frame["comp_log_regular"] = comp_lr
        frame["comp_depth"] = comp_dep
        frame["log_units"] = _log_units(train["units"].to_numpy())
        frame["grp"] = f["product_code"].astype(int).to_numpy()

        prod_dum = pd.get_dummies(
            pd.Categorical(f["product_code"].astype(int),
                           categories=list(range(len(self.fs.product_vocab)))),
            prefix="fsprod", drop_first=True).astype(float).to_numpy()

        if self.max_train is not None and len(train) > self.max_train:
            idx = train.sample(self.max_train, random_state=self.random_state
                               ).index.to_numpy()
            frame = frame.iloc[idx].reset_index(drop=True)
            Z = Z[idx]
            prod_dum = prod_dum[idx]

        self.exog_used = [c for c in self.FIXED[4:]
                          if float(np.ptp(frame[c].to_numpy(dtype=float))) > 0]
        exog_fs = np.column_stack([
            np.ones(len(frame)),
            frame[self.exog_used].to_numpy(dtype=float),
            prod_dum,
        ])
        design = np.column_stack([Z, exog_fs])
        self.first_stage_r2 = {}
        for endog, cf_col in zip(self.FIXED[:4], self.CF_COLS):
            y = frame[endog].to_numpy(dtype=float)
            coef = np.linalg.lstsq(design, y, rcond=None)[0]
            resid = y - design @ coef
            frame[cf_col] = resid
            denom = float(np.sum((y - y.mean()) ** 2))
            self.first_stage_r2[endog] = (
                1.0 - float(resid @ resid) / denom if denom > 0 else np.nan)

        formula = "log_units ~ " + " + ".join(
            self.FIXED[:4] + self.exog_used + self.CF_COLS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            md = smf.mixedlm(formula, frame, groups=frame["grp"],
                             re_formula="~log_regular")
            self.result = md.fit(method="lbfgs", maxiter=200)
        self.fe = self.result.fe_params
        self.re = {int(k): v for k, v in self.result.random_effects.items()}
        self.own_elasticity = float(self.fe.get("log_regular", np.nan))
        self.depth_response = float(self.fe.get("depth", np.nan))
        self.cross_channel = float(self.fe.get("comp_log_regular", np.nan))
        self.cross_depth = float(self.fe.get("comp_depth", np.nan))
        self.cf_coefs = {c: float(self.fe.get(c, np.nan)) for c in self.CF_COLS}
        return self

    def _predict_units_raw(self, f: pd.DataFrame) -> np.ndarray:
        D = self._treatments(f)
        fe = self.fe
        base = np.full(len(f), float(fe.get("Intercept", 0.0)))
        for i, col in enumerate(self.FIXED[:4]):
            base = base + float(fe.get(col, 0.0)) * D[:, i]
        for col in self.FIXED[4:]:
            base = base + float(fe.get(col, 0.0)) * f[col].to_numpy(dtype=float)
        codes = f["product_code"].astype(int).to_numpy()
        re_int = np.zeros(len(f))
        re_slope = np.zeros(len(f))
        for i, c in enumerate(codes):
            eff = self.re.get(int(c))
            if eff is None:
                continue
            re_int[i] = float(eff.get("Group", 0.0))
            re_slope[i] = float(eff.get("log_regular", 0.0))
        log_q = base + re_int + re_slope * D[:, 0]
        return np.exp(np.clip(log_q, -20.0, 20.0))


class BLPLogitModel(BaseUnitsModel):
    name = "Homogeneous Berry logit"

    ENDOG = ["logprice", "comp_logprice"]
    EXOG = ["promo_flag", "log_promo_cost", "seasonality",
            "log_household", "week_idx"]

    def __init__(self, fs: FeatureSpace, market_potential_mult: float = 3.0,
                 random_state: int = 42):
        super().__init__(fs)
        self.mp_mult = market_potential_mult
        self.random_state = random_state

    def _market_size_ref(self, cell_train: pd.DataFrame) -> dict:
        vol = (cell_train.groupby(["store_id", "week"])["units"].sum()
               .groupby("store_id").mean())
        return {str(k): self.mp_mult * float(v) for k, v in vol.items()}

    def _rival_supply(self, train: pd.DataFrame, f: pd.DataFrame,
                      supply: np.ndarray) -> np.ndarray:
        sw = pd.DataFrame({
            "store_id": train["store_id"].values,
            "week": train["week"].values,
            "s": supply,
        })
        group = sw.groupby(["store_id", "week"])["s"]
        return (
            (group.transform("sum") - sw["s"])
            / (group.transform("count") - 1).replace(0, np.nan)
        ).fillna(sw["s"]).to_numpy()

    def fit(self, train: pd.DataFrame) -> "BLPLogitModel":
        from linearmodels.iv import IV2SLS

        train = train.reset_index(drop=True)
        f = self.fs.raw_features(train)
        sw_units = train.groupby(["store_id", "week"])["units"].transform("sum")
        M = self.mp_mult * sw_units.to_numpy(dtype=float)
        s_j = train["units"].to_numpy(dtype=float) / M
        s_0 = 1.0 - 1.0 / self.mp_mult
        delta = np.log(np.clip(s_j, 1e-12, None)) - np.log(s_0)

        prod_dum = pd.get_dummies(
            pd.Categorical(f["product_code"].astype(int),
                           categories=list(range(len(self.fs.product_vocab)))),
            prefix="prod", drop_first=True).astype(float).reset_index(drop=True)

        exog_cols = list(self.EXOG)
        exog = pd.concat([pd.Series(1.0, index=f.index, name="const"),
                          f[exog_cols].reset_index(drop=True), prod_dum], axis=1)
        endog = f[self.ENDOG].reset_index(drop=True)
        design = pd.concat([endog, exog], axis=1)
        design = _drop_collinear(
            design, protect=[*self.ENDOG, "const"]
        )
        endog = design[self.ENDOG]
        exog = design[[c for c in design if c not in self.ENDOG]]

        supply = np.log(np.clip(train["supply_cost_proxy"].to_numpy(dtype=float), 1e-9, None))
        comp_supply = self._rival_supply(train, f, supply)
        instr = pd.DataFrame({
            "z_supply": supply,
            "z_comp_supply": comp_supply,
        })

        dep = pd.Series(delta, name="delta")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = IV2SLS(dep, exog, endog, instr).fit(cov_type="robust")
        self.params = res.params
        self.iv_result = res
        self.exog_cols = exog_cols
        self.prod_dum_cols = list(prod_dum.columns)
        self.alpha = float(self.params.get("logprice", np.nan))
        self.cross_channel = float(self.params.get("comp_logprice", np.nan))
        self.market_ref = self._market_size_ref(train)
        self.store_week_ref = (train.groupby(["store_id", "week"])["units"].sum())
        return self

    def _delta(self, f: pd.DataFrame) -> np.ndarray:
        p = self.params
        d = np.full(len(f), float(p.get("const", 0.0)))
        d = d + float(p.get("logprice", 0.0)) * f["logprice"].to_numpy(dtype=float)
        d = d + float(p.get("comp_logprice", 0.0)) * \
            f["comp_logprice"].to_numpy(dtype=float)
        for c in self.exog_cols:
            d = d + float(p.get(c, 0.0)) * f[c].to_numpy(dtype=float)
        codes = f["product_code"].astype(int).to_numpy()
        for i, c in enumerate(codes):
            col = f"prod_{c}"
            if col in p.index:
                d[i] += float(p[col])
        return d

    def _predict_units_raw(self, f: pd.DataFrame) -> np.ndarray:
        delta = self._delta(f)
        out = np.zeros(len(f), dtype=float)
        idx = np.arange(len(f))
        store = f["store_id"].to_numpy()
        week = f["week"].to_numpy()
        df = pd.DataFrame({"i": idx, "store_id": store, "week": week, "delta": delta})
        for (st, wk), grp in df.groupby(["store_id", "week"], sort=False):
            ex = np.exp(np.clip(grp["delta"].to_numpy(), -50, 50))
            denom = 1.0 + ex.sum()
            shares = ex / denom
            M = self.market_ref.get(str(st))
            if M is None:
                M = self.mp_mult * float(self.store_week_ref.mean())
            out[grp["i"].to_numpy()] = shares * M
        return out
