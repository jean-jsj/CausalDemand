"""Log-log family reference grid: 2x2 (instrument x text) Poisson estimators.

All four corners share one form — a per-product Poisson (PPML) count model
with store + week fixed effects, an own log-price term, and a weighted
neighbor log-price index. The corners differ only along the two axes: IV
corners add a control-function residual (first stage: log price on
``supply_cost_proxy``); text corners weight neighbors by TF-IDF text
distances instead of brand membership. Stage 2 freezes the price response
and fits per-(product, store) levels plus seasonal harmonics that, unlike
week fixed effects, extrapolate to holdout weeks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from causaldemand.baselines.text_distance import (
    brand_distance_matrix,
    kernel_weights,
    rank_normalize,
    text_distance_matrix,
)

SEASON_PERIOD_WEEKS = 52
NEIGHBOR_K = 5
KERNEL_BANDWIDTH = 0.5
MIN_OFFSET_UNITS = 0.05


def _season_features(week: pd.Series) -> pd.DataFrame:
    phase = 2.0 * np.pi * (week.astype(float) % SEASON_PERIOD_WEEKS) / SEASON_PERIOD_WEEKS
    return pd.DataFrame(
        {
            "season_sin1": np.sin(phase),
            "season_cos1": np.cos(phase),
            "season_sin2": np.sin(2 * phase),
            "season_cos2": np.cos(2 * phase),
        },
        index=week.index,
    )


def _neighbor_weights(products: pd.DataFrame, use_text: bool) -> pd.DataFrame:
    # Text distances are rank-normalized (order is the recoverable signal;
    # raw TF-IDF cosine on the templated prose compresses near 1). The brand
    # matrix is already on a 0/1 scale.
    if use_text:
        distances = rank_normalize(text_distance_matrix(products))
    else:
        distances = brand_distance_matrix(products)
    return kernel_weights(distances, bandwidth=KERNEL_BANDWIDTH, k=NEIGHBOR_K)


def _log_price_wide(frame: pd.DataFrame) -> pd.DataFrame:
    prices = frame[["product_id", "store_id", "week", "price"]].copy()
    prices["log_price"] = np.log(prices["price"].astype(float).clip(lower=0.01))
    return prices.pivot_table(
        index=["store_id", "week"], columns="product_id", values="log_price", aggfunc="mean"
    )


def _neighbor_index(frame: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """Weighted neighbor log-price index per row of ``frame``."""
    wide = _log_price_wide(frame)
    wide = wide.reindex(columns=weights.columns)
    fill = wide.mean(axis=0)
    wide = wide.fillna(fill).fillna(0.0)
    npi_wide = wide.to_numpy() @ weights.to_numpy().T
    npi = pd.DataFrame(npi_wide, index=wide.index, columns=weights.index)
    row_index = pd.MultiIndex.from_frame(frame[["store_id", "week"]])
    out = np.empty(len(frame), dtype=float)
    for product_id, block in frame.groupby("product_id").groups.items():
        out[frame.index.get_indexer(block)] = row_index[frame.index.get_indexer(block)].map(
            npi[product_id]
        )
    return pd.Series(out, index=frame.index)


MAX_STAGE1_ROWS = 60_000


def _fit_price_response(sub: pd.DataFrame, use_iv: bool) -> dict:
    """Stage 1: beta/gamma via Poisson with store + week fixed effects.

    No promotion control — in this DGP promotions move demand through the
    price cut itself, so a promo regressor absorbs genuine price variation and
    attenuates beta. IV corners add a control-function residual whose first
    stage carries the SAME fixed effects.
    """
    work = sub
    if len(work) > MAX_STAGE1_ROWS:
        work = work.sample(MAX_STAGE1_ROWS, random_state=37)
    fixed = pd.get_dummies(work[["store_id", "week"]].astype(str), drop_first=True, dtype=float)
    x = pd.concat([work[["log_price", "npi"]].astype(float), fixed], axis=1)
    if use_iv:
        first_x = pd.concat([work[["supply_cost_proxy"]].astype(float), fixed], axis=1)
        first = sm.OLS(
            work["log_price"].astype(float), sm.add_constant(first_x, has_constant="add")
        ).fit()
        x["cf_residual"] = first.resid
    model = sm.GLM(
        work["units"].astype(float),
        sm.add_constant(x, has_constant="add"),
        family=sm.families.Poisson(),
    ).fit(maxiter=80, disp=0)
    return {
        "beta": float(model.params.get("log_price", 0.0)),
        "gamma": float(model.params.get("npi", 0.0)),
        "converged": bool(getattr(model, "converged", False)),
    }


def _fit_forecast_calibration(sub: pd.DataFrame, beta: float, gamma: float) -> dict:
    """Stage 2: seasonality/level around the FIXED stage-1 price response.

    The price and neighbor terms enter as an offset (centered within
    product-store so the per-store level is carried once by the store mean),
    leaving only the intercept and week-of-year harmonics to estimate — these
    extrapolate to holdout weeks, unlike stage 1's week fixed effects.
    """
    offset = (
        sub["offset"].to_numpy(dtype=float)
        + beta * (sub["log_price"] - sub["p_bar"]).to_numpy(dtype=float)
        + gamma * (sub["npi"] - sub["npi_bar"]).to_numpy(dtype=float)
    )
    x = _season_features(sub["week"])
    season_cols = list(x.columns)
    try:
        model = sm.GLM(
            sub["units"].astype(float),
            sm.add_constant(x, has_constant="add"),
            family=sm.families.Poisson(),
            offset=offset,
        ).fit(maxiter=80, disp=0)
        return {
            "const": float(model.params.get("const", 0.0)),
            **{c: float(model.params.get(c, 0.0)) for c in season_cols},
        }
    except Exception:
        return {"const": 0.0, **{c: 0.0 for c in season_cols}}


def fit_loglog_corner(
    transactions: pd.DataFrame,
    products: pd.DataFrame,
    *,
    use_iv: bool,
    use_text: bool,
) -> dict:
    """Fit one 2x2 corner on the public training transactions.

    Two stages per product: the price response (beta, gamma) from a store+week
    fixed-effects Poisson on carried stores, then a forecasting calibration
    (level + seasonal harmonics) around those fixed coefficients.
    """
    weights = _neighbor_weights(products, use_text=use_text)
    train = transactions.copy()
    train["log_price"] = np.log(train["price"].astype(float).clip(lower=0.01))
    train["npi"] = _neighbor_index(train, weights)

    offsets = (
        train.groupby(["product_id", "store_id"])["units"]
        .mean()
        .clip(lower=MIN_OFFSET_UNITS)
        .apply(np.log)
        .rename("offset")
    )
    store_means = (
        train.groupby(["product_id", "store_id"])[["log_price", "npi"]]
        .mean()
        .rename(columns={"log_price": "p_bar", "npi": "npi_bar"})
    )
    train = train.join(offsets, on=["product_id", "store_id"])
    train = train.join(store_means, on=["product_id", "store_id"])

    coefs: dict[str, dict[str, float]] = {}
    for product_id, sub in train.groupby("product_id"):
        sub = sub.copy()
        carried = sub.groupby("store_id")["units"].transform("sum") > 0
        carried_sub = sub[carried]
        if carried_sub["units"].sum() <= 0 or len(carried_sub) < 50:
            coefs[product_id] = {
                "beta": 0.0, "gamma": 0.0, "const": 0.0, "converged": False,
                "season_sin1": 0.0, "season_cos1": 0.0, "season_sin2": 0.0, "season_cos2": 0.0,
            }
            continue
        try:
            response = _fit_price_response(carried_sub, use_iv)
        except Exception:
            response = {"beta": 0.0, "gamma": 0.0, "converged": False}
        calibration = _fit_forecast_calibration(sub, response["beta"], response["gamma"])
        coefs[product_id] = {**response, **calibration}

    return {
        "family": "log_log",
        "use_iv": use_iv,
        "use_text": use_text,
        "coefs": coefs,
        "weights": weights,
        "offsets": offsets,
        "store_means": store_means,
    }


def _predict_mean(params: dict, frame: pd.DataFrame) -> np.ndarray:
    """Model mean for rows carrying product_id/store_id/week/price."""
    frame = frame.copy()
    frame["log_price"] = np.log(frame["price"].astype(float).clip(lower=0.01))
    frame["npi"] = _neighbor_index(frame, params["weights"])
    frame = frame.join(params["offsets"], on=["product_id", "store_id"])
    frame = frame.join(params["store_means"], on=["product_id", "store_id"])
    frame["offset"] = frame["offset"].fillna(float(params["offsets"].median()))
    frame["p_bar"] = frame["p_bar"].fillna(frame["log_price"])
    frame["npi_bar"] = frame["npi_bar"].fillna(frame["npi"])
    season = _season_features(frame["week"])
    out = np.zeros(len(frame), dtype=float)
    for product_id, block_idx in frame.groupby("product_id").groups.items():
        c = params["coefs"].get(product_id)
        if c is None:
            continue
        block = frame.loc[block_idx]
        linear = (
            c["const"]
            + block["offset"].to_numpy(dtype=float)
            + c["beta"] * (block["log_price"] - block["p_bar"]).to_numpy(dtype=float)
            + c["gamma"] * (block["npi"] - block["npi_bar"]).to_numpy(dtype=float)
            + season.loc[block_idx].to_numpy() @ np.array(
                [c["season_sin1"], c["season_cos1"], c["season_sin2"], c["season_cos2"]]
            )
        )
        out[frame.index.get_indexer(block_idx)] = np.exp(np.clip(linear, -30.0, 30.0))
    return out


def predict_holdout_units(params: dict, holdout_context: pd.DataFrame) -> pd.DataFrame:
    """forecast_predictions frame: predicted units on the holdout context."""
    predicted = _predict_mean(params, holdout_context)
    return pd.DataFrame(
        {
            "product_id": holdout_context["product_id"].to_numpy(),
            "store_id": holdout_context["store_id"].to_numpy(),
            "week": holdout_context["week"].to_numpy(),
            "predicted_units": predicted,
        }
    )


def elasticity_matrix(params: dict, products: pd.DataFrame) -> pd.DataFrame:
    """elasticity_matrix frame: full JxJ elasticities.

    Own: ``beta_j``. Cross: ``gamma_j * w_jk`` (the neighbor index passes
    ``w_jk`` of competitor k's log price into product j's demand).
    """
    weights = params["weights"]
    rows = []
    for affected in weights.index:
        c = params["coefs"].get(affected, {"beta": 0.0, "gamma": 0.0})
        for priced in weights.columns:
            if priced == affected:
                value = c["beta"]
            else:
                value = c["gamma"] * float(weights.loc[affected, priced])
            rows.append(
                {
                    "affected_product_id": affected,
                    "priced_product_id": priced,
                    "elasticity": float(value),
                }
            )
    return pd.DataFrame(rows)


def predict_sweep_deltas(
    params: dict,
    sweep_context: pd.DataFrame,
    holdout_context: pd.DataFrame,
) -> pd.DataFrame:
    """counterfactual_deltas frame: predicted delta units per sweep row.

    Delta = mean(baseline) * (exp(beta*dlogp_own + gamma*d_npi) - 1), computed
    per intervention. Promo flags for the sweep weeks come from the holdout
    context (same weeks, same store-product grid).
    """
    promo = holdout_context[["product_id", "store_id", "week", "promo_flag"]]
    frames = []
    for intervention_id, ctx in sweep_context.groupby("intervention_id"):
        base = ctx.rename(columns={"baseline_price": "price"})[
            ["product_id", "store_id", "week", "price"]
        ].merge(promo, on=["product_id", "store_id", "week"], how="left")
        base["promo_flag"] = base["promo_flag"].fillna(0)
        cf = ctx.rename(columns={"intervention_price": "price"})[
            ["product_id", "store_id", "week", "price"]
        ].merge(promo, on=["product_id", "store_id", "week"], how="left")
        cf["promo_flag"] = cf["promo_flag"].fillna(0)
        base_units = _predict_mean(params, base)
        cf_units = _predict_mean(params, cf)
        frames.append(
            pd.DataFrame(
                {
                    "intervention_id": intervention_id,
                    "product_id": ctx["product_id"].to_numpy(),
                    "store_id": ctx["store_id"].to_numpy(),
                    "week": ctx["week"].to_numpy(),
                    "predicted_delta_units": cf_units - base_units,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
