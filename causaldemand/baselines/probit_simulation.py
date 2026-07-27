"""Simulation-based reference estimator for the discrete-choice family.

Rebuilds the covariance-probit mechanics from public data and answers every
scored task by re-simulating the choice process: (1) error covariance from a
Gaussian kernel on rank-normalized text distances (identity when text-blind);
(2) per-store mean utilities by simulated share inversion; (3) a price-utility
scale calibrated to the reduced-form share regression (OLS naive, IV
instrumented); (4) counterfactuals by re-simulating choices with common random
numbers, scaled by the incidence margin ``exp(rho * dCV)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from causaldemand.baselines.probit_shares import (
    OMEGA_MIX,
    RANK_BANDWIDTH,
    SEASON_PERIOD_WEEKS,
    fit_pooled_clipped,
    fit_share_price_sensitivity,
)
from causaldemand.baselines.text_distance import rank_normalize, text_distance_matrix

N_DRAWS = 4000
INVERSION_ITERS = 250
INVERSION_STEP = 0.4
INVERSION_TOL = 2e-3
MIN_SHARE = 1e-4
RNG_SEED = 1913


def _build_kernel(products: pd.DataFrame, use_text: bool) -> np.ndarray | None:
    if not use_text:
        return None
    ranked = rank_normalize(text_distance_matrix(products)).to_numpy()
    return np.exp(-(ranked**2) / (2.0 * RANK_BANDWIDTH**2))


def _build_sigma_chol(kernel: np.ndarray | None, j: int) -> np.ndarray:
    if kernel is None:
        return np.eye(j)
    sigma = OMEGA_MIX * kernel + (1.0 - OMEGA_MIX) * np.eye(j)
    # Public-side PD repair (the estimator clips; the generator fails shut).
    eigval, eigvec = np.linalg.eigh(sigma)
    sigma = (eigvec * np.clip(eigval, 1e-6, None)) @ eigvec.T
    return np.linalg.cholesky(sigma + 1e-9 * np.eye(j))


def _allocate_cross(
    s_base: np.ndarray, s_focal_cf: np.ndarray, focal_idx: int, kernel: np.ndarray | None
) -> np.ndarray:
    """Counterfactual shares: simulated focal response + EXPECTED diversion.

    Realized flipped-draw destinations put only a handful of Monte-Carlo draws
    on each small competitor, so their cross pattern would be noise at any
    feasible draw count. Instead the displaced focal mass is allocated by its
    expectation under the model: proportional to ``s_k (omega K_jk + 1-omega)``
    — competitors weighted by size and error-correlation closeness.
    """
    out = s_base.copy()
    out[:, focal_idx] = s_focal_cf
    displaced = s_base[:, focal_idx] - s_focal_cf
    if kernel is None:
        weights = s_base.copy()
    else:
        weights = s_base * (OMEGA_MIX * kernel[focal_idx][None, :] + (1.0 - OMEGA_MIX))
    weights[:, focal_idx] = 0.0
    totals = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, totals, out=np.zeros_like(weights), where=totals > 0)
    out += displaced[:, None] * weights
    return out


def _simulate_shares(delta: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Average argmax shares for utility ``delta[j] + eps[n, j]``.

    ``delta`` may be (J,) or (B, J) for a batch; returns matching shape.
    """
    if delta.ndim == 1:
        winners = np.argmax(delta[None, :] + eps, axis=1)
        return np.bincount(winners, minlength=delta.shape[0]) / eps.shape[0]
    batch, j = delta.shape
    out = np.empty((batch, j))
    for i in range(batch):
        winners = np.argmax(delta[i][None, :] + eps, axis=1)
        out[i] = np.bincount(winners, minlength=j) / eps.shape[0]
    return out


def _invert_shares(target: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Damped fixed-point delta such that simulated shares match ``target``.

    Targets are floored at the Monte-Carlo resolution (1 / draws): shares
    below it are not resolvable and would drag their delta to -inf. The
    log-share step is damped — the argmax-probit share map is stiffer than
    the logit map the undamped BLP contraction assumes — and convergence is
    tracked on the resolvable entries.
    """
    floor = max(MIN_SHARE, 1.0 / eps.shape[0])
    target = np.clip(target, floor, None)
    target = target / target.sum(axis=1, keepdims=True)
    delta = np.log(target)
    for _ in range(INVERSION_ITERS):
        sim = np.clip(_simulate_shares(delta, eps), floor / 10, None)
        step = np.log(target) - np.log(sim)
        delta = delta + INVERSION_STEP * step
        delta = delta - delta.mean(axis=1, keepdims=True)
        if np.max(np.abs(step)) < INVERSION_TOL:
            break
    return delta


def _calibrate_price_scale(
    delta_stores: np.ndarray,
    shares_stores: np.ndarray,
    mean_prices: np.ndarray,
    eps: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Utility-per-price scale matching per-product d log(share)/d price.

    ``targets[j]`` is the reduced-form semi-elasticity for product j (the
    per-product estimate where identified, the pooled slope otherwise) and
    ``weights[j]`` its category share. Matching per product matters: the
    scored flagship is the share-highest product, whose own log-share
    sensitivity is far below the pooled average — a pooled target would force
    the utility scale up until the flagship over-responds. One scale is
    chosen to zero the share-weighted gap between model and target
    sensitivities (secant; the response is near-linear in the scale).
    """
    dp = 0.05 * float(np.nanmean(mean_prices))
    base_sim = np.clip(_simulate_shares(delta_stores, eps), MIN_SHARE, None)

    def weighted_gap(a: float) -> float:
        gap = 0.0
        for j in range(delta_stores.shape[1]):
            if weights[j] <= 0:
                continue
            bumped = delta_stores.copy()
            bumped[:, j] += a * dp
            sim = np.clip(_simulate_shares(bumped, eps), MIN_SHARE, None)
            model_e = float(np.mean(np.log(sim[:, j]) - np.log(base_sim[:, j])) / dp)
            gap += weights[j] * (model_e - targets[j])
        return gap / max(weights.sum(), 1e-9)

    a1 = float(np.average(targets, weights=np.clip(weights, 1e-9, None)))
    g1 = weighted_gap(a1)
    a2 = a1 * 0.5 if abs(g1) > 1e-9 else a1
    g2 = weighted_gap(a2)
    if abs(g2 - g1) < 1e-9:
        return a2
    a3 = float(a2 - g2 * (a2 - a1) / (g2 - g1))
    # One more secant refinement, bounded away from sign flips.
    a3 = float(np.clip(a3, 4.0 * min(a1, 0.0), 0.0))
    g3 = weighted_gap(a3)
    if abs(g3 - g2) > 1e-9:
        a4 = float(a3 - g3 * (a3 - a2) / (g3 - g2))
        return float(np.clip(a4, 4.0 * min(a1, 0.0), 0.0))
    return a3


def _calibrate_incidence_rho(transactions: pd.DataFrame, price_scale: float) -> float:
    """Incidence pass-through from category volume's price-index response.

    The category-value regression attenuates badly (its regressor is itself
    estimated), so ``rho`` is backed out of a directly estimable reduced form:
    ``d log M / d p_bar = rho * price_scale`` where ``p_bar`` is the
    share-weighted category price index. The observational slope is used for
    every corner — the incidence margin is a separate channel from own-price
    identification, and the observational slope on the category price
    index is not confounded the way a single product's price is.
    """
    work = transactions[["store_id", "week", "product_id", "units", "price"]].copy()
    weights = work.groupby("product_id")["units"].sum()
    weights = (weights / max(float(weights.sum()), 1.0)).rename("w")
    work = work.join(weights, on="product_id")
    work["wp"] = work["w"] * work["price"].astype(float)
    grouped = work.groupby(["store_id", "week"])
    frame = pd.DataFrame(
        {
            "m": grouped["units"].sum(),
            "p_bar": grouped["wp"].sum() / grouped["w"].sum().clip(lower=1e-9),
        }
    ).reset_index()
    frame = frame[frame["m"] > 0].copy()
    frame["log_m"] = np.log(frame["m"])
    cols = ["log_m", "p_bar"]
    for _ in range(4):
        for g in ["store_id", "week"]:
            frame[cols] = frame[cols] - frame.groupby(g)[cols].transform("mean")
    x = frame["p_bar"].to_numpy()
    y = frame["log_m"].to_numpy()
    denom = float(x @ x)
    slope = float(x @ y) / denom if denom > 1e-9 else 0.0
    if price_scale >= -1e-9:
        return 0.0
    return float(np.clip(slope / price_scale, 0.0, 1.5))


def fit_probit_simulation(
    transactions: pd.DataFrame,
    products: pd.DataFrame,
    stores: pd.DataFrame,
    *,
    use_iv: bool,
    use_text: bool,
) -> dict:
    product_ids = products["product_id"].tolist()
    j = len(product_ids)
    rng = np.random.default_rng(RNG_SEED)
    kernel = _build_kernel(products, use_text)
    chol = _build_sigma_chol(kernel, j)
    eps = (chol @ rng.standard_normal((j, N_DRAWS))).T

    if j < 10:
        sensitivity = fit_pooled_clipped(transactions, use_iv=use_iv)
    else:
        sensitivity = fit_share_price_sensitivity(transactions, use_iv=use_iv)
    b_target = sensitivity["b"]

    # Store-level observed average shares over the train window.
    units_ps = (
        transactions.groupby(["store_id", "product_id"])["units"].sum().unstack(fill_value=0.0)
    )
    units_ps = units_ps.reindex(columns=product_ids, fill_value=0.0)
    carried = units_ps.sum(axis=1) > 0
    units_ps = units_ps[carried]
    store_ids = list(units_ps.index)
    shares_stores = units_ps.to_numpy(dtype=float)
    shares_stores = shares_stores / shares_stores.sum(axis=1, keepdims=True)

    delta_stores = _invert_shares(shares_stores, eps)
    mean_prices_series = transactions.groupby("product_id")["price"].mean().reindex(product_ids)
    mean_prices = mean_prices_series.fillna(mean_prices_series.mean()).to_numpy(dtype=float)
    b_by_product = sensitivity.get("b_by_product", {})
    targets = np.array([float(b_by_product.get(p, b_target)) for p in product_ids])
    category_shares = shares_stores.mean(axis=0)
    price_scale = _calibrate_price_scale(
        delta_stores, shares_stores, mean_prices, eps, targets, category_shares
    )

    rho = _calibrate_incidence_rho(transactions, price_scale)

    market = transactions.groupby(["store_id", "week"])["units"].sum().rename("m")
    store_mean_m = market.groupby("store_id").mean()
    m_frame = market.reset_index()
    phase = 2.0 * np.pi * (m_frame["week"].astype(float) % SEASON_PERIOD_WEEKS) / SEASON_PERIOD_WEEKS
    rel = m_frame["m"] / m_frame["store_id"].map(store_mean_m)
    xs = np.column_stack([np.ones(len(m_frame)), np.sin(phase), np.cos(phase)])
    season_coef, *_ = np.linalg.lstsq(xs, rel.to_numpy(dtype=float), rcond=None)

    mean_price_by_store = (
        transactions.groupby(["store_id", "product_id"])["price"].mean().unstack()
    )
    mean_price_by_store = mean_price_by_store.reindex(index=store_ids, columns=product_ids)
    mean_price_by_store = mean_price_by_store.apply(lambda col: col.fillna(col.mean()), axis=0)
    mean_price_by_store = mean_price_by_store.fillna(float(np.nanmean(mean_prices)))

    return {
        "family": "covariance_probit",
        "estimator": "simulation",
        "use_iv": use_iv,
        "use_text": use_text,
        "product_ids": product_ids,
        "store_ids": store_ids,
        "store_index": {s: i for i, s in enumerate(store_ids)},
        "delta_stores": delta_stores,
        "shares_stores": shares_stores,
        "eps": eps,
        "sigma_chol": chol,
        "kernel": kernel,
        "price_scale": price_scale,
        "b": b_target,
        "rho": rho,
        "store_mean_m": store_mean_m,
        "season_coef": season_coef,
        "store_mean_prices": mean_price_by_store.to_numpy(dtype=float),
    }


def _market_size(params: dict, store_id, week) -> float:
    base = float(params["store_mean_m"].get(store_id, params["store_mean_m"].median()))
    phase = 2.0 * np.pi * (float(week) % SEASON_PERIOD_WEEKS) / SEASON_PERIOD_WEEKS
    c = params["season_coef"]
    return base * max(float(c[0] + c[1] * np.sin(phase) + c[2] * np.cos(phase)), 0.05)


def _shares_at_prices(params: dict, store_rows: np.ndarray, price_matrix: np.ndarray) -> np.ndarray:
    """Simulated shares for (store_row, price vector) pairs (B, J)."""
    delta = params["delta_stores"][store_rows] + params["price_scale"] * (
        price_matrix - params["store_mean_prices"][store_rows]
    )
    return _simulate_shares(delta, params["eps"])


def predict_holdout_units(params: dict, holdout_context: pd.DataFrame) -> pd.DataFrame:
    ctx = holdout_context[["product_id", "store_id", "week", "price"]].copy()
    wide = ctx.pivot_table(index=["store_id", "week"], columns="product_id", values="price", aggfunc="mean")
    wide = wide.reindex(columns=params["product_ids"])
    known = wide.index.get_level_values("store_id").map(params["store_index"]).notna()
    wide = wide[known]
    store_rows = wide.index.get_level_values("store_id").map(params["store_index"]).to_numpy(dtype=int)
    prices = wide.to_numpy(dtype=float)
    fallback = params["store_mean_prices"][store_rows]
    prices = np.where(np.isnan(prices), fallback, prices)
    shares = _shares_at_prices(params, store_rows, prices)
    m_hat = np.array([_market_size(params, s, w) for s, w in wide.index])
    units = shares * m_hat[:, None]
    frame = pd.DataFrame(units, index=wide.index, columns=params["product_ids"])
    long = frame.stack().rename("predicted_units").reset_index()
    long.columns = ["store_id", "week", "product_id", "predicted_units"]
    out = ctx.merge(long, on=["product_id", "store_id", "week"], how="left")
    out["predicted_units"] = out["predicted_units"].fillna(0.0)
    return out[["product_id", "store_id", "week", "predicted_units"]]


def predict_sweep_deltas(params: dict, sweep_context: pd.DataFrame) -> pd.DataFrame:
    frames = []
    rho = params["rho"]
    for intervention_id, ctx in sweep_context.groupby("intervention_id"):
        ctx = ctx.reset_index(drop=True)
        base_wide = ctx.pivot_table(index=["store_id", "week"], columns="product_id", values="baseline_price", aggfunc="mean").reindex(columns=params["product_ids"])
        cf_wide = ctx.pivot_table(index=["store_id", "week"], columns="product_id", values="intervention_price", aggfunc="mean").reindex(columns=params["product_ids"])
        known = base_wide.index.get_level_values("store_id").map(params["store_index"]).notna()
        base_wide, cf_wide = base_wide[known], cf_wide[known]
        store_rows = base_wide.index.get_level_values("store_id").map(params["store_index"]).to_numpy(dtype=int)
        fallback = params["store_mean_prices"][store_rows]
        pb = np.where(np.isnan(base_wide.to_numpy(dtype=float)), fallback, base_wide.to_numpy(dtype=float))
        pc = np.where(np.isnan(cf_wide.to_numpy(dtype=float)), fallback, cf_wide.to_numpy(dtype=float))
        s_base = np.clip(_shares_at_prices(params, store_rows, pb), 1e-9, None)
        s_base = s_base / s_base.sum(axis=1, keepdims=True)
        changed = ctx[np.abs(ctx["intervention_price"] - ctx["baseline_price"]) > 1e-12]
        focals = changed["product_id"].unique()
        if len(focals) == 1 and focals[0] in params["product_ids"]:
            focal_idx = params["product_ids"].index(focals[0])
            s_cf_sim = np.clip(_shares_at_prices(params, store_rows, pc), 1e-9, None)
            s_cf_sim = s_cf_sim / s_cf_sim.sum(axis=1, keepdims=True)
            s_cf = _allocate_cross(s_base, s_cf_sim[:, focal_idx], focal_idx, params["kernel"])
        else:
            s_cf = np.clip(_shares_at_prices(params, store_rows, pc), 1e-9, None)
            s_cf = s_cf / s_cf.sum(axis=1, keepdims=True)
        m_base = np.array([_market_size(params, s, w) for s, w in base_wide.index])
        # Replay-form incidence margin: dCV from the mean-utility shift of the
        # priced products, evaluated with the calibrated utility scale.
        d_util = params["price_scale"] * (pc - pb)
        dcv = np.log(np.clip((s_base * np.exp(d_util)).sum(axis=1), 1e-9, None))
        m_cf = m_base * np.exp(np.clip(rho * dcv, -10, 10))
        delta = s_cf * m_cf[:, None] - s_base * m_base[:, None]
        frame = pd.DataFrame(delta, index=base_wide.index, columns=params["product_ids"])
        long = frame.stack().rename("predicted_delta_units").reset_index()
        long.columns = ["store_id", "week", "product_id", "predicted_delta_units"]
        out = ctx.merge(long, on=["product_id", "store_id", "week"], how="left")
        out["predicted_delta_units"] = out["predicted_delta_units"].fillna(0.0)
        out.insert(0, "int_id", intervention_id)
        frames.append(out[["int_id", "product_id", "store_id", "week", "predicted_delta_units"]].rename(columns={"int_id": "intervention_id"}))
    return pd.concat(frames, ignore_index=True)


N_DRAWS_ELASTICITY = 120_000
ELASTICITY_ARC = 0.10


def elasticity_matrix(params: dict, products: pd.DataFrame) -> pd.DataFrame:
    """JxJ elasticities by simulated price bumps at pooled level.

    Cross-share responses to a small bump are far below the Monte-Carlo
    resolution of the fitting panel, so this uses a dedicated large panel of
    draws and a 10% arc (rescaled to per-1%): the flipped-draw count per
    competitor is what carries the cross pattern.
    """
    product_ids = params["product_ids"]
    jn = len(product_ids)
    rng = np.random.default_rng(RNG_SEED + 1)
    chol = params.get("sigma_chol")
    eps_big = (chol @ rng.standard_normal((jn, N_DRAWS_ELASTICITY))).T
    # Pool SHARES, not store utilities: averaging deltas collapses products
    # carried in few stores. Invert the category-average shares on the large
    # panel so every product sits at its true average share.
    pooled_target = params["shares_stores"].mean(axis=0)
    pooled_target = pooled_target / pooled_target.sum()
    pooled_delta = _invert_shares(pooled_target[None, :], eps_big)[0]
    pooled_prices = params["store_mean_prices"].mean(axis=0)
    base = np.clip(_simulate_shares(pooled_delta, eps_big), 1e-9, None)
    rho = params["rho"]
    rows = []
    for jp, priced in enumerate(product_ids):
        dp = ELASTICITY_ARC * float(pooled_prices[jp])
        bumped = pooled_delta.copy()
        bumped[jp] += params["price_scale"] * dp
        cf_sim = np.clip(_simulate_shares(bumped, eps_big), 1e-9, None)
        cf = _allocate_cross(base[None, :], cf_sim[jp : jp + 1], jp, params["kernel"])[0]
        dcv = float(np.log(np.clip((base * np.exp(np.eye(jn)[jp] * params["price_scale"] * dp)).sum(), 1e-9, None)))
        m_factor = float(np.exp(np.clip(rho * dcv, -10, 10)))
        for ja, affected in enumerate(product_ids):
            q0, q1 = base[ja], cf[ja] * m_factor
            elasticity = ((q1 - q0) / q0) / ELASTICITY_ARC if q0 > 0 else 0.0
            rows.append({"affected_product_id": affected, "priced_product_id": priced, "elasticity": float(elasticity)})
    return pd.DataFrame(rows)
