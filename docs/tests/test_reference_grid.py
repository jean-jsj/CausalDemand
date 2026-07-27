"""Smoke + contract tests for the 2x2 reference grid on a tiny synthetic cell."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")
pytest.importorskip("sklearn")

from causaldemand.baselines import loglog_grid, probit_shares, probit_simulation, text_distance
from causaldemand.baselines.run_reference_grid import VARIANTS, run_cell

PRODUCTS = pd.DataFrame(
    {
        "product_id": ["P1", "P2", "P3", "P4"],
        "product_text": [
            "soft gentle tissue for sensitive noses, plush and comforting",
            "soft plush tissue, gentle comfort for the family",
            "strong durable value tissue for big households",
            "sturdy value pack tissue, dependable and thrifty",
        ],
        "brand_code": ["A", "A", "B", "B"],
    }
)

STORES = pd.DataFrame(
    {
        "store_id": ["S1", "S2", "S3"],
        "market": ["m1", "m1", "m2"],
        "chain": ["c1", "c2", "c1"],
    }
)


def _make_transactions(seed: int = 7, weeks: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    base_price = {"P1": 2.0, "P2": 2.2, "P3": 1.6, "P4": 1.5}
    for week, store, product in itertools.product(
        range(1, weeks + 1), STORES["store_id"], PRODUCTS["product_id"]
    ):
        cost = 1.0 + 0.15 * np.sin(week / 6.0) + rng.normal(0, 0.12)
        promo = int(rng.random() < 0.15)
        depth = rng.uniform(0.10, 0.35) if promo else 0.0
        price = base_price[product] * (1 + 0.8 * (cost - 1.0)) * (1.0 - depth)
        # Promotions move demand only through the price cut (as in the DGP).
        mean_units = np.exp(4.0 - 1.6 * np.log(price) + rng.normal(0, 0.1))
        units = rng.poisson(mean_units)
        rows.append(
            {
                "product_id": product,
                "store_id": store,
                "week": week,
                "units": float(units),
                "dollars": float(units) * price,
                "price": round(price, 2),
                "promo_flag": promo,
                "promo_cost": 0.2 if promo else 0.0,
                "supply_cost_proxy": round(cost, 4),
            }
        )
    return pd.DataFrame(rows)


TRANSACTIONS = _make_transactions()
HOLDOUT = (
    _make_transactions(seed=11, weeks=86)
    .query("week > 80")[
        ["product_id", "store_id", "week", "price", "promo_flag", "promo_cost", "supply_cost_proxy"]
    ]
    .reset_index(drop=True)
)


def _make_sweep(holdout: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for direction, factor in (("plus10", 1.1), ("minus10", 0.9)):
        ctx = holdout[["product_id", "store_id", "week", "price"]].copy()
        ctx = ctx.rename(columns={"price": "baseline_price"})
        ctx["intervention_price"] = np.where(
            ctx["product_id"] == "P1",
            (ctx["baseline_price"] * factor).round(2),
            ctx["baseline_price"],
        )
        ctx.insert(0, "intervention_id", f"sweep_single_random_{direction}")
        ctx["promo_cost"] = 0.0
        frames.append(ctx)
    return pd.concat(frames, ignore_index=True)


SWEEP = _make_sweep(HOLDOUT)


def test_text_distance_matrix_groups_similar_texts() -> None:
    dist = text_distance.text_distance_matrix(PRODUCTS)
    assert dist.shape == (4, 4)
    assert np.allclose(np.diag(dist.to_numpy()), 0.0)
    # P1/P2 share soft/gentle vocabulary; P3/P4 share value vocabulary.
    assert dist.loc["P1", "P2"] < dist.loc["P1", "P3"]
    assert dist.loc["P3", "P4"] < dist.loc["P3", "P1"]


def test_text_distance_deterministic() -> None:
    a = text_distance.text_distance_matrix(PRODUCTS)
    b = text_distance.text_distance_matrix(PRODUCTS)
    pd.testing.assert_frame_equal(a, b)


def test_brand_distance_matrix() -> None:
    dist = text_distance.brand_distance_matrix(PRODUCTS)
    assert dist.loc["P1", "P2"] == 0.0
    assert dist.loc["P1", "P3"] == 1.0


def test_kernel_weights_rows_normalized() -> None:
    dist = text_distance.text_distance_matrix(PRODUCTS)
    weights = text_distance.kernel_weights(dist, k=2)
    sums = weights.sum(axis=1)
    assert np.allclose(sums, 1.0)
    assert np.allclose(np.diag(weights.to_numpy()), 0.0)


@pytest.mark.parametrize("use_iv,use_text", [(False, False), (True, True)])
def test_loglog_corner_recovers_negative_own_price(use_iv: bool, use_text: bool) -> None:
    params = loglog_grid.fit_loglog_corner(
        TRANSACTIONS, PRODUCTS, use_iv=use_iv, use_text=use_text
    )
    betas = [c["beta"] for c in params["coefs"].values()]
    if use_iv:
        # A 3-store fixture supports only a weak first stage; the real cells
        # (731 stores) identify the IV corner tightly.
        assert sum(b < 0 for b in betas) >= 3, betas
        assert np.mean(betas) < -0.8, betas
    else:
        assert all(b < 0 for b in betas), betas
        assert np.mean(betas) == pytest.approx(-1.6, abs=0.6)


def test_loglog_submission_shapes() -> None:
    params = loglog_grid.fit_loglog_corner(TRANSACTIONS, PRODUCTS, use_iv=False, use_text=True)
    forecast = loglog_grid.predict_holdout_units(params, HOLDOUT)
    assert list(forecast.columns) == ["product_id", "store_id", "week", "predicted_units"]
    assert len(forecast) == len(HOLDOUT)
    assert (forecast["predicted_units"] >= 0).all()

    elasticity = loglog_grid.elasticity_matrix(params, PRODUCTS)
    assert len(elasticity) == 16
    own = elasticity[elasticity["affected_product_id"] == elasticity["priced_product_id"]]
    assert (own["elasticity"] < 0).all()

    deltas = loglog_grid.predict_sweep_deltas(params, SWEEP, HOLDOUT)
    assert len(deltas) == len(SWEEP)
    focal = deltas[
        (deltas["product_id"] == "P1")
        & (deltas["intervention_id"] == "sweep_single_random_plus10")
    ]
    assert focal["predicted_delta_units"].mean() < 0


def test_share_price_sensitivity_negative() -> None:
    pooled = probit_shares.fit_pooled_clipped(TRANSACTIONS, use_iv=False)
    assert pooled["b"] < 0
    pooled_iv = probit_shares.fit_pooled_clipped(TRANSACTIONS, use_iv=True)
    assert pooled_iv["b"] < 0


def test_probit_simulation_sensitivity_and_shapes() -> None:
    params = probit_simulation.fit_probit_simulation(
        TRANSACTIONS, PRODUCTS, STORES, use_iv=True, use_text=True
    )
    assert params["price_scale"] < 0
    assert 0.0 <= params["rho"] <= 1.5

    forecast = probit_simulation.predict_holdout_units(params, HOLDOUT)
    assert len(forecast) == len(HOLDOUT)
    assert (forecast["predicted_units"] >= 0).all()

    deltas = probit_simulation.predict_sweep_deltas(params, SWEEP)
    assert len(deltas) == len(SWEEP)
    plus = deltas[deltas["intervention_id"] == "sweep_single_random_plus10"]
    focal_delta = plus[plus["product_id"] == "P1"]["predicted_delta_units"].sum()
    rival_delta = plus[plus["product_id"] != "P1"]["predicted_delta_units"].sum()
    assert focal_delta < 0
    assert rival_delta > 0

    elasticity = probit_simulation.elasticity_matrix(params, PRODUCTS)
    own = elasticity[elasticity["affected_product_id"] == elasticity["priced_product_id"]]
    assert (own["elasticity"] <= 0).all()


def test_run_cell_writes_all_variants(tmp_path: Path) -> None:
    cell = tmp_path / "cells" / "tiny_log_log_exogenous_seed001"
    public = cell / "public"
    public.mkdir(parents=True)
    TRANSACTIONS.to_csv(public / "transactions_train_public.csv", index=False)
    PRODUCTS.to_csv(public / "products_public.csv", index=False)
    STORES.to_csv(public / "stores_public.csv", index=False)
    HOLDOUT.to_csv(public / "transactions_holdout_context_public.csv", index=False)
    SWEEP.to_csv(public / "counterfactual_sweep_context_public.csv", index=False)

    out_root = tmp_path / "reference_scores"
    run_cell(cell, out_root, VARIANTS)
    for variant in VARIANTS:
        target = out_root / variant / cell.name
        for name in (
            "forecast_predictions.csv",
            "elasticity_matrix.csv",
            "counterfactual_deltas.csv",
        ):
            assert (target / name).is_file(), f"{variant}/{name} missing"
