from __future__ import annotations

import numpy as np
import pandas as pd

import os

_CENT = float(os.environ.get("CARD_PRICE_ENDING_CENT") or "0.01")

TRAIN_COLUMNS = [
    "product_id",
    "store_id",
    "week",
    "units",
    "price",
    "promo_flag",
    "promo_cost",
    "supply_cost_proxy",
]


def load_public_cell(public_dir) -> dict:
    import os

    read = lambda name: pd.read_csv(os.path.join(str(public_dir), name))
    return {
        "transactions": read("transactions_train_public.csv"),
        "products": read("products_public.csv"),
        "stores": read("stores_public.csv"),
        "holdout_context": read("transactions_holdout_context_public.csv"),
        "sweep_context": read("counterfactual_sweep_context_public.csv"),
    }


def _nearest_fill(frame: pd.DataFrame, value_col: str, by: list) -> pd.Series:
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


def reconstruct_grid(public: dict) -> pd.DataFrame:
    tx = public["transactions"].copy()
    stores = public["stores"][["store_id", "chain"]]
    products = public["products"][["product_id", "brand_code"]]

    weeks = np.arange(tx["week"].min(), tx["week"].max() + 1)
    pairs = tx[["product_id", "store_id"]].drop_duplicates()
    grid = (
        pairs.merge(pd.DataFrame({"week": weeks}), how="cross")
        .merge(tx, on=["product_id", "store_id", "week"], how="left")
        .merge(stores, on="store_id", how="left")
        .merge(products, on="product_id", how="left")
    )
    grid["present"] = grid["units"].notna()
    grid["units"] = grid["units"].fillna(0.0)

    obs = grid[grid["present"]]
    flag_cbw = (
        obs.groupby(["chain", "brand_code", "week"])["promo_flag"].first().rename("flag_cbw")
    )
    pcost_cbw = (
        obs[obs["promo_flag"] == 1]
        .groupby(["chain", "brand_code", "week"])["promo_cost"]
        .first()
        .rename("pcost_cbw")
    )
    scost_cpw = (
        obs.groupby(["chain", "product_id", "week"])["supply_cost_proxy"]
        .first()
        .rename("scost_cpw")
    )

    grid = grid.join(flag_cbw, on=["chain", "brand_code", "week"])
    grid = grid.join(pcost_cbw, on=["chain", "brand_code", "week"])
    grid = grid.join(scost_cpw, on=["chain", "product_id", "week"])

    grid["promo_flag_src"] = np.where(
        grid["present"], "observed", np.where(grid["flag_cbw"].notna(), "chain", "timefill")
    )
    grid["promo_flag"] = grid["promo_flag"].fillna(grid["flag_cbw"])
    grid = grid.sort_values(["chain", "brand_code", "store_id", "product_id", "week"]).reset_index(
        drop=True
    )
    still = grid["promo_flag"].isna()
    if still.any():
        filled = _nearest_fill(grid, "promo_flag", ["chain", "brand_code", "store_id", "product_id"])
        grid.loc[still, "promo_flag"] = filled[still]
    grid["promo_flag"] = grid["promo_flag"].fillna(0.0).astype(int)

    grid["promo_cost"] = np.where(
        grid["promo_flag"] == 0, 0.0, grid["promo_cost"].fillna(grid["pcost_cbw"])
    )
    on = grid["promo_flag"] == 1
    missing_pc = on & pd.isna(grid["promo_cost"])
    if missing_pc.any():
        tmp = grid["promo_cost"].where(~missing_pc)
        grid["promo_cost"] = tmp
        filled = _nearest_fill(grid, "promo_cost", ["chain", "brand_code", "store_id", "product_id"])
        grid.loc[missing_pc, "promo_cost"] = filled[missing_pc]
        grid["promo_cost"] = grid["promo_cost"].fillna(0.0)

    grid["supply_cost_proxy"] = grid["supply_cost_proxy"].fillna(grid["scost_cpw"])
    still = grid["supply_cost_proxy"].isna()
    if still.any():
        filled = _nearest_fill(grid, "supply_cost_proxy", ["chain", "product_id", "store_id"])
        grid.loc[still, "supply_cost_proxy"] = filled[still]
    grid["supply_cost_proxy"] = grid.groupby("product_id")["supply_cost_proxy"].transform(
        lambda s: s.fillna(s.mean())
    )

    grid["regular_obs"] = np.where(
        grid["present"] & (grid["promo_flag"] == 0), grid["price"] + _CENT, np.nan
    )
    grid = grid.sort_values(["store_id", "product_id", "week"]).reset_index(drop=True)
    grid["regular_price"] = _nearest_fill(grid, "regular_obs", ["store_id", "product_id"])
    grid["regular_src"] = np.where(
        grid["regular_obs"].notna(), "observed", np.where(grid["regular_price"].notna(), "timefill", "fallback")
    )
    chain_med = (
        grid.groupby(["chain", "product_id", "week"])["regular_price"].transform("median")
    )
    grid["regular_price"] = grid["regular_price"].fillna(chain_med)
    prod_med = grid.groupby("product_id")["regular_price"].transform("median")
    grid["regular_price"] = grid["regular_price"].fillna(prod_med)

    on = (grid["promo_flag"] == 1).to_numpy()
    present = grid["present"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        depth_obs = 1.0 - (grid["price"].to_numpy() + _CENT) / grid["regular_price"].to_numpy()
    depth = np.where(on & present, np.clip(depth_obs, 0.0, 0.90), np.nan)
    grid["depth"] = depth
    grid["depth_src"] = np.where(~on, "off", np.where(present, "observed", "imputed"))
    cpw_mean = grid.groupby(["chain", "product_id", "week"])["depth"].transform("mean")
    pw_mean = grid.groupby(["product_id", "week"])["depth"].transform("mean")
    p_mean = grid.groupby("product_id")["depth"].transform("mean")
    fill = cpw_mean.fillna(pw_mean).fillna(p_mean).fillna(0.0)
    grid["depth"] = np.where(on, np.where(np.isnan(depth), fill, depth), 0.0)

    grid["log_regular"] = np.log(grid["regular_price"].astype(float).clip(lower=0.01))
    return grid.drop(columns=["flag_cbw", "pcost_cbw", "scost_cpw", "regular_obs"])


def depth_channel_interventions(
    sweep_context: pd.DataFrame, holdout_context: pd.DataFrame
) -> dict:
    promo = holdout_context.set_index(["product_id", "store_id", "week"])[
        "promo_flag"
    ]
    out = {}
    for intervention_id, ctx in sweep_context.groupby("intervention_id"):
        moved = ctx[
            (
                ctx["intervention_price"].astype(float)
                - ctx["baseline_price"].astype(float)
            ).abs()
            > 1e-9
        ]
        if moved.empty:
            out[str(intervention_id)] = False
            continue
        key = pd.MultiIndex.from_frame(moved[["product_id", "store_id", "week"]])
        flags = pd.Series(key.map(promo).to_numpy(), dtype=float)
        known = flags.dropna()
        out[str(intervention_id)] = bool(
            len(known) > 0 and (known == 0).sum() == 0
        )
    return out


def reconstruction_provenance(grid: pd.DataFrame) -> dict:
    return {
        "rows": int(len(grid)),
        "present_rows": int(grid["present"].sum()),
        "zero_rows_reinserted": int((~grid["present"]).sum()),
        "promo_flag_sources": grid["promo_flag_src"].value_counts().to_dict(),
        "regular_sources": grid["regular_src"].value_counts().to_dict(),
        "depth_sources": grid["depth_src"].value_counts().to_dict(),
    }
