from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from card_eval_lib import Cell

PERTURBATION = 0.01


def _evaluation_grid(cell: Cell) -> pd.DataFrame:
    stores = sorted(cell.stores["store_id"].astype(str).unique())
    index = pd.MultiIndex.from_product(
        [cell.product_ids, stores, cell.eval_weeks],
        names=["product_id", "store_id", "week"],
    )
    return index.to_frame(index=False)


def build_layer1(model, cell: Cell) -> pd.DataFrame:
    ctx = cell.holdout_context.copy()
    ctx["predicted_units"] = model.predict_units(ctx)
    pred = ctx[["product_id", "store_id", "week", "predicted_units"]]
    complete = _evaluation_grid(cell).merge(
        pred, on=["product_id", "store_id", "week"], how="left"
    )
    complete["predicted_units"] = complete["predicted_units"].fillna(0.0)
    return complete


def _eval_baseline_frame(cell: Cell) -> pd.DataFrame:
    return cell.holdout_context.copy()


def build_layer2(model, cell: Cell) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = _eval_baseline_frame(cell)
    q_base = model.predict_units(base)
    base = base.assign(_q_base=q_base)

    products = cell.product_ids
    denom = base.groupby("product_id")["_q_base"].sum()

    matrix = pd.DataFrame(0.0, index=pd.Index(products, name="affected_product_id"),
                          columns=pd.Index(products, name="priced_product_id"))
    for j in products:
        pert = base.copy()
        mask = pert["product_id"] == j
        if not mask.any():
            continue
        pert.loc[mask, "price"] = pert.loc[mask, "price"].astype(float) * (1.0 + PERTURBATION)
        q_pert = model.predict_units(pert)
        dq = q_pert - base["_q_base"].to_numpy()
        dq_by_i = pd.Series(dq, index=base.index).groupby(base["product_id"]).sum()
        for i in products:
            di = float(dq_by_i.get(i, 0.0))
            dn = float(denom.get(i, 0.0))
            matrix.loc[i, j] = di / (PERTURBATION * dn) if dn > 0 else 0.0

    long_df = (matrix.reset_index()
               .melt(id_vars="affected_product_id", var_name="priced_product_id",
                     value_name="elasticity"))
    long_df = long_df[["priced_product_id", "affected_product_id", "elasticity"]]
    return long_df, matrix


def build_layer3(model, cell: Cell) -> pd.DataFrame:
    sw = cell.sweep_context.copy()
    key = ["product_id", "store_id", "week"]
    public_flags = cell.holdout_context[key + ["promo_flag"]].drop_duplicates(key)
    sw = sw.merge(
        public_flags,
        on=key,
        how="left",
        validate="many_to_one",
    )
    fallback_flag = (
        pd.to_numeric(sw["promo_cost"], errors="coerce").fillna(0.0) != 0.0
    ).astype(int)
    sw["promo_flag"] = pd.to_numeric(
        sw["promo_flag"], errors="coerce"
    ).fillna(fallback_flag).astype(int)

    base_scn = (sw.drop_duplicates(key)
                  .rename(columns={"baseline_price": "price"}))
    q_base_vec = model.predict_units(base_scn)
    base_lookup = pd.Series(q_base_vec, index=pd.MultiIndex.from_frame(base_scn[key]))

    out = []
    for iid, grp in sw.groupby("intervention_id", sort=False):
        q_base = base_lookup.reindex(pd.MultiIndex.from_frame(grp[key])).to_numpy()
        cf = grp.rename(columns={"intervention_price": "price"})
        q_cf = model.predict_units(cf)
        d = grp[["intervention_id", "product_id", "store_id", "week"]].copy()
        d["predicted_delta_units"] = q_cf - q_base
        d["_baseline_units"] = q_base
        out.append(d)
    return pd.concat(out, ignore_index=True)


def _complete_layer3_submission(l3: pd.DataFrame, cell: Cell) -> pd.DataFrame:
    grid = _evaluation_grid(cell)
    keys = ["product_id", "store_id", "week"]
    out = []
    for intervention_id in cell.sweep_context["intervention_id"].astype(str).unique():
        block = grid.copy()
        block["intervention_id"] = intervention_id
        submitted = l3[l3["intervention_id"].astype(str) == intervention_id][
            ["intervention_id", *keys, "predicted_delta_units"]
        ]
        block = block.merge(submitted, on=["intervention_id", *keys], how="left")
        block["predicted_delta_units"] = block["predicted_delta_units"].fillna(0.0)
        out.append(block[["intervention_id", *keys, "predicted_delta_units"]])
    return pd.concat(out, ignore_index=True)


def write_submission(model, cell: Cell, out_dir: str | Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    l1 = build_layer1(model, cell)
    l1.to_csv(out_dir / "layer1_demand_predictions.csv", index=False)

    l2_long, l2_matrix = build_layer2(model, cell)
    l2_long.to_csv(out_dir / "layer2_elasticities.csv", index=False)

    l3 = build_layer3(model, cell)
    l3_submission = _complete_layer3_submission(l3, cell)
    l3_submission.to_csv(out_dir / "layer3_counterfactual_deltas.csv", index=False)

    return {"layer2_matrix": l2_matrix, "layer3_frame": l3, "out_dir": out_dir}
