from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from causaldemand import _dff_read as dff
from causaldemand._panel_select import panel_pinned_depth_selections

dff.log.addHandler(logging.NullHandler())
dff.log.addFilter(lambda record: not str(record.msg).startswith("SIZE is not a weight"))

BRANDS = Path(__file__).resolve().parent / "dominicks_brands"
N_WEEKS = 156
N_TEST = 16
RANK_WEEKS = 52
MODE_HALF = 6
COST_REF_WEEKS = 9
PROMO_THRESHOLD = 0.05
STORE_WEEK_SHARE = 0.95
GAP_FRAC = 0.5
DEPTH_CAP = 0.90
MARGIN_CEILING = 0.45
PRICE_OUTLIER_FOLD = 10.0
CHAIN, MARKET = "DFF", "CHICAGO"
KEY = ["product_id", "store_id", "week"]
CSV = {"index": False, "lineterminator": "\n"}


def rolling_mode(mat: np.ndarray, half: int) -> np.ndarray:
    n, t = mat.shape
    pad = np.full((n, half), np.nan)
    win = np.lib.stride_tricks.sliding_window_view(np.concatenate([pad, mat, pad], axis=1), 2 * half + 1, axis=1)
    out = np.full((n, t), np.nan)
    for i0 in range(0, n, 500):
        w = win[i0:i0 + 500]
        cnt = (w[..., :, None] == w[..., None, :]).sum(-1).astype(float)
        cnt[np.isnan(w)] = -1
        best = cnt.max(-1, keepdims=True)
        val = np.where(cnt == best, w, -np.inf).max(-1)
        val[best[..., 0] < 2] = np.nan
        out[i0:i0 + 500] = val
    return out


def nearest_fill(df: pd.DataFrame, keys: list[str], week: str, value: str) -> np.ndarray:
    grp = [df[k] for k in keys]
    seen_week = df[week].where(df[value].notna()).astype(float)
    prev_v = df[value].groupby(grp, sort=False).ffill()
    prev_w = seen_week.groupby(grp, sort=False).ffill()
    next_v = df[value].groupby(grp, sort=False).bfill()
    next_w = seen_week.groupby(grp, sort=False).bfill()
    w = df[week].astype(float)
    use_prev = prev_v.notna() & (next_v.isna() | ((w - prev_w) <= (next_w - w)))
    return np.where(use_prev, prev_v, next_v)


def regular_and_depth(rows: pd.DataFrame, train_last: int) -> pd.DataFrame:
    r = rows.sort_values(["store", "upc", "week"]).copy()
    r["off_price"] = r["price"].where(r["promo_flag"] == 0)
    tr = r["week"] <= train_last
    reg = pd.Series(np.nan, index=r.index)
    reg[tr] = nearest_fill(r[tr], ["store", "upc"], "week", "off_price")
    reg[~tr] = pd.Series(nearest_fill(r, ["store", "upc"], "week", "off_price"), index=r.index)[~tr]
    med = reg[tr].groupby(r.loc[tr, "upc"]).median()
    n_fallback = int(reg.isna().sum())
    reg = reg.fillna(r["upc"].map(med)).fillna(r["price"])
    with np.errstate(divide="ignore", invalid="ignore"):
        dep = 1.0 - r["price"].to_numpy(float) / reg.to_numpy(float)
    r["regular"] = reg
    r["depth"] = np.where(r["promo_flag"].to_numpy() == 1, np.clip(np.nan_to_num(dep, nan=0.0), 0.0, DEPTH_CAP), 0.0)
    r.attrs["n_regular_fallback"] = n_fallback
    return r


def demean(x: pd.Series, groupings: list[list[pd.Series]], iters: int = 30) -> pd.Series:
    y = x.astype(float).copy()
    for _ in range(iters):
        for g in groupings:
            y = y - y.groupby(g).transform("mean")
    return y


def read_store_file(raw: Path) -> pd.DataFrame:
    with zipfile.ZipFile(raw / "demo_stata.zip") as z:
        demo = pd.read_stata(io.BytesIO(z.read("demo.dta")))
    demo = demo.dropna(subset=["store"]).copy()
    demo["store"] = demo["store"].astype(int)
    tier = np.select([demo["priclow"] == 1, demo["pricmed"] == 1, demo["prichigh"] == 1], ["low", "medium", "high"], "")
    demo["price_tier"] = tier
    return demo[["store", "city", "zone", "price_tier", "weekvol"]].drop_duplicates("store").set_index("store")


def build(args) -> dict:
    cat = args.category
    raw, out = Path(args.raw), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; choose a new folder")
    last = int(args.last_week)
    first = last - N_WEEKS + 1
    train_last = last - N_TEST
    rank_first = train_last - RANK_WEEKS + 1
    late_first = train_last - N_TEST + 1
    pre = first - max(MODE_HALF, COST_REF_WEEKS - 1)
    if pre < 1:
        raise SystemExit(f"--last-week {last} leaves no room for the {first - pre} weeks before the window")
    for sub in ("public", "ground_truth"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    cfg = dff.Config(category=cat, data_dir=raw)
    upc_file = dff.build_upc_descriptions(dff.read_upc_file(raw / f"upc{cat}.csv"), cfg.oz_per_eq_unit)
    mv = dff.read_movement(raw / f"w{cat}.zip", cfg, set(upc_file["upc"].astype("int64")))
    mv = mv[(mv["week"] >= pre) & (mv["week"] <= last)].copy()
    mv["store"] = mv["store"].astype(int)
    mv["week"] = mv["week"].astype(int)
    mv["upc"] = mv["upc"].astype("int64")
    mv["dollars"] = mv["price"] * mv["move"] / mv["qty"]
    margin = dff.profit_to_margin(mv["profit"])
    mv["cost"] = (mv["price"] / mv["qty"] * (1.0 - margin)).where(margin.notna())
    mv.loc[~np.isfinite(mv["cost"]) | (mv["cost"] <= 0), "cost"] = np.nan
    mv["sale"] = mv["sale"].fillna("").astype(str).str.strip().str.upper() if "sale" in mv else ""
    keys = ["store", "upc", "week"]
    mv = (mv.groupby(keys, as_index=False)
            .agg(units=("move", "sum"), dollars=("dollars", "sum"), cost=("cost", "median"), sale=("sale", "max")))
    sold = mv[mv["units"] > 0].copy()

    win = sold[sold["week"] >= first]
    per_week = win.groupby("week").agg(rows=("units", "size"), stores=("store", "nunique"),
                                       rev=("dollars", "sum")).reindex(range(first, last + 1), fill_value=0)
    thin = per_week.index[(per_week["rows"] < GAP_FRAC * per_week["rows"].median())
                          | (per_week["stores"] < GAP_FRAC * per_week["stores"].median())]
    copied = per_week.index[(per_week["rows"] > 0) & (per_week["rows"] == per_week["rows"].shift())
                            & (per_week["rev"].round(2) == per_week["rev"].round(2).shift())]
    if len(thin) or len(copied):
        raise SystemExit(f"weeks {first}-{last} are not complete: thin weeks {list(thin)}, copied weeks {list(copied)}")
    weeks_selling = win.groupby("store")["week"].nunique()
    rank_rows = win[(win["week"] >= rank_first) & (win["week"] <= train_last)]
    store_rev = rank_rows.groupby("store")["dollars"].sum()
    eligible = weeks_selling.index[weeks_selling >= STORE_WEEK_SHARE * N_WEEKS - 1e-9]
    stores_sel = list(store_rev.reindex(eligible).fillna(0).sort_values(ascending=False, kind="mergesort").index)
    in_s = win[win["store"].isin(stores_sel)]
    prod_rev = (in_s[(in_s["week"] >= rank_first) & (in_s["week"] <= train_last)].groupby("upc")["dollars"].sum()
                .sort_values(ascending=False, kind="mergesort"))
    upcs_sel = [int(u) for u in prod_rev.index[: args.n_products]]
    if len(upcs_sel) < args.n_products:
        raise SystemExit(f"only {len(upcs_sel)} UPCs sell in the ranking weeks")
    n_late = in_s[(in_s["week"] >= late_first) & (in_s["week"] <= train_last)].groupby("upc").size()
    n_test = in_s[in_s["week"] > train_last].groupby("upc").size()
    presence = (n_test.reindex(upcs_sel).fillna(0) / n_late.reindex(upcs_sel)).fillna(0.0)
    demo = read_store_file(raw)
    miss = [s for s in stores_sel if s not in demo.index or not np.isfinite(demo.loc[s, "weekvol"])
            or demo.loc[s, "weekvol"] <= 0]
    if miss:
        raise SystemExit(f"selected stores without a weekly volume in the store file: {miss}")
    store_ids = {s: f"S{i:04d}" for i, s in enumerate(stores_sel, start=1)}
    stores_hidden = pd.DataFrame({
        "store_id": [store_ids[s] for s in stores_sel], "dff_store": stores_sel,
        "city": demo.loc[stores_sel, "city"].astype(str).str.strip().to_numpy(),
        "zone": demo.loc[stores_sel, "zone"].to_numpy(), "price_tier": demo.loc[stores_sel, "price_tier"].to_numpy(),
        "weekvol": demo.loc[stores_sel, "weekvol"].astype(float).to_numpy(),
        "weeks_selling_in_window": weeks_selling.loc[stores_sel].to_numpy(),
        "revenue_rank_window": store_rev.reindex(stores_sel).fillna(0).round(2).to_numpy(),
    })
    stores_public = pd.DataFrame({"store_id": stores_hidden["store_id"], "market": MARKET, "chain": CHAIN,
                                  "household_count": stores_hidden["weekvol"]})
    brands = pd.read_csv(args.brands or BRANDS / f"brands_{cat}.csv")
    brand_of = dict(zip(brands["upc"].astype("int64"), brands["brand_code"].astype(str).str.strip()))
    no_brand = [u for u in upcs_sel if not brand_of.get(u)]
    if no_brand:
        raise SystemExit(f"selected UPCs missing from the brand map: {no_brand}")
    prod_ids = {u: f"P{i:03d}" for i, u in enumerate(upcs_sel, start=1)}
    uf = upc_file.set_index("upc").loc[upcs_sel]
    products_hidden = pd.DataFrame({
        "product_id": [prod_ids[u] for u in upcs_sel], "upc": upcs_sel,
        "brand_code": [brand_of[u] for u in upcs_sel], "description": uf["description"].to_numpy(),
        "descrip": uf["descrip"].to_numpy(), "size": uf["size"].to_numpy(), "nitem": uf["nitem"].to_numpy(),
        "commodity_code": uf["commodity_code"].to_numpy(), "manufacturer_code": uf["manufacturer_code"].to_numpy(),
        "manufacturer_brand_prepare_loglog": uf["brand"].to_numpy(),
        "revenue_rank_window": prod_rev.loc[upcs_sel].round(2).to_numpy(),
        "test_presence": presence.loc[upcs_sel].round(4).to_numpy(),
    })
    products_public = products_hidden[["product_id", "brand_code"]]

    sel_upc = set(upcs_sel)
    cw = sold[sold["upc"].isin(sel_upc) & sold["cost"].notna()].groupby(["upc", "week"])["cost"].median()
    weeks_all = pd.Index(range(pre, last + 1), name="week")
    chain = cw.unstack().reindex(index=pd.Index(upcs_sel, name="upc"), columns=weeks_all)
    chain = chain.T.ffill().bfill().T
    if chain.isna().any().any():
        raise SystemExit("a selected product has no wholesale cost in any week")
    base = chain.T.rolling(COST_REF_WEEKS, min_periods=1).max().T
    drop = (1.0 - chain / base).clip(0.0, DEPTH_CAP)
    inst = pd.DataFrame({"chain_cost": chain.stack(), "base_cost": base.stack(), "cost_drop": drop.stack()}).reset_index()

    sample = sold[sold["store"].isin(stores_sel) & sold["upc"].isin(sel_upc)].copy()
    sample["price"] = (sample["dollars"] / sample["units"]).round(2)
    med = sample["upc"].map(sample[(sample["week"] >= first) & (sample["week"] <= train_last)]
                            .groupby("upc")["price"].median())
    outlier = sample["price"] > PRICE_OUTLIER_FOLD * med
    sample = sample[~outlier & (sample["price"] > 0)].reset_index(drop=True)

    cents = sample.pivot_table(index=["store", "upc"], columns="week", values="price", aggfunc="first") \
                  .reindex(columns=weeks_all) * 100.0
    cents = cents.round()
    m = cents.to_numpy(float)
    n_tr = train_last - pre + 1
    ref = np.concatenate([rolling_mode(m[:, :n_tr], MODE_HALF), rolling_mode(m, MODE_HALF)[:, n_tr:]], axis=1)
    ref = pd.DataFrame(ref, index=cents.index, columns=weeks_all).stack().rename("ref_cents").reset_index()
    sample = sample.merge(ref, on=["store", "upc", "week"], how="left", validate="one_to_one")
    with np.errstate(divide="ignore", invalid="ignore"):
        cut = 1.0 - np.round(sample["price"] * 100.0) / sample["ref_cents"]
    sample["promo_flag"] = (cut.fillna(0.0) >= PROMO_THRESHOLD - 1e-9).astype(int)
    sample = sample[sample["week"] >= first].reset_index(drop=True)
    sample = sample.merge(inst, on=["upc", "week"], how="left", validate="many_to_one")
    sample["supply_cost_proxy"] = sample["base_cost"].round(4)
    sample["promo_cost"] = sample["cost_drop"].round(4)
    if not (np.isfinite(sample["supply_cost_proxy"]).all() and (sample["supply_cost_proxy"] > 0).all()):
        raise SystemExit("supply_cost_proxy is not positive and finite on every row")
    rd = regular_and_depth(sample[["store", "upc", "week", "price", "promo_flag"]], train_last)
    sample = sample.merge(rd[["store", "upc", "week", "regular", "depth"]], on=["store", "upc", "week"],
                          how="left", validate="one_to_one")
    sample["sale_code"] = sample["sale"]

    sample["product_id"] = sample["upc"].map(prod_ids)
    sample["store_id"] = sample["store"].map(store_ids)
    sample["units"] = sample["units"].astype(float)
    sample["dollars"] = sample["dollars"].round(2)
    sample = sample.sort_values(["week", "store_id", "product_id"]).reset_index(drop=True)
    train = sample[sample["week"] <= train_last]
    pairs = train[["product_id", "store_id"]].drop_duplicates()
    test = sample[sample["week"] > train_last].merge(pairs, on=["product_id", "store_id"], how="inner")
    train_cols = ["product_id", "store_id", "week", "units", "dollars", "price", "promo_flag",
                  "promo_cost", "supply_cost_proxy"]
    hold_cols = ["product_id", "store_id", "week", "price", "promo_flag", "promo_cost", "supply_cost_proxy"]
    holdout = test[hold_cols].reset_index(drop=True)

    rank_train = train[train["week"] >= rank_first]
    universe = sorted(train["product_id"].unique())
    promoted = sorted(holdout.loc[holdout["promo_flag"] == 1, "product_id"].unique())
    by_units = panel_pinned_depth_selections(rank_train, products_public, universe, n_products=len(universe))
    product_rules = [s for s in panel_pinned_depth_selections(rank_train, products_public, promoted)
                     if s["rule"].startswith("panel_product")]
    brand_rules = [s for s in by_units if s["rule"].startswith("panel_brand")]
    for s in product_rules:
        s["basis"]["universe"] = "products with at least one promoted test-week row"
    selections = product_rules + brand_rules
    for sel in selections:
        sel["basis"]["pinned"] = "train_last52_units_over_universe"
        sel["basis"]["defined_on"] = "observed_promo_flag"
        sel["basis"]["ranking_weeks"] = [rank_first, train_last]

    refh = holdout[KEY + ["price", "promo_flag"]].sort_values(["store_id", "product_id", "week"]).copy()
    refh["regular"] = np.where(refh["promo_flag"] == 0, refh["price"], np.nan)
    g = refh.groupby(["store_id", "product_id"], sort=False)["regular"]
    refh["regular"] = g.ffill().fillna(g.bfill())
    last_off = (train[train["promo_flag"] == 0].sort_values("week")
                .groupby(["store_id", "product_id"])["price"].last().rename("last_off"))
    refh = refh.join(last_off, on=["store_id", "product_id"])
    refh["regular"] = refh["regular"].fillna(refh["last_off"])
    refh["regular"] = refh["regular"].fillna(refh.groupby("product_id")["regular"].transform("median"))
    refh["regular"] = refh["regular"].fillna(refh["price"])
    holdout_reg = holdout.merge(refh[KEY + ["regular"]], on=KEY, how="left", validate="one_to_one")["regular"].to_numpy()
    promo_hold = holdout["promo_flag"].to_numpy() == 1

    basecols = holdout[KEY + ["price", "promo_flag", "promo_cost"]].copy()
    blocks = []
    for sel in selections:
        focal = {str(p) for p in sel["focal_products"]}
        for direction, s in sel["directions"]:
            iid = f"sweep_{sel['rule']}_{direction}"
            b = basecols.copy()
            active = b["product_id"].isin(focal).to_numpy() & promo_hold
            p0 = b["price"].to_numpy(float)
            hi = np.maximum(holdout_reg, p0)
            lo = np.minimum((1.0 - MARGIN_CEILING) * holdout_reg, p0)
            new = np.round(np.clip(p0 * (1.0 + s), lo, hi), 3)
            moved = active & (np.sign(new - p0) == np.sign(s)) & (np.abs(new - p0) >= 0.0005)
            b["intervention_id"] = iid
            b["baseline_price"] = b["price"]
            b["intervention_price"] = np.where(moved, new, p0)
            blocks.append(b[["intervention_id", "product_id", "store_id", "week", "baseline_price",
                             "intervention_price", "promo_cost"]])
    panel = pd.concat(blocks, ignore_index=True)

    pub = out / "public"
    train[train_cols].to_csv(pub / "transactions_train_public.csv", **CSV)
    holdout.to_csv(pub / "transactions_holdout_context_public.csv", **CSV)
    panel.to_csv(pub / "counterfactual_sweep_context_panel.csv", **CSV)
    products_public.to_csv(pub / "products_public.csv", **CSV)
    stores_public.drop(columns="market").to_csv(pub / "stores_public.csv", **CSV)
    test[KEY + ["units"]].rename(columns={"units": "q"}).assign(q_realized=lambda d: d["q"]).to_csv(
        out / "ground_truth" / "ground_truth_forecast.csv", **CSV)
