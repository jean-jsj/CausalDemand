"""Load the actual-data arm: Dominick's (Kilts Center) Bathroom Tissues.

Builds the real panel deterministically from the category files every
participant downloads from the Kilts Center, so all participants reconstruct
the identical cell (fixed 156-week window, 35-UPC universe, price-outlier
guard). Real data has no counterfactual truth: only sales forecasting and the
validity checks score on this arm.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Marker value tagging a cell dict as belonging to the actual-data arm. Routing branches on ``cell["data_arm"] == ACTUAL_ARM``; this module is its sole author.
ACTUAL_ARM = "actual"

#: Default contiguous window: ~3 yr, matched to the synthetic panel. NOT the full Dominick's span — that is an OPTIONAL robustness override (window_weeks=FULL_SPAN_WINDOW_WEEKS), never the default.
DEFAULT_WINDOW_WEEKS = 156

#: Full-span robustness override (documented non-default): 7 years of the Dominick's feed.
FULL_SPAN_WINDOW_WEEKS = 364

#: Default Dominick's category: Bathroom Tissues (movement file ``wtti.csv``, UPC file ``upctti.csv``) — the closest real analog to the synthetic facial-tissue category.
DEFAULT_CATEGORY = "tti"

#: Product-universe rule: smallest set of top-revenue UPCs covering this share of training-window revenue — mirrors the synthetic 80%-revenue SKU rule.
UNIVERSE_REVENUE_SHARE = 0.80

#: Well-covered-week rule for the default window anchor: the window ends at the LAST week whose post-hygiene row count is >= this fraction of the median weekly row count (guards the eval split against a sparse end-of-feed tail).
WELL_COVERED_FRACTION = 0.5

#: Price-outlier guard: drop rows whose unit price exceeds this multiple of the product's median unit price. Catches isolated recording errors (the bathroom-tissue feed has one row at 650x the product median) while never touching legitimate promo/pack variation.
PRICE_OUTLIER_FOLD = 10.0

#: Held-out eval-window length (weeks). Matches the released synthetic cells' ``counterfactual_eval_weeks`` (16 in the released config).
DEFAULT_HOLDOUT_WEEKS = 16

#: Own-price-sweep magnitude (both signs). ±10% mirrors the synthetic flagship ±X% headline scenario.
FIXTURE_SWEEP_PCT = 0.10

#: Exact column contract for ``sweep_context``. This is the public own-price-sweep table the ``validity_actual`` file is scored against; note ``baseline_units`` (needed to net dq) rather than the raw public CSV's ``promo_cost``.
SWEEP_CONTEXT_COLUMNS = [
    "intervention_id",
    "product_id",
    "store_id",
    "week",
    "baseline_price",
    "intervention_price",
    "baseline_units",
]


class ActualDataNotAvailable(FileNotFoundError):
    """Raised when the Dominick's category files are not present at ``data_root``.

    Subclasses :class:`FileNotFoundError` so existing ``except FileNotFoundError`` handlers still catch it, while callers that want the specific "real data not downloaded yet" case can catch this narrower type. Carries an actionable message naming what to point ``data_root`` at.
    """


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_actual_cell(
    data_root: Path,
    *,
    category: str = DEFAULT_CATEGORY,
    window_weeks: int = DEFAULT_WINDOW_WEEKS,  # ~3 yr, matched to synthetic panel
    window_end: int | None = None,  # default = latest week in the source
    holdout_weeks: int = DEFAULT_HOLDOUT_WEEKS,  # SAME split length as synthetic
) -> dict[str, Any]:
    """Load ONE actual-data cell from the Dominick's category files at ``data_root``.

    Reads ``w<category>.csv`` (movement) — download it from the Kilts Center — applies the standard Dominick's hygiene (drop ``OK == 0``, non-positive price), maps to the synthetic public-panel schema, restricts to the top-revenue universe (:data:`UNIVERSE_REVENUE_SHARE` of training-window revenue), takes the contiguous ``window_weeks`` window ending at ``window_end`` (default: latest week in the source), and applies the IDENTICAL split rule as the synthetic cells: ``eval_weeks`` = last ``holdout_weeks`` weeks of the window; ``training`` = the rest. sales forecasting fits on ``training``, forecasts ``eval_weeks``; validity checks places the ``sweep_context`` interventions OVER the ``eval_weeks`` store-weeks (same placement as the synthetic counterfactual prediction sweep).

    Returns the cell dict: ``{cfg, family="actual", transactions_full, training, eval_weeks, data_arm="actual", sweep_context}``.

    Raises
    ------
    ActualDataNotAvailable
        When ``data_root`` does not contain ``w<category>.csv``.
    """
    movement_path = _movement_path(data_root, category)
    if movement_path is None:
        raise ActualDataNotAvailable(
            f"Dominick's category files not found under data_root="
            f"{str(data_root)!r}: expected the movement file w{category}.csv "
            "(e.g. wtti.csv for Bathroom Tissues). Download the category from "
            "the Kilts Center (https://www.chicagobooth.edu/research/kilts/"
            "research-data/dominicks; free for academic research, attribution "
            "required) and point --actual-data-root at the directory holding "
            "it, or use build_fixture_actual_cell(<synthetic dev cell>) to "
            "exercise the actual-arm path without the real data."
        )

    panel = _load_movement(movement_path)

    # Contiguous window, anchored at the last WELL-COVERED week: an end-of-feed tail can be sparse, and "latest week" would then anchor the window (and the held-out eval weeks) on unreliable data. Default ``window_end`` = the last week whose post-hygiene row count is at least ``WELL_COVERED_FRACTION`` of the median weekly row count. (On the bathroom-tissue feed this is the literal last week — coverage is flat to the end — so the guard only bites on genuinely thin tails.)
    max_week = (
        _last_well_covered_week(panel) if window_end is None else int(window_end)
    )
    min_week = max_week - int(window_weeks) + 1
    panel = panel[(panel["week"] >= min_week) & (panel["week"] <= max_week)]
    if panel.empty:
        raise ActualDataNotAvailable(
            f"Dominick's movement file {movement_path.name} has no rows in the "
            f"requested window [{min_week}, {max_week}]."
        )

    weeks = sorted(int(w) for w in panel["week"].unique())
    eval_weeks = weeks[-int(holdout_weeks) :]
    training_mask = ~panel["week"].isin(set(eval_weeks))

    # Product universe: top-revenue UPCs covering UNIVERSE_REVENUE_SHARE of TRAINING-window revenue (selection never reads the held-out weeks).
    revenue = (
        panel[training_mask].groupby("product_id")["dollars"].sum().sort_values(ascending=False)
    )
    cum_share = revenue.cumsum() / revenue.sum()
    keep = cum_share[cum_share.shift(fill_value=0.0) < UNIVERSE_REVENUE_SHARE].index
    panel = panel[panel["product_id"].isin(set(keep))].reset_index(drop=True)

    transactions_full = panel
    training = panel[~panel["week"].isin(set(eval_weeks))]

    cfg = {
        "benchmark_version": f"dominicks-{category}",
        "benchmark_family": {"active_cell": {"model_family": ACTUAL_ARM}},
        "simulation": {"counterfactual_eval_weeks": int(holdout_weeks)},
    }

    sweep_context = _synthesize_sweep_context(
        synthetic_cell_dir=None,
        transactions_full=transactions_full,
        eval_weeks=eval_weeks,
    )

    return {
        "cfg": cfg,
        "family": ACTUAL_ARM,
        "transactions_full": transactions_full,
        "training": training,
        "eval_weeks": [int(w) for w in eval_weeks],
        "data_arm": ACTUAL_ARM,
        "sweep_context": sweep_context,
    }


def _last_well_covered_week(panel: pd.DataFrame) -> int:
    """Last week whose row count >= WELL_COVERED_FRACTION x median weekly rows.

    Deterministic sparse-tail guard for the default window anchor. If every trailing week is thin (degenerate input), falls back to the overall last week rather than failing.
    """
    weekly_rows = panel.groupby("week").size().sort_index()
    threshold = WELL_COVERED_FRACTION * float(weekly_rows.median())
    covered = weekly_rows[weekly_rows >= threshold]
    if covered.empty:
        return int(weekly_rows.index.max())
    return int(covered.index.max())


def _movement_path(data_root: Path, category: str) -> Path | None:
    """Path of the category movement CSV under ``data_root``, else None."""
    if data_root is None:
        return None
    root = Path(data_root)
    if not root.exists():
        return None
    for name in (f"w{category}.csv", f"w{category}.CSV", f"W{category}.csv"):
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def _load_movement(movement_path: Path) -> pd.DataFrame:
    """Read one Dominick's movement CSV into the synthetic public-panel schema.

    Standard hygiene: drop ``OK == 0`` and non-positive prices; unit price = ``PRICE/QTY``; revenue = ``PRICE*MOVE/QTY``; ``promo_flag`` = 1 iff the ``SALE`` deal code is non-blank; ``supply_cost_proxy`` = wholesale cost from the gross margin (``unit_price*(1-PROFIT/100)``). Rows are summed over duplicate (product, store, week) keys (price-line splits).
    """
    raw = pd.read_csv(movement_path)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    required = {"store", "upc", "week", "move", "qty", "price"}
    missing = required - set(raw.columns)
    if missing:
        raise ActualDataNotAvailable(
            f"{movement_path.name} is missing expected Dominick's movement "
            f"column(s): {sorted(missing)}. Found: {list(raw.columns)}."
        )

    if "ok" in raw.columns:
        raw = raw[raw["ok"] != 0]
    raw = raw[(raw["price"] > 0) & (raw["qty"] > 0) & (raw["move"] > 0)]

    unit_price = raw["price"] / raw["qty"]
    frame = pd.DataFrame(
        {
            # "U"-prefixed so ids stay strings through submission-CSV round trips (bare numeric UPCs would be re-read as int64 and break scoring merges; synthetic ids are alphanumeric for the same reason).
            "product_id": "U" + raw["upc"].astype("int64").astype(str),
            "store_id": "S" + raw["store"].astype("int64").astype(str).str.zfill(3),
            "week": raw["week"].astype(int),
            "units": raw["move"].astype(float),
            "dollars": (unit_price * raw["move"]).astype(float),
            "price": unit_price.astype(float),
            "promo_flag": _promo_flag(raw),
            "supply_cost_proxy": _cost_proxy(raw, unit_price),
        }
    )

    # Price-outlier guard: isolated recording errors (unit price many-fold the product's level) would poison the price series and the validity checks sweep baselines. Deterministic rule: drop rows above PRICE_OUTLIER_FOLD x the product's median unit price.
    med_price = frame.groupby("product_id")["price"].transform("median")
    frame = frame[frame["price"] <= PRICE_OUTLIER_FOLD * med_price]

    # Duplicate (product, store, week) rows (price-line splits): sum the mass, unit-average the per-unit columns.
    grouped = frame.groupby(["product_id", "store_id", "week"], as_index=False).agg(
        units=("units", "sum"),
        dollars=("dollars", "sum"),
        price=("price", "mean"),
        promo_flag=("promo_flag", "max"),
        supply_cost_proxy=("supply_cost_proxy", "mean"),
    )
    return grouped


def _promo_flag(raw: pd.DataFrame) -> pd.Series:
    """1 iff the Dominick's ``SALE`` deal code (B/C/S) is present."""
    if "sale" not in raw.columns:
        return pd.Series(0, index=raw.index, dtype=int)
    sale = raw["sale"].fillna("").astype(str).str.strip()
    return (sale != "").astype(int)


def _cost_proxy(raw: pd.DataFrame, unit_price: pd.Series) -> pd.Series:
    """Wholesale-cost proxy from the gross-margin ``PROFIT`` column (percent)."""
    if "profit" not in raw.columns:
        return pd.Series(float("nan"), index=raw.index)
    return (unit_price * (1.0 - raw["profit"].astype(float) / 100.0)).astype(float)


# ---------------------------------------------------------------------------
# Fixture builder (synthetic-derived actual cell; no real data required)
# ---------------------------------------------------------------------------


def build_fixture_actual_cell(synthetic_cell_dir: Path) -> dict[str, Any]:
    """Derive a VALID actual-arm cell from an EXISTING synthetic dev cell.

    Lets the actual-arm path (routing + integration tests) run WITHOUT the real data. Reuses ``_load_cell`` for the public panel, strips any hidden truth (the fixture keeps only the public/observed panel), tags the cell as the actual arm, and synthesizes ``sweep_context``.

    Parameters
    ----------
    synthetic_cell_dir:
        A synthetic dev cell directory (the same shape ``_load_cell`` consumes).

    Returns
    -------
    dict
        Keys ``{cfg, family, transactions_full, training, eval_weeks, data_arm, sweep_context}`` with ``family == "actual"`` and ``data_arm == "actual"``. ``sweep_context`` is the full own-price sweep: one intervention per product, BOTH signs, anchored on the ``eval_weeks`` store-weeks, columns exactly :data:`SWEEP_CONTEXT_COLUMNS`. No elasticity file is produced or expected (own-elasticity band is derived from dq downstream). No hidden truth is attached — validity checks is label-free.
    """
    # Import here (read-only) to avoid a hard module-load coupling and any import side effects at `import causaldemand.actual_data` time.
    from causaldemand.evaluate_submission import _load_cell

    synthetic_cell_dir = Path(synthetic_cell_dir)
    cell = _load_cell(synthetic_cell_dir)

    # Strip hidden truth: _load_cell already returns only the public-shaped five keys (no hidden/ truth object leaks into the dict); the observed panel it carries IS the public/observed panel for the actual arm. We keep exactly those, then overlay the actual-arm markers.
    transactions_full = cell["transactions_full"]
    training = cell["training"]
    eval_weeks = [int(w) for w in cell["eval_weeks"]]

    sweep_context = _synthesize_sweep_context(
        synthetic_cell_dir=synthetic_cell_dir,
        transactions_full=transactions_full,
        eval_weeks=eval_weeks,
    )

    return {
        "cfg": cell["cfg"],
        "family": ACTUAL_ARM,  # family echo = "actual" for the actual arm
        "transactions_full": transactions_full,
        "training": training,
        "eval_weeks": eval_weeks,
        "data_arm": ACTUAL_ARM,  # routing marker
        "sweep_context": sweep_context,
    }


def _synthesize_sweep_context(
    *,
    synthetic_cell_dir: Path | None,
    transactions_full: pd.DataFrame,
    eval_weeks: list[int],
) -> pd.DataFrame:
    """Build the public own-price sweep table from the observed panel + products.

    Full own-price sweep: ONE intervention per product, moved ONCE, BOTH signs (+X% and −X%), so validity checks monotonicity pairs exist and the derived own-elasticity band covers every product. Interventions are anchored on the ``eval_weeks`` store-weeks (the held-out tail) — the SAME placement as the synthetic counterfactual prediction sweep, keeping the arm leak-free and structurally identical.
    """
    products = _load_public_products(synthetic_cell_dir, transactions_full)

    eval_set = set(int(w) for w in eval_weeks)
    panel = transactions_full[transactions_full["week"].isin(eval_set)]

    # Observed baseline price + units per (product, store, week) on the held-out tail. If the panel is empty (degenerate fixture), fall back to the full panel restricted to nothing — the loop below simply yields no rows.
    price_col = "price" if "price" in panel.columns else None
    units_col = "units" if "units" in panel.columns else None

    rows: list[dict[str, Any]] = []
    for product_id in products:
        prod_panel = panel[panel["product_id"] == product_id]
        if prod_panel.empty:
            continue
        for sign, tag in ((+1.0, "plus"), (-1.0, "minus")):
            intervention_id = f"actual_own_sweep_{product_id}_{tag}"
            for _, r in prod_panel.iterrows():
                baseline_price = (
                    float(r[price_col]) if price_col is not None else float("nan")
                )
                baseline_units = (
                    float(r[units_col]) if units_col is not None else float("nan")
                )
                intervention_price = baseline_price * (1.0 + sign * FIXTURE_SWEEP_PCT)
                rows.append(
                    {
                        "intervention_id": intervention_id,
                        "product_id": product_id,
                        "store_id": r["store_id"],
                        "week": int(r["week"]),
                        "baseline_price": baseline_price,
                        "intervention_price": intervention_price,
                        "baseline_units": baseline_units,
                    }
                )

    return pd.DataFrame(rows, columns=SWEEP_CONTEXT_COLUMNS)


def _load_public_products(
    synthetic_cell_dir: Path | None, transactions_full: pd.DataFrame
) -> list[str]:
    """Product universe for the sweep — public products file, else the panel.

    Prefers ``public/products_public.csv`` (columns include ``product_id``); if it is absent (or no synthetic cell dir applies, as on the real Dominick's panel), falls back to the distinct products observed in the panel. Order is stable (sorted) for a deterministic sweep.
    """
    if synthetic_cell_dir is not None:
        products_path = Path(synthetic_cell_dir) / "public" / "products_public.csv"
        if products_path.exists():
            products_df = pd.read_csv(products_path)
            if "product_id" in products_df.columns:
                return sorted(products_df["product_id"].astype(str).unique().tolist())
    return sorted(transactions_full["product_id"].astype(str).unique().tolist())
