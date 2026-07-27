"""Unit tests for the Dominick's movement-file loader (causaldemand.actual_data).

Hermetic: a tiny synthetic wtti.csv fixture in Dominick's raw schema — no real download. Covers the documented hygiene (OK/price filters), bundle-price arithmetic, promo flag, cost proxy, the 80%-revenue universe rule, and the train/held-out split.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from causaldemand.actual_data import (
    ACTUAL_ARM,
    ActualDataNotAvailable,
    DEFAULT_HOLDOUT_WEEKS,
    load_actual_cell,
)


def _write_fixture(root: Path, n_weeks: int = 40) -> Path:
    """Two dominant UPCs + one tail UPC across `n_weeks` weeks, one store."""
    rows = []
    for week in range(1, n_weeks + 1):
        # UPC 111: big seller, 2-unit bundle, on deal every 4th week.
        rows.append(
            dict(STORE=5, UPC=111, WEEK=week, MOVE=100, QTY=2, PRICE=4.0,
                 SALE="B" if week % 4 == 0 else "", PROFIT=25.0, OK=1)
        )
        # UPC 222: mid seller, single unit.
        rows.append(
            dict(STORE=5, UPC=222, WEEK=week, MOVE=50, QTY=1, PRICE=3.0,
                 SALE="", PROFIT=20.0, OK=1)
        )
        # UPC 333: tiny tail seller (outside the 80% universe).
        rows.append(
            dict(STORE=5, UPC=333, WEEK=week, MOVE=1, QTY=1, PRICE=1.0,
                 SALE="", PROFIT=10.0, OK=1)
        )
    # Hygiene rows: flagged + non-positive price — must be dropped.
    rows.append(dict(STORE=5, UPC=111, WEEK=1, MOVE=999, QTY=1, PRICE=4.0,
                     SALE="", PROFIT=25.0, OK=0))
    rows.append(dict(STORE=5, UPC=222, WEEK=1, MOVE=999, QTY=1, PRICE=0.0,
                     SALE="", PROFIT=20.0, OK=1))
    path = root / "wtti.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_missing_root_raises_actionable():
    with pytest.raises(ActualDataNotAvailable):
        load_actual_cell(Path("/nonexistent/dominicks/root"))


def test_loader_schema_split_and_universe(tmp_path: Path):
    _write_fixture(tmp_path)
    cell = load_actual_cell(tmp_path, window_weeks=40, holdout_weeks=8)

    assert cell["data_arm"] == ACTUAL_ARM and cell["family"] == ACTUAL_ARM
    panel = cell["transactions_full"]
    assert list(panel.columns) == [
        "product_id", "store_id", "week", "units", "dollars", "price",
        "promo_flag", "supply_cost_proxy",
    ]

    # 80%-revenue universe: UPC 111 (~72%) + UPC 222 (~27%) stay; 333 is cut.
    assert set(panel["product_id"]) == {"U111", "U222"}

    # Bundle arithmetic: unit price 4.0/2, revenue = unit price * MOVE. The hygiene rows (OK=0 / price<=0) must not have inflated week-1 units.
    week1_111 = panel[(panel["product_id"] == "U111") & (panel["week"] == 1)]
    assert week1_111["price"].iloc[0] == pytest.approx(2.0)
    assert week1_111["units"].iloc[0] == pytest.approx(100.0)
    assert week1_111["dollars"].iloc[0] == pytest.approx(200.0)
    # Cost proxy from the 25% gross margin.
    assert week1_111["supply_cost_proxy"].iloc[0] == pytest.approx(1.5)

    # Promo flag from the SALE deal code.
    p111 = panel[panel["product_id"] == "U111"].set_index("week")["promo_flag"]
    assert p111.loc[4] == 1 and p111.loc[1] == 0

    # Split: last 8 weeks held out, training excludes them.
    assert cell["eval_weeks"] == list(range(33, 41))
    assert set(cell["training"]["week"]) == set(range(1, 33))

    # cfg mirrors the synthetic contract keys the scorer reads.
    assert cell["cfg"]["simulation"]["counterfactual_eval_weeks"] == 8

    # Sweep context: both signs for every kept product, anchored on eval weeks.
    sweep = cell["sweep_context"]
    assert set(sweep["week"]) == set(range(33, 41))
    assert set(sweep["product_id"]) == {"U111", "U222"}
    ids = set(sweep["intervention_id"])
    assert any(i.endswith("_plus") for i in ids)
    assert any(i.endswith("_minus") for i in ids)


def test_default_holdout_matches_synthetic_release():
    assert DEFAULT_HOLDOUT_WEEKS == 16


def test_sparse_tail_anchors_window_end(tmp_path: Path):
    """Thin end-of-feed weeks must not anchor the window (well-covered rule)."""
    path = _write_fixture(tmp_path)
    # Append two trailing weeks with a single tiny row each (~1/3 of the 3-row median weekly count -> below the 50% threshold).
    thin = pd.DataFrame([
        dict(STORE=5, UPC=111, WEEK=41, MOVE=2, QTY=1, PRICE=2.0, SALE="",
             PROFIT=25.0, OK=1),
        dict(STORE=5, UPC=111, WEEK=42, MOVE=2, QTY=1, PRICE=2.0, SALE="",
             PROFIT=25.0, OK=1),
    ])
    thin.to_csv(path, mode="a", header=False, index=False)

    cell = load_actual_cell(tmp_path, window_weeks=40, holdout_weeks=8)
    # Window anchors at week 40 (last well-covered), not 42.
    assert max(cell["transactions_full"]["week"]) == 40
    assert cell["eval_weeks"] == list(range(33, 41))


def test_price_outlier_row_dropped(tmp_path: Path):
    """A single absurd-price row (recording error) must not survive hygiene."""
    path = _write_fixture(tmp_path)
    bad = pd.DataFrame([
        dict(STORE=5, UPC=111, WEEK=10, MOVE=43, QTY=1, PRICE=838.16, SALE="",
             PROFIT=25.0, OK=1),
    ])
    bad.to_csv(path, mode="a", header=False, index=False)

    cell = load_actual_cell(tmp_path, window_weeks=40, holdout_weeks=8)
    p = cell["transactions_full"]
    week10 = p[(p["product_id"] == "U111") & (p["week"] == 10)]
    # The legit week-10 row survives with its normal price; the 838.16 row is gone (it would otherwise raise the unit-averaged price far above 2.0).
    assert week10["price"].iloc[0] == pytest.approx(2.0)
