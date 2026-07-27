"""Unit tests for the causaldemand.evaluate_all multi-cell wrapper.

Discovery, slug-matching, eval-cell/missing-submission skips, and per-cell error isolation. Real-cell scoring goes through `evaluate` (tested against generated cells in the verification runs), so it is stubbed here.
"""

from __future__ import annotations

import json
from pathlib import Path

import causaldemand.evaluate_all as evaluate_all_module
from causaldemand.evaluate_all import discover_cells, evaluate_all


def _make_cell(root: Path, slug: str, with_truth: bool = True) -> Path:
    cell = root / slug
    (cell / "reports").mkdir(parents=True)
    (cell / "reports" / "run_config_resolved.json").write_text("{}", encoding="utf-8")
    if with_truth:
        (cell / "hidden").mkdir()
        (cell / "hidden" / "transactions_full_hidden.csv").write_text("", encoding="utf-8")
    return cell


def _canned_scores(slug: str) -> dict:
    return {
        "schema_version": 1,
        "submission_name": "m",
        "cell_slug": slug,
        "counterfactual_prediction": {
            "headline_scenario": {"distribution": {"headline": 0.5, "n_store_weeks": 10}}
        },
    }


def test_discover_cells_accepts_released_cells(tmp_path: Path) -> None:
    """A released cell carries release/scoring_params.json, no reports/ tree."""
    cell = tmp_path / "complex_log_log_endogenous_seed001"
    (cell / "release").mkdir(parents=True)
    (cell / "release" / "scoring_params.json").write_text("{}", encoding="utf-8")

    found = discover_cells(tmp_path, None)
    assert [c.name for c in found] == ["complex_log_log_endogenous_seed001"]


def test_discover_cells_requires_run_config_and_filters(tmp_path: Path) -> None:
    _make_cell(tmp_path, "complex_log_log_exogenous_seed001")
    _make_cell(tmp_path, "complex_log_log_endogenous_seed001")
    (tmp_path / "not_a_cell").mkdir()  # no reports/run_config_resolved.json

    found = discover_cells(tmp_path, None)
    assert [c.name for c in found] == [
        "complex_log_log_endogenous_seed001",
        "complex_log_log_exogenous_seed001",
    ]
    filtered = discover_cells(tmp_path, ["*_endogenous*"])
    assert [c.name for c in filtered] == ["complex_log_log_endogenous_seed001"]


def test_evaluate_all_scores_skips_and_writes(tmp_path: Path, monkeypatch) -> None:
    cells_root = tmp_path / "cells"
    _make_cell(cells_root, "cell_dev_seed001", with_truth=True)
    _make_cell(cells_root, "cell_eval_seed002", with_truth=False)  # truth private
    _make_cell(cells_root, "cell_unsubmitted_seed001", with_truth=True)

    submissions = tmp_path / "subs"
    (submissions / "cell_dev_seed001").mkdir(parents=True)
    (submissions / "cell_eval_seed002").mkdir(parents=True)
    # no subdir for cell_unsubmitted_seed001

    monkeypatch.setattr(
        evaluate_all_module,
        "evaluate",
        lambda cell_dir, sub_dir, name, dump_values=None: _canned_scores(cell_dir.name),
    )
    out_dir = tmp_path / "scores"
    payloads, skipped, n_errors = evaluate_all(cells_root, submissions, "m", out_dir)

    assert n_errors == 0
    assert [p["cell_slug"] for p in payloads] == ["cell_dev_seed001"]
    written = out_dir / "m__cell_dev_seed001.json"
    assert json.loads(written.read_text())["cell_slug"] == "cell_dev_seed001"
    reasons = {r["cell"]: r["reason"] for r in skipped}
    assert "hidden truth absent" in reasons["cell_eval_seed002"]
    assert reasons["cell_unsubmitted_seed001"] == "no submission subdirectory"


def test_evaluate_all_isolates_per_cell_errors(tmp_path: Path, monkeypatch) -> None:
    cells_root = tmp_path / "cells"
    _make_cell(cells_root, "cell_bad_seed001")
    _make_cell(cells_root, "cell_good_seed001")
    submissions = tmp_path / "subs"
    (submissions / "cell_bad_seed001").mkdir(parents=True)
    (submissions / "cell_good_seed001").mkdir(parents=True)

    def fake_evaluate(cell_dir, sub_dir, name, dump_values=None):
        if "bad" in cell_dir.name:
            raise ValueError("boom")
        return _canned_scores(cell_dir.name)

    monkeypatch.setattr(evaluate_all_module, "evaluate", fake_evaluate)
    payloads, skipped, n_errors = evaluate_all(
        cells_root, submissions, "m", tmp_path / "scores"
    )
    assert n_errors == 1
    assert [p["cell_slug"] for p in payloads] == ["cell_good_seed001"]
    assert any("scoring error: boom" in r["reason"] for r in skipped)
