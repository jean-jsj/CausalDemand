"""Score one submission tree against every available benchmark cell.

Usage:
    python3 -m causaldemand.evaluate_all \
        --cells-root benchmark/dev/ \ --submissions-root my_model/ \ [--submission-name my_model] [--out-dir scores/] \ [--cells 'complex_*_endogenous*' ...] \ [--reference-scores benchmark/reference_scores/] \ [--dump-values-dir dumps/] [--format table|markdown]

Multi-cell convenience wrapper around `causaldemand.evaluate_submission`: discovers cell directories under `--cells-root`, pairs each with the submission subdirectory of the same cell slug (`<submissions-root>/<cell_slug>/`, the layout SUBMISSION_FORMAT.md prescribes), scores every pair, writes one `scores.json` per cell into `--out-dir`, and prints a combined per-cell-type leaderboard at the end.

Cells whose hidden truth is absent (the eval seeds in the release packaging — truth is maintainer-only) are skipped with a notice, as are cells without a matching submission subdirectory. A scoring error in one cell does not stop the others; it is reported and reflected in the exit code.

Actual-data arm. SYNTHETIC cells score sales forecasting/2/3 and are the ranked leaderboard. The ACTUAL-data arm (real POS panel) scores sales forecasting + validity checks only — it is PUBLIC-ONLY by design (no hidden truth EVER exists on real data), so it must NOT be caught by the "hidden truth absent → skip" branch that withholds eval-seed synthetic cells. Because an actual cell is a PRE-BUILT dict (`causaldemand.actual_data.load_actual_cell(data_root)`), not a synthetic cell dir discoverable under `--cells-root`, it is routed through `evaluate_prebuilt` from a separate `--actual-data-root`, NOT through `discover_cells`. The actual arm is a DIAGNOSTIC PANEL in its own `(cell_type, data_arm)` leaderboard partition — never ranked into the synthetic own-price-WMPE headline.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path
from typing import Any

from causaldemand.actual_data import ActualDataNotAvailable, load_actual_cell
from causaldemand.evaluate_submission import evaluate, evaluate_prebuilt
from causaldemand.leaderboard import leaderboard_rows, to_markdown


def discover_cells(cells_root: Path, patterns: list[str] | None) -> list[Path]:
    """Cell directories under `cells_root` (those carrying a scoring config).

    Released cells carry `release/scoring_params.json`; maintainer-side full outputs carry `reports/run_config_resolved.json`. Either marks a cell.
    """
    cells = sorted(
        path
        for path in cells_root.iterdir()
        if path.is_dir()
        and (
            (path / "release" / "scoring_params.json").exists()
            or (path / "reports" / "run_config_resolved.json").exists()
        )
    )
    if patterns:
        cells = [c for c in cells if any(fnmatch.fnmatch(c.name, p) for p in patterns)]
    return cells


def evaluate_all(
    cells_root: Path,
    submissions_root: Path,
    submission_name: str,
    out_dir: Path,
    patterns: list[str] | None = None,
    dump_values_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], int]:
    """Score every (cell, submission) pair found.

    Returns `(score_payloads, skipped, n_errors)` where `skipped` carries one `{cell, reason}` record per unscored cell.
    """
    payloads: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    n_errors = 0
    cells = discover_cells(cells_root, patterns)
    if not cells:
        print(f"no cell directories found under {cells_root}", file=sys.stderr)
    for cell_dir in cells:
        slug = cell_dir.name
        submission_dir = submissions_root / slug
        if not submission_dir.is_dir():
            skipped.append({"cell": slug, "reason": "no submission subdirectory"})
            continue
        if not (cell_dir / "hidden" / "transactions_full_hidden.csv").exists():
            # Release packaging: eval-seed cells ship public/ only; their truth is maintainer-private. Local scoring is dev-cell-only by design.
            skipped.append({"cell": slug, "reason": "hidden truth absent (eval cell?)"})
            continue
        dump_values = (dump_values_dir / f"{submission_name}__{slug}.csv") if dump_values_dir else None
        try:
            scores = evaluate(cell_dir, submission_dir, submission_name, dump_values=dump_values)
        except Exception as exc:  # one broken cell must not sink the run
            n_errors += 1
            skipped.append({"cell": slug, "reason": f"scoring error: {exc}"})
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{submission_name}__{slug}.json"
        out_path.write_text(json.dumps(scores, indent=2, default=float), encoding="utf-8")
        headline = (scores.get("counterfactual_prediction") or {}).get("headline") or {}
        own_wmpe = (headline.get("own_price") or {}).get("own_price_wmpe")
        sub_wape = (headline.get("substitution") or {}).get("substitution_wape")
        own_str = f"{own_wmpe:+.4f}" if own_wmpe is not None else "n/a"
        sub_str = f"{sub_wape:.4f}" if sub_wape is not None else "n/a"
        print(f"scored {slug}: L3 own-WMPE = {own_str}  substitution-WAPE = {sub_str}  -> {out_path}")
        payloads.append(scores)
    return payloads, skipped, n_errors


def evaluate_actual_arm(
    actual_data_root: Path,
    submissions_root: Path,
    submission_name: str,
    out_dir: Path,
) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    """Score the ACTUAL-data arm (sales forecasting + validity checks only), never blanket-skipped.

    The actual cell is a PRE-BUILT dict (`load_actual_cell`) rather than a synthetic cell dir under `--cells-root`, so it is loaded here from `actual_data_root` and routed through `evaluate_prebuilt` — it must NOT pass through the synthetic "hidden truth absent → skip" branch, because real data has NO hidden truth by design. Returns `(scores, None)` on success or `(None, {cell, reason})` when the real panel is not present / scoring failed.
    """
    try:
        cell = load_actual_cell(actual_data_root)
    except ActualDataNotAvailable as exc:
        # Public-only actual arm: the Dominick's panel is not downloaded in this checkout. This is NOT the synthetic hidden-truth skip; report it distinctly so it never masquerades as an eval-seed synthetic cell.
        return None, {"cell": "actual", "reason": f"actual data not available: {exc}"}
    slug = cell.get("cell_slug", cell.get("family", "actual"))
    submission_dir = submissions_root / slug
    if not submission_dir.is_dir():
        submission_dir = submissions_root / "actual"
    if not submission_dir.is_dir():
        return None, {"cell": slug, "reason": "no actual-arm submission subdirectory"}
    try:
        scores = evaluate_prebuilt(cell, submission_dir, submission_name)
    except Exception as exc:  # a broken actual cell must not sink the synthetic run
        return None, {"cell": slug, "reason": f"actual-arm scoring error: {exc}"}
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{submission_name}__{slug}__actual.json"
    out_path.write_text(json.dumps(scores, indent=2, default=float), encoding="utf-8")
    l4 = scores.get("validity_checks_actual") or {}
    own_sign = (l4.get("own_price_sign") or {}).get("frac_correct_sign")
    l1 = scores.get("sales_forecasting") or {}
    l1_wmape = l1.get("demand_wmape")
    own_str = f"{own_sign:.3f}" if own_sign is not None else "n/a"
    l1_str = f"{l1_wmape:.4f}" if l1_wmape is not None else "n/a"
    print(
        f"scored {slug} [actual arm]: L1 WMAPE = {l1_str}  L4 own-sign frac = {own_str}  -> {out_path}"
    )
    return scores, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-root", required=True, type=Path)
    parser.add_argument("--submissions-root", required=True, type=Path)
    parser.add_argument("--submission-name", default="submission")
    parser.add_argument("--out-dir", type=Path, default=Path("scores"))
    parser.add_argument(
        "--cells",
        nargs="*",
        default=None,
        help="Optional glob pattern(s) on cell directory names (e.g. 'complex_*_endogenous*').",
    )
    parser.add_argument(
        "--reference-scores",
        type=Path,
        default=None,
        help="Directory of reference scores.json files; reference rows are merged "
        "into the final table so each cell-type shows your placement.",
    )
    parser.add_argument(
        "--dump-values-dir",
        type=Path,
        default=None,
        help="Reserved; accepted and ignored.",
    )
    parser.add_argument(
        "--actual-data-root",
        type=Path,
        default=None,
        help="Optional Dominick's POS panel root (the Kilts category files, e.g. wtti.csv). When given, the actual-data "
        "arm (sales forecasting + validity checks) is scored via load_actual_cell -> evaluate_prebuilt "
        "and added as its OWN (cell_type, data_arm) diagnostic partition — NOT ranked "
        "into the synthetic own-price-WMPE headline. Absent panel is reported, "
        "never treated as a synthetic hidden-truth skip.",
    )
    parser.add_argument("--format", choices=["table", "markdown"], default="table")
    args = parser.parse_args()

    payloads, skipped, n_errors = evaluate_all(
        args.cells_root,
        args.submissions_root,
        args.submission_name,
        args.out_dir,
        patterns=args.cells,
        dump_values_dir=args.dump_values_dir,
    )

    # Actual-data arm: scored separately from the synthetic discovery loop so the "hidden truth absent -> skip" branch never fires on real data.
    if args.actual_data_root is not None:
        actual_scores, actual_skip = evaluate_actual_arm(
            args.actual_data_root,
            args.submissions_root,
            args.submission_name,
            args.out_dir,
        )
        if actual_scores is not None:
            payloads.append(actual_scores)
        if actual_skip is not None:
            skipped.append(actual_skip)

    for record in skipped:
        print(f"skipped {record['cell']}: {record['reason']}")

    table_payloads = list(payloads)
    if args.reference_scores is not None:
        # Merge reference rows, but guard the actual/synthetic arm split. A reference joins a cell ONLY when BOTH cell_slug AND data_arm match a scored cell — treating a missing data_arm as "synthetic" on both sides. When the actual arm coexists with the synthetic arm for the same cell_slug, a cell_slug-only match would splice a synthetic reference row into the actual panel (or vice-versa); the data_arm equality prevents it, keeping the actual arm its own clean (cell_type, data_arm) partition.
        def _key(p: dict[str, Any]) -> tuple[str, str]:
            return (
                p.get("cell_slug") or Path(p.get("cell_dir", "")).name,
                p.get("data_arm", "synthetic"),
            )

        scored_arms_by_slug: dict[str, set[str]] = {}
        for p in payloads:
            slug, arm = _key(p)
            scored_arms_by_slug.setdefault(slug, set()).add(arm)
        reference_payloads = [
            json.loads(path.read_text())
            for path in sorted(args.reference_scores.glob("*.json"))
        ]
        # Keep a reference unless its cell_slug is scored under a DIFFERENT arm only (i.e. the slug collides across arms and this reference's arm is absent from the scored set) — that is the cross-arm splice the guard exists to block.
        kept_references = [
            p
            for p in reference_payloads
            if (
                _key(p)[0] not in scored_arms_by_slug
                or _key(p)[1] in scored_arms_by_slug[_key(p)[0]]
            )
        ]
        table_payloads = kept_references + table_payloads
    if table_payloads:
        table = leaderboard_rows(table_payloads)
        print(f"\n=== leaderboard across {len(payloads)} scored cell(s) ===")
        print(to_markdown(table) if args.format == "markdown" else table.to_string(index=False))
    sys.exit(1 if n_errors else 0)


if __name__ == "__main__":
    main()
