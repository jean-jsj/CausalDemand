"""Run the 2x2 reference grid on benchmark cells and write submission CSVs.

For every cell under ``--cells-root`` this fits the four (instrument x text)
reference corners for the cell's model family and writes one submission per
corner:

    <out-root>/<variant>/<cell_slug>/forecast_predictions.csv
    <out-root>/<variant>/<cell_slug>/elasticity_matrix.csv
    <out-root>/<variant>/<cell_slug>/counterfactual_deltas.csv

Variants: ``no_iv_no_text``, ``iv_no_text``, ``no_iv_text``, ``iv_text``.
Each variant directory is a complete submissions root scoreable with
``metrics/evaluate_all.py --submissions-root <out-root>/<variant>``.

Only the public cell surface is read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from causaldemand.baselines import loglog_grid, probit_simulation

VARIANTS: dict[str, dict[str, bool]] = {
    "no_iv_no_text": {"use_iv": False, "use_text": False},
    "iv_no_text": {"use_iv": True, "use_text": False},
    "no_iv_text": {"use_iv": False, "use_text": True},
    "iv_text": {"use_iv": True, "use_text": True},
}


def _load_public(cell_dir: Path) -> dict[str, pd.DataFrame]:
    public = cell_dir / "public"
    return {
        "transactions": pd.read_csv(public / "transactions_train_public.csv"),
        "products": pd.read_csv(public / "products_public.csv"),
        "stores": pd.read_csv(public / "stores_public.csv"),
        "holdout": pd.read_csv(public / "transactions_holdout_context_public.csv"),
        "sweep": pd.read_csv(public / "counterfactual_sweep_context_public.csv"),
    }


def _family(cell_dir: Path) -> str:
    if "covariance_probit" in cell_dir.name:
        return "covariance_probit"
    if "log_log" in cell_dir.name:
        return "log_log"
    raise ValueError(f"cannot infer family from cell dir name: {cell_dir.name}")


def _has_text(products: pd.DataFrame) -> bool:
    if "product_text" not in products.columns:
        return False
    return products["product_text"].fillna("").astype(str).str.strip().any()


def run_cell(cell_dir: Path, out_root: Path, variants: dict[str, dict[str, bool]]) -> None:
    frames = _load_public(cell_dir)
    family = _family(cell_dir)
    cell_has_text = _has_text(frames["products"])
    for variant, axes in variants.items():
        if axes["use_text"] and not cell_has_text:
            # Simple cells ship no product text; the text axis is undefined
            # there and the grid degenerates to the instrument pair.
            print(f"{cell_dir.name} [{variant}] skipped (no product text)", flush=True)
            continue
        started = time.perf_counter()
        if family == "log_log":
            params = loglog_grid.fit_loglog_corner(
                frames["transactions"], frames["products"], **axes
            )
            forecast = loglog_grid.predict_holdout_units(params, frames["holdout"])
            elasticity = loglog_grid.elasticity_matrix(params, frames["products"])
            deltas = loglog_grid.predict_sweep_deltas(params, frames["sweep"], frames["holdout"])
        else:
            params = probit_simulation.fit_probit_simulation(
                frames["transactions"], frames["products"], frames["stores"], **axes
            )
            forecast = probit_simulation.predict_holdout_units(params, frames["holdout"])
            elasticity = probit_simulation.elasticity_matrix(params, frames["products"])
            deltas = probit_simulation.predict_sweep_deltas(params, frames["sweep"])

        target = out_root / variant / cell_dir.name
        target.mkdir(parents=True, exist_ok=True)
        forecast.to_csv(target / "forecast_predictions.csv", index=False)
        elasticity.to_csv(target / "elasticity_matrix.csv", index=False)
        deltas.to_csv(target / "counterfactual_deltas.csv", index=False)
        meta = {
            "variant": variant,
            "family": family,
            "cell": cell_dir.name,
            "axes": axes,
            "fit_seconds": round(time.perf_counter() - started, 2),
        }
        if family == "covariance_probit":
            meta["b"] = params["b"]
            meta["rho"] = params["rho"]
            meta["price_scale"] = params["price_scale"]
        (target / "reference_fit_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"{cell_dir.name} [{variant}] done in {meta['fit_seconds']}s", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cells-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--cells",
        nargs="*",
        default=None,
        help="Optional subset of cell dir names (default: every dir with a public/ subdir)",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        choices=sorted(VARIANTS),
        help="Optional subset of variants (default: all four)",
    )
    args = parser.parse_args()

    cell_dirs = [
        d
        for d in sorted(args.cells_root.iterdir())
        if d.is_dir() and (d / "public").is_dir()
    ]
    if args.cells:
        cell_dirs = [d for d in cell_dirs if d.name in set(args.cells)]
    if not cell_dirs:
        print("no cells found", file=sys.stderr)
        return 2
    variants = (
        {name: VARIANTS[name] for name in args.variants} if args.variants else dict(VARIANTS)
    )
    for cell_dir in cell_dirs:
        run_cell(cell_dir, args.out_root, variants)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
