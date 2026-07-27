"""The `causaldemand` command: download data, score submissions, build the leaderboard.

    causaldemand quickstart                     # fetch the starter slice, score a naive submission
    causaldemand download --cell complex_log_log_endogenous_seed001 [--mini]
    causaldemand score --cell-dir ... --submission-dir ... --submission-name ... --out ...
    causaldemand score-all --cells-root ... --submissions-root ... --submission-name ... --out-dir ...
    causaldemand leaderboard scores/*.json
    causaldemand diagnostics scores/*.json --out diag.csv

`score`, `score-all`, `leaderboard`, and `diagnostics` accept the same options
as their modules (`python -m causaldemand.<name> --help`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ID = "jean-jsj/CausalDemand"
CELLS = [
    f"complex_{family}_{endo}_seed001"
    for family in ("log_log", "covariance_probit")
    for endo in ("exogenous", "endogenous")
]
MINI_CELLS = [c for c in CELLS if "log_log" in c]


def download(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="causaldemand download",
                                     description="Fetch one benchmark cell from Hugging Face.")
    parser.add_argument("--cell", choices=CELLS, required=True)
    parser.add_argument("--mini", action="store_true",
                        help="~18 MB 10-store starter slice instead of the full ~1 GB cell")
    parser.add_argument("--local-dir", type=Path, default=Path("benchmark"))
    args = parser.parse_args(argv)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("pip install huggingface_hub")
    if args.mini and args.cell not in MINI_CELLS:
        raise SystemExit(f"mini slices exist for {MINI_CELLS} only")

    tree = "dev_mini" if args.mini else "dev"
    snapshot_download(repo_id=REPO_ID, repo_type="dataset",
                      allow_patterns=[f"{tree}/{args.cell}/*"], local_dir=args.local_dir)
    print(f"done: {args.local_dir / tree / args.cell}")
    return 0


def quickstart(argv: list[str]) -> int:
    """Fetch the starter slice if needed, score a naive submission, print the headline."""
    import json
    import subprocess
    from itertools import product as iproduct

    import pandas as pd

    parser = argparse.ArgumentParser(prog="causaldemand quickstart",
                                     description=quickstart.__doc__)
    parser.add_argument("--cell-dir", type=Path, default=None,
                        help="a downloaded cell (default: the mini endogenous log-log cell, fetched if missing)")
    args = parser.parse_args(argv)

    cell_dir = args.cell_dir
    if cell_dir is None:
        cell_dir = Path("benchmark/dev_mini") / MINI_CELLS[1]
        if not cell_dir.is_dir():
            download(["--cell", MINI_CELLS[1], "--mini"])
    if not (cell_dir / "hidden").is_dir():
        raise SystemExit(f"{cell_dir} has no hidden/ truth — local scoring works on the dev seed only.")

    public = cell_dir / "public"
    train = pd.read_csv(public / "transactions_train_public.csv")
    holdout = pd.read_csv(public / "transactions_holdout_context_public.csv")
    products = pd.read_csv(public / "products_public.csv")

    sub_dir = Path("submissions_local/naive") / cell_dir.name
    sub_dir.mkdir(parents=True, exist_ok=True)

    recent = train[train["week"] > train["week"].max() - 8]
    mean_units = recent.groupby(["product_id", "store_id"])["units"].mean().rename("predicted_units")
    holdout[["product_id", "store_id", "week"]].merge(
        mean_units, on=["product_id", "store_id"], how="left"
    ).fillna({"predicted_units": 0.0}).to_csv(sub_dir / "forecast_predictions.csv", index=False)

    ids = products["product_id"].tolist()
    pd.DataFrame(
        [(j, i, 0.0) for j, i in iproduct(ids, ids)],
        columns=["priced_product_id", "affected_product_id", "elasticity"],
    ).to_csv(sub_dir / "elasticity_matrix.csv", index=False)

    pd.read_csv(
        public / "counterfactual_sweep_context_public.csv",
        usecols=["intervention_id", "product_id", "store_id", "week"],
    ).drop_duplicates().assign(predicted_delta_units=0.0).to_csv(
        sub_dir / "counterfactual_deltas.csv", index=False
    )

    out = Path("scores") / f"{cell_dir.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    code = subprocess.call(
        [sys.executable, "-m", "causaldemand.evaluate_submission",
         "--cell-dir", str(cell_dir), "--submission-dir", str(sub_dir),
         "--submission-name", "naive", "--out", str(out)]
    )
    if code:
        return code

    score = json.loads(out.read_text())
    headline = score["counterfactual_prediction"]["headline"]
    print(f"own-price bias  {headline['own_price']['own_price_wmpe']:+.2f}   (ranked headline; 0 = unbiased)")
    print(f"forecast error   {score['sales_forecasting']['demand_wmape']:.2f}   (displayed, never ranked)")
    return 0


def main() -> int:
    commands = {"download": download, "quickstart": quickstart}
    lazy = {"score": "evaluate_submission", "score-all": "evaluate_all",
            "leaderboard": "leaderboard", "diagnostics": "diagnostics"}
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        return 0
    cmd, argv = sys.argv[1], sys.argv[2:]
    if cmd in commands:
        return commands[cmd](argv)
    if cmd in lazy:
        import importlib

        module = importlib.import_module(f"causaldemand.{lazy[cmd]}")
        sys.argv = [f"causaldemand {cmd}"] + argv
        return module.main() or 0
    print(f"unknown command {cmd!r}; run `causaldemand --help`", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
