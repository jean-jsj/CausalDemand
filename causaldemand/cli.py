from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import re
import sys
from pathlib import Path

from causaldemand import __version__
from causaldemand.download import OUT, DiskSpaceError
from causaldemand.names import CATEGORIES, CATEGORY_TITLES, CATEGORY_WORDS, UnknownNameError, check_category, select
from causaldemand.sources import SourceError, make_source


def _n(category: str, **flags) -> int:
    return len(select(category, **flags))


def _category_lines() -> str:
    rows = []
    for c in CATEGORY_WORDS:
        n = _n(c)
        title = "every dataset (download and rescore only)" if c == "all" else CATEGORY_TITLES[c]
        rows.append(f"  {c:<15} {title:<50} {n:>2} {'dataset' if n == 1 else 'datasets'}")
    return "\n".join(rows)


def _tissue_lines() -> str:
    dose = _n("tissue", tissue_dose=True) - _n("tissue")
    switching = _n("tissue", tissue_switching=True) - _n("tissue")
    return (f"  --tissue-dose       adds {dose} datasets with confounding strengths 0.15 and 0.45\n"
            f"  --tissue-switching  adds {switching} datasets with household-switching product distances")


USAGE = """causaldemand download CATEGORY [--tissue-dose] [--tissue-switching]
       causaldemand score CATEGORY PREDICTIONS_DIR [--tissue-dose] [--tissue-switching]
       causaldemand rescore CATEGORY [--tissue-dose] [--tissue-switching]
       causaldemand rerun CATEGORY MODEL [--tissue-dose] [--tissue-switching]
       causaldemand help"""

DESCRIPTION = f"""\
CausalDemand: retail sales datasets for testing whether a method predicts how sales respond to a change in
price. Every dataset has 140 weeks of training sales, 16 evaluation weeks and 32 price scenarios (the price of
one product or one brand raised or lowered by 10%). The synthetic datasets combine two demand models
(log-linear and discrete-choice) with confounding off and on (on: stores discount more when an unobserved
demand shock is high, so prices and sales move together for a reason the data do not show), each at five
seeds (1, 10, 20, 30, 40); scores are means over the seeds. Cereal and snack crackers are observed sales of
Dominick's Finer Foods.

commands:
  download   write the datasets of CATEGORY as CSV files
  score      score the predictions in PREDICTIONS_DIR on every dataset of CATEGORY ("causaldemand score
             --help" says what to write)
  rescore    re-score the reference models on CATEGORY and check every score against the archived scores
  rerun      refit the reference model MODEL on every downloaded dataset of CATEGORY with the paper's settings
             and check its predictions against the released runs
  help       show this page

CATEGORY is one of:
{_category_lines()}

with tissue only:
{_tissue_lines()}

MODEL is one of:
  lgbm, xgb, rf, dml, hier   LightGBM, XGBoost, random forest, Double ML (IV), hierarchical linear (IV)
  tabpfn, tabfm, chronos2    TabPFN, TabFM, Chronos-2 (Linux with a CUDA GPU; TabPFN needs a TABPFN_TOKEN)

steps:
  1. causaldemand download tissue
  2. fit your method on each dataset and write its predictions, as "causaldemand score --help" describes
  3. causaldemand score tissue my_method
"""

EPILOG = """\
note:
  Cereal and snack crackers are not in the release: download fetches the original files from the Kilts Center
  website (no login), builds each dataset on this computer (under a minute, about 4 GB of memory) and checks
  it against the files the reference models used. The Dominick's data are for academic research only, and
  publications must acknowledge the Kilts Center.

  The released files come from the Hugging Face dataset jean-jsj/CausalDemand. Accept its access conditions
  with a free Hugging Face account on the dataset page, then set HF_TOKEN to an access token of that account.
  score, rescore and rerun also read from it (answer keys, archived scores, run records), so they need
  HF_TOKEN too.

  rescore prints the paper's results of a category and saves them to causaldemand_rescored/<category>/.

  rerun runs the model in the Python environment of causaldemand, which needs Python 3.12 or later and the
  model's requirements: models/requirements.txt, or models/requirements-fm.txt for the foundation models,
  both installed with the package (rerun prints the full path if a requirement is missing; pip install -r
  that path). On macOS, LightGBM and XGBoost also need libomp (brew install libomp). The released CPU runs
  used Python 3.14, the foundation-model runs Python 3.12 on Linux with an NVIDIA A10 or A10G GPU; rerun of
  a foundation model stops when torch finds no CUDA GPU. A category takes hours. rerun saves the runs to
  causaldemand_reruns/ and continues an interrupted rerun. Each released run record names the computer the
  run was made on (Linux x86_64 or macOS arm64). On another kind of computer a few predictions can differ
  slightly, and LightGBM and random forest can differ slightly even on the same one; rerun then lists them
  and exits with status 1.
"""

OPTION_HINTS = {
    "--seed": "Commands act on every seed of a category; the tables report the mean over the five seeds.",
    "--demand-model": "Commands act on both demand models of a category.",
    "--confounding": "Commands act on confounding off and on; --tissue-dose adds the tissue datasets with "
                     "confounding strengths 0.15 and 0.45.",
    "--dose-level": "The tissue datasets with confounding strengths 0.15 and 0.45 are selected with --tissue-dose.",
    "--category": "The category is the first word after the command, for example: causaldemand download yogurt.",
    "--runs": "rescore takes a category, for example: causaldemand rescore tissue.",
    "--model": "rescore re-scores every reference model of a category.",
    "--compare-with": "rescore always compares its scores with the archived scores in the release.",
    "--forecast": "score takes a category and a folder with one folder per dataset, each holding forecast.csv "
                  "and scenarios.csv: causaldemand score tissue PREDICTIONS_DIR.",
    "--scenarios": "score takes a category and a folder with one folder per dataset, each holding forecast.csv "
                   "and scenarios.csv: causaldemand score tissue PREDICTIONS_DIR.",
    "--dataset": "Commands take a category and act on all its datasets, for example: causaldemand score tissue "
                 "PREDICTIONS_DIR.",
    "--json": "score saves its results, summary.json among them, to causaldemand_scores/<method>/<category>/.",
    "--name": "score names the method after the folder PREDICTIONS_DIR.",
    "--ground-truth": "--ground-truth was removed: score and rescore fetch the ground truth themselves.",
}
COMMAND_OPTION_HINTS = {
    ("download", "--out"): f"--out was removed: download always writes to {OUT}/ in the working folder.",
    ("download", "--dry-run"): "--dry-run was removed: download checks the free disk space before writing and stops "
                               "if it is too little.",
    ("download", "--overwrite"): "--overwrite was removed: download keeps each file of an earlier download whose "
                                 "SHA-256 still matches and writes the others again.",
    ("score", "--out"): "score saves its results to causaldemand_scores/<method>/<category>/.",
}
ONE_CATEGORY = ("download takes one category per command; run it once for each category, or once with all for "
                "every dataset.")
REMOVED_WITH_VALUE = ("--out",)
COMMAND_HINTS = {
    "elasticity": "The elasticity diagnostic is part of score: put elasticities.csv next to forecast.csv and "
                  "scenarios.csv in each dataset folder.",
}


class _Formatter(argparse.RawDescriptionHelpFormatter):
    def __init__(self, prog):
        super().__init__(prog, width=110, max_help_position=30)


def _second_category(words) -> bool:
    after_value_option = False
    for w in words:
        if w in CATEGORY_WORDS and not after_value_option:
            return True
        after_value_option = w in REMOVED_WITH_VALUE
    return False


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        hints = []
        if "unrecognized arguments" in message:
            words = message.split("unrecognized arguments:", 1)[1].split()
            given = re.findall(r"(--[a-z][a-z-]*)", " ".join(words))
            command = self.prog.split()[-1]
            hints = [OPTION_HINTS[o] for o in dict.fromkeys(given) if o in OPTION_HINTS]
            hints += [COMMAND_OPTION_HINTS[(command, o)] for o in dict.fromkeys(given)
                      if (command, o) in COMMAND_OPTION_HINTS]
            if command == "download" and _second_category(words):
                hints.append(ONE_CATEGORY)
        m = re.search(r"invalid choice: '([^']*)'", message)
        if m and m.group(1) in COMMAND_HINTS:
            hints.append(COMMAND_HINTS[m.group(1)])
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: {message}" + "".join(f"\n{h}" for h in dict.fromkeys(hints)) + "\n")


def _tissue_options(p: argparse.ArgumentParser, verb: str) -> None:
    p.add_argument("--tissue-dose", action="store_true",
                   help=f"tissue only: also {verb} the 20 datasets with confounding strengths 0.15 and 0.45 (the "
                        "main datasets have 0 and 0.30)")
    p.add_argument("--tissue-switching", action="store_true",
                   help=f"tissue only: also {verb} the 20 datasets whose product distances come from household "
                        "switching instead of product text")


def _subparser(sub, name: str, help_text: str, description: str, usage: str):
    return sub.add_parser(name, help=help_text, description=description, usage=usage,
                          formatter_class=_Formatter)


def _parser() -> argparse.ArgumentParser:
    ap = _Parser(prog="causaldemand", usage=USAGE, description=DESCRIPTION, epilog=EPILOG,
                 formatter_class=_Formatter)
    ap.add_argument("--version", action="version", version=f"causaldemand {__version__}")
    sub = ap.add_subparsers(dest="command", metavar="command", help=argparse.SUPPRESS, parser_class=_Parser,
                            prog="causaldemand")

    d = _subparser(sub, "download", "write datasets as CSV files", usage=(
        "causaldemand download CATEGORY [--tissue-dose] [--tissue-switching]"), description=f"""\
Download every dataset of one category to {OUT}/. "causaldemand help" lists the categories.
""")
    d.add_argument("category", metavar="CATEGORY", help="tissue, yogurt, cereal, snack-crackers or all")
    _tissue_options(d, "download")

    s = sub.add_parser("score", help="score a method's predictions", formatter_class=_Formatter, usage=(
        "causaldemand score CATEGORY PREDICTIONS_DIR [--tissue-dose] [--tissue-switching]"),
        description="Score one method on every dataset of one category, next to the reference models.",
        epilog="""\
examples:
  causaldemand score tissue my_method
  causaldemand score tissue my_method --tissue-dose --tissue-switching
  causaldemand score cereal my_method

input files in public/:
  products_public.csv                      product_id, brand_code, product_text (tissue and --tissue-dose
                                           datasets only; cereal, snack crackers: brand_code is a brand name)
  stores_public.csv                        store_id, chain, household_count
  transactions_train_public.csv            product_id, store_id, week, units, dollars, price, promo_flag,
                                           promo_cost, supply_cost_proxy; one row per product, store and week
                                           with a sale (no row: no sales)
  transactions_holdout_context_public.csv  the training columns without units and dollars, for every
                                           evaluation week of every product-store pair that sold in training
                                           (cereal, snack crackers: only the evaluation weeks with a sale)
  counterfactual_sweep_context_panel.csv   intervention_id, product_id, store_id, week, baseline_price,
                                           intervention_price, promo_cost; every holdout row under each of the
                                           32 scenarios (a 10% price rise or cut for a top product or brand)
  seasonality_index_public.csv             week, seasonality_index (discrete-choice datasets only)

  price               price paid, after any discount
  promo_flag          1 in a promoted week
  promo_cost          instrumental variable that sets the depth of a promotion's discount
  supply_cost_proxy   instrumental variable that moves the regular price
  household_count     households around the store (cereal, snack crackers: the store's weekly sales volume)

what to write:
  PREDICTIONS_DIR holds one folder per dataset of the category, named like the dataset folders that download
  writes in causaldemand_data/, with two files from the same fitted model:
  forecast.csv      product_id, store_id, week, predicted_units, for every row of
                    transactions_holdout_context_public.csv
  scenarios.csv     intervention_id, product_id, store_id, week, predicted_delta_units, for every scenario
                    row whose intervention_price differs from baseline_price: the predicted units at the
                    scenario price minus the predicted units at the actual price
  elasticities.csv  optional, tissue and yogurt only: priced_product_id, affected_product_id, elasticity
                    (a supplementary diagnostic, never used to rank methods)

  Files may be gzip-compressed (.csv.gz), and extra columns and the folders of other categories are ignored.
  The ground truth is fetched from the release (with HF_TOKEN, as for download); a method never needs it.

output:
  The tables are printed and saved to causaldemand_scores/<method>/<category>/ (table.md, table.csv,
  per_dataset.csv, per_scenario.csv, summary.json), where <method> is the name of PREDICTIONS_DIR; a new run
  of the same category replaces them, whatever its options. With --tissue-dose the figure is saved there as
  confounding_strength.pdf.

  The scores (counterfactual WMAPE and bias, forecast WMAPE against expected units, and for cereal and
  snack crackers the share of price rises with predicted lower sales) and the validity rules are defined in
  section 9 of the datasheet on Hugging Face.

  exit status: 0 every score valid, 1 a score is invalid (its scores are saved, but no table is built), a
  folder or file is missing or cannot be read, or cereal or snack crackers cannot be built from the Kilts
  Center files, 2 the command line is wrong.
""")
    s.add_argument("category", metavar="CATEGORY", help="tissue, yogurt, cereal or snack-crackers")
    s.add_argument("predictions", metavar="PREDICTIONS_DIR",
                   help="folder with one folder of predictions per dataset; its name labels the method")
    _tissue_options(s, "score")

    r = _subparser(sub, "rescore", "re-score the reference models and check the archived scores", usage=(
        "causaldemand rescore CATEGORY [--tissue-dose] [--tissue-switching]"), description="""\
Re-score the released predictions of the reference models of one category and check every score against
the archived scores. "causaldemand help" lists the categories.
""")
    r.add_argument("category", nargs="?", metavar="CATEGORY", help=", ".join(CATEGORY_WORDS))
    _tissue_options(r, "re-score")

    rr = _subparser(sub, "rerun", "refit a reference model and check its released predictions", usage=(
        "causaldemand rerun CATEGORY MODEL [--tissue-dose] [--tissue-switching]"), description="""\
Refit one reference model on every downloaded dataset of one category, with the settings of the paper, and
check its predictions against the released runs. On another kind of computer than the released run's, a few
predictions can differ slightly (exit status 1). "causaldemand help" lists the categories and the models.
""")
    rr.add_argument("category", metavar="CATEGORY", help="tissue, yogurt, cereal or snack-crackers")
    rr.add_argument("model", metavar="MODEL", help="lgbm, xgb, rf, dml, hier, tabpfn, tabfm or chronos2")
    _tissue_options(rr, "rerun")

    sub.add_parser("help", help="show the overview", usage="causaldemand help",
                   description='Show the overview, the same as "causaldemand --help".', formatter_class=_Formatter)

    return ap


def _source():
    return make_source(None)


def _ignored_with_all(a, categories) -> None:
    if "all" in categories:
        flags = [f for f, on in (("--tissue-dose", a.tissue_dose), ("--tissue-switching", a.tissue_switching)) if on]
        if flags:
            print(f"note: all holds every dataset already; {' and '.join(flags)} "
                  f"{'are' if len(flags) == 2 else 'is'} ignored.", file=sys.stderr)


class NotAvailable(RuntimeError):
    pass


def _function(module: str, name: str):
    full = f"causaldemand.{module}"
    if importlib.util.find_spec(full) is None:
        raise NotAvailable
    fn = getattr(importlib.import_module(full), name, None)
    if fn is None:
        raise NotAvailable
    return fn


def _run(fn, *args) -> int:
    try:
        status = fn(*args)
    except NotImplementedError:
        raise NotAvailable from None
    return 0 if status is None else int(status)


def cmd_download(a) -> int:
    from causaldemand.download import download
    datasets = select(a.category, a.tissue_dose, a.tissue_switching)
    _ignored_with_all(a, [a.category])
    download(_source(), datasets)
    return 0


def cmd_score(a) -> int:
    if a.category == "all":
        raise UnknownNameError(f"score takes one category per command: {', '.join(CATEGORIES)}.")
    check_category(a.category, CATEGORIES)
    select(a.category, a.tissue_dose, a.tissue_switching)
    predictions = Path(os.path.abspath(a.predictions))
    if not predictions.name:
        raise UnknownNameError(f"{a.predictions} cannot name a method; give the folder of the predictions by its "
                               "name, for example: causaldemand score tissue my_method")
    source = _source()
    fn = _function("scoring", "run_score")
    return _run(fn, a.category, predictions, a.tissue_dose, a.tissue_switching, source)


def cmd_rescore(a) -> int:
    if a.category is None:
        raise UnknownNameError(
            "name the category to re-score: tissue (the main table; --tissue-dose adds the confounding-strength "
            "figure, --tissue-switching the table of the household-switching datasets), yogurt, cereal, "
            "snack-crackers, or all (every table, written to a file). Example: causaldemand rescore tissue")
    select(a.category, a.tissue_dose, a.tissue_switching)
    _ignored_with_all(a, [a.category])
    tissue_dose, tissue_switching = (False, False) if a.category == "all" else (a.tissue_dose, a.tissue_switching)
    source = _source()
    fn = _function("rescore", "run_rescore")
    return _run(fn, a.category, tissue_dose, tissue_switching, source)


def cmd_rerun(a) -> int:
    if a.category == "all":
        raise UnknownNameError(f"rerun takes one category per command: {', '.join(CATEGORIES)}.")
    check_category(a.category, CATEGORIES)
    select(a.category, a.tissue_dose, a.tissue_switching)
    fn = _function("rerun", "run_rerun")
    return _run(fn, a.category, a.model, a.tissue_dose, a.tissue_switching, _source())


def cmd_help(a) -> int:
    _parser().print_help()
    return 0


COMMANDS = {"download": cmd_download, "score": cmd_score, "rescore": cmd_rescore, "rerun": cmd_rerun,
            "help": cmd_help}


def _raised_by_package(exc: BaseException) -> bool:
    return type(exc).__module__.split(".")[0] == "causaldemand"


def _subparsers(parser: argparse.ArgumentParser) -> dict:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    a, extra = parser.parse_known_args(argv)
    if a.command is None:
        if extra:
            parser.error(f"unrecognized arguments: {' '.join(extra)}")
        parser.print_help()
        return 2
    if extra:
        _subparsers(parser)[a.command].error(f"unrecognized arguments: {' '.join(extra)}")
    prefix = f"causaldemand {a.command}"
    try:
        return COMMANDS[a.command](a)
    except UnknownNameError as exc:
        print(f"{prefix}: {exc}", file=sys.stderr)
        return 2
    except NotAvailable:
        print(f"{prefix}: not available in this build", file=sys.stderr)
        return 1
    except (SourceError, DiskSpaceError, OSError) as exc:
        print(f"{prefix}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        if not _raised_by_package(exc):
            raise
        print(f"{prefix}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
