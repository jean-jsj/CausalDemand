from __future__ import annotations

import csv
import io
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from causaldemand import __version__
from causaldemand import figure as F
from causaldemand import tables as T
from causaldemand._code import file_sha256, module_sha256, output_folder, panel_scorer
from causaldemand.download import DiskSpaceError, ground_truth_csv, show
from causaldemand.names import (REAL_DATASETS, Dataset, UnknownNameError, all_datasets, by_id, by_release_path,
                                count_text, select)
from causaldemand.sources import SourceError, make_source, read_sha256sums

FORECAST, SCENARIO = "layer1_predictions.csv.gz", "panel_deltas.csv.gz"
RECORD, TUNING = "manifest.json", "tuning.json"
OUTPUT_ROOT = "causaldemand_rescored"
TOLERANCE = 1e-12
CPU_MODELS = ("lgbm", "xgb", "rf", "dml", "hier")
TUNED_MODELS = ("lgbm", "xgb", "rf", "dml")
SAMPLED_MODELS = ("tabpfn", "tabfm")
FM_RUNS = tuple(f"{m}__sampler{k}" for m in SAMPLED_MODELS for k in range(3)) + ("chronos2",)
RUN_ORDER = CPU_MODELS + FM_RUNS
_SAMPLER_RUN = re.compile(r"^(tabpfn|tabfm)__sampler(\d+)$")
T975 = T.T975


class RescoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunSet:
    name: str
    variant: str
    models: tuple
    archived: str
    settings: str

    @property
    def levels(self) -> bool:
        return self.variant == "dose"

    @property
    def foundation(self) -> bool:
        return self.name.endswith("/foundation_models")

    def run_path(self, ds: Dataset, model: str) -> str:
        level = f"{ds.dose_level}/" if self.levels else ""
        return f"runs/{self.name}/{level}{ds.release_name}/{model}"

    def record_pattern(self) -> str:
        return f"runs/{self.name}/" + ("*/" if self.levels else "") + f"*/*/{RECORD}"

    def archived_labels(self, ds: Dataset) -> list[str]:
        if self.settings == "tuned":
            return [f"tuned__{ds.generator_name}", f"tuned__{ds.release_name}"]
        if self.levels:
            return [f"{ds.dose_level}__complex_{ds.family}_seed{ds.seed:03d}", f"{ds.dose_level}__{ds.release_name}"]
        return [ds.generator_name, ds.release_name]

    def comparisons(self, ds: Dataset) -> list[tuple[str, list[str]]]:
        out = [(self.archived, self.archived_labels(ds))]
        if self.name == "main/cpu_default":
            level = "delta0p30" if ds.confounding == "on" else "delta0p00"
            out.append(("dose_response/cpu", [f"{level}__complex_{ds.family}_seed{ds.seed:03d}"]))
        return out


RUN_SETS = {rs.name: rs for rs in (
    RunSet("main/cpu_default", "main", CPU_MODELS, "main/cpu_default", "default"),
    RunSet("main/cpu_tuned", "main", TUNED_MODELS, "main/cpu_tuned", "tuned"),
    RunSet("main/foundation_models", "main", FM_RUNS, "main/foundation_models", "default"),
    RunSet("dose_response/cpu", "dose", CPU_MODELS, "dose_response/cpu", "default"),
    RunSet("dose_response/foundation_models", "dose", FM_RUNS, "dose_response/foundation_models", "default"),
    RunSet("switching/cpu_default", "switching", CPU_MODELS, "switching/cpu_default", "default"),
    RunSet("switching/foundation_models", "switching", FM_RUNS, "switching/foundation_models", "default"),
    RunSet("yogurt/cpu_default", "yogurt", CPU_MODELS, "yogurt/default", "default"),
    RunSet("yogurt/cpu_tuned", "yogurt", TUNED_MODELS, "yogurt/tuned", "tuned"),
    RunSet("yogurt/foundation_models", "yogurt", FM_RUNS, "yogurt/foundation_models", "default"),
)}
DEFAULT_RUNS = (("cpu_default", CPU_MODELS), ("foundation_models", FM_RUNS))
SETTINGS_RUNS = DEFAULT_RUNS + (("cpu_tuned", TUNED_MODELS),)
REAL_ARMS = (("cpu_default", ("hier",)), ("cpu_tuned", TUNED_MODELS),
             ("foundation_models", tuple(f"{m}__sampler{k}" for m in SAMPLED_MODELS for k in range(5)) + ("chronos2",)))
REAL_NOT_RELEASED = tuple(f"cpu_default/{m} (no run folder)" for m in TUNED_MODELS)


def category_run_sets(category: str, tissue_dose: bool = False, tissue_switching: bool = False) -> dict[str, tuple]:
    select(category, tissue_dose, tissue_switching)
    if category == "all":
        return {**category_run_sets("tissue", True, True), **category_run_sets("yogurt")}
    if category == "tissue":
        out = {f"main/{arm}": models for arm, models in SETTINGS_RUNS}
        if tissue_dose:
            out["dose_response/cpu"] = CPU_MODELS
            out["dose_response/foundation_models"] = FM_RUNS
        if tissue_switching:
            out.update({f"switching/{arm}": models for arm, models in DEFAULT_RUNS})
        return out
    if category == "yogurt":
        return {f"yogurt/{arm}": models for arm, models in SETTINGS_RUNS}
    return {}


def real_datasets(category: str) -> list[Dataset]:
    return [ds for ds in select(category) if ds.is_real]


@dataclass(frozen=True)
class PlannedRun:
    run_set: RunSet
    dataset: Dataset
    model: str

    @property
    def path(self) -> str:
        return self.run_set.run_path(self.dataset, self.model)


def base_model(model: str) -> str:
    m = _SAMPLER_RUN.match(model)
    return m.group(1) if m else model


def _selection(run_sets) -> dict[str, tuple]:
    if isinstance(run_sets, str):
        run_sets = [run_sets]
    items = run_sets.items() if isinstance(run_sets, Mapping) else ((name, None) for name in run_sets)
    out = {}
    for name, models in items:
        if name not in RUN_SETS:
            raise RescoreError(f"{name!r} is not a run set; run sets: {', '.join(RUN_SETS)}")
        rs = RUN_SETS[name]
        models = tuple(models) if models else rs.models
        unknown = [m for m in models if m not in rs.models]
        if unknown:
            raise RescoreError(f"{name} has no runs of {', '.join(unknown)}; its runs: {', '.join(rs.models)}")
        out[name] = models
    return out


def _wanted_datasets(datasets) -> set | None:
    if datasets is None:
        return None
    out = set()
    for d in ([datasets] if isinstance(datasets, (str, Dataset)) else datasets):
        if isinstance(d, Dataset):
            out.add(d.release_path)
            continue
        try:
            out.add(by_id(d).release_path)
        except UnknownNameError:
            out.add(by_release_path(d).release_path)
    return out


def plan_runs(run_sets, datasets=None, models=None) -> list[PlannedRun]:
    selection = _selection(run_sets)
    wanted = _wanted_datasets(datasets)
    order = list(RUN_SETS)
    out = []
    for name, ms in selection.items():
        rs = RUN_SETS[name]
        for ds in all_datasets():
            if ds.variant != rs.variant or (wanted is not None and ds.release_path not in wanted):
                continue
            for model in ms:
                if models and model not in models and base_model(model) not in models:
                    continue
                out.append(PlannedRun(rs, ds, model))
    position = {ds: i for i, ds in enumerate(all_datasets())}
    out.sort(key=lambda r: (position[r.dataset], order.index(r.run_set.name), RUN_ORDER.index(r.model)))
    return out


def release_sums(source) -> dict[str, str] | None:
    try:
        root = source.fetch(["SHA256SUMS"])
    except SourceError:
        return None
    return read_sha256sums(root)


def _sums_state(sums: dict | None) -> str:
    return "absent" if sums is None else "checked"


SUMS_ABSENT_NOTE = ("The release has no SHA256SUMS, so the run records and the archived scores are not checked "
                    "against it; the prediction files are still checked against their run records.")


def _listed(sums: dict | None, rel: str, path: Path) -> str | None:
    if sums is None:
        return None
    if rel not in sums:
        return f"{rel} is not listed in SHA256SUMS"
    if file_sha256(path) != sums[rel]:
        return f"{rel}: SHA-256 differs from SHA256SUMS"
    return None


def check_run_record(run_dir: Path, rel: str, *, release_name: str, release_path: str, model: str,
                     foundation: bool, tuned: bool, sums: dict | None, scorer_sha256: str) -> tuple[list[str], dict]:
    run_dir = Path(run_dir)
    problems = []
    rpath = run_dir / RECORD
    if not rpath.is_file():
        return [f"{rel}: no run record ({RECORD})"], {}
    p = _listed(sums, f"{rel}/{RECORD}", rpath)
    if p:
        problems.append(p)
    try:
        record = json.loads(rpath.read_text(encoding="utf-8"))
    except ValueError as exc:
        return problems + [f"{rel}/{RECORD} cannot be read: {exc}"], {}
    if not isinstance(record, dict):
        return problems + [f"{rel}/{RECORD} is not a JSON object"], {}
    if record.get("dataset") != release_name or record.get("dataset_dir") != release_path:
        problems.append(f"{rel}: the run record names dataset {record.get('dataset')!r} in "
                        f"{record.get('dataset_dir')!r}, not {release_name!r} in {release_path!r}")
    field, base = ("model" if foundation else "model_key"), base_model(model)
    if record.get(field) != base:
        problems.append(f"{rel}: the run record names model {record.get(field)!r}, not {base!r}")
    sampler = _SAMPLER_RUN.match(model)
    if sampler and (record.get("context") or {}).get("sampler_seed") != int(sampler.group(2)):
        problems.append(f"{rel}: the run record names sampler seed "
                        f"{(record.get('context') or {}).get('sampler_seed')!r}, not {int(sampler.group(2))}")
    files = record.get("released_files") or {}
    for name in (FORECAST, SCENARIO) + ((TUNING,) if tuned else ()):
        entry = files.get(name) or {}
        if not entry.get("sha256"):
            problems.append(f"{rel}: the run record gives no SHA-256 of {name}")
        elif not (run_dir / name).is_file():
            problems.append(f"{rel}/{name}: no such file")
        elif file_sha256(run_dir / name) != entry["sha256"]:
            problems.append(f"{rel}/{name}: SHA-256 differs from the run record")
    scorer = record.get("scorer") or {}
    if scorer_sha256 not in (scorer.get("sha256"), scorer.get("scores_written_by_sha256")):
        problems.append(f"{rel}: the run record names the scorer {str(scorer.get('sha256'))[:12]}, not this "
                        f"package's metrics/panel_scorer.py ({scorer_sha256[:12]})")
    return problems, record


def _relative(a: float, b: float) -> float:
    if a == b or (a != a and b != b):
        return 0.0
    if a != a or b != b:
        return float("inf")
    return abs(a - b) / max(abs(a), abs(b))


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def compare_csv_text(text: str, archived: str, rel_tol: float = TOLERANCE, abs_tol: float = 0.0) -> dict:
    out = {"identical": text == archived, "within_tolerance": True, "max_relative_difference": 0.0,
           "max_absolute_difference": 0.0, "first_difference": None}
    if out["identical"]:
        return out
    a = list(csv.reader(io.StringIO(text)))
    b = list(csv.reader(io.StringIO(archived)))
    if len(a) != len(b) or (a and a[0] != b[0]):
        out.update(within_tolerance=False, max_relative_difference=float("inf"),
                   max_absolute_difference=float("inf"),
                   first_difference=f"{len(a)} rows against {len(b)}" if len(a) != len(b) else "the columns differ")
        return out
    header = a[0] if a else []
    for i, (ra, rb) in enumerate(zip(a, b)):
        if ra == rb:
            continue
        if len(ra) != len(rb):
            out.update(within_tolerance=False, first_difference=f"row {i}: {len(ra)} cells against {len(rb)}")
            out["max_relative_difference"] = out["max_absolute_difference"] = float("inf")
            return out
        for j, (x, y) in enumerate(zip(ra, rb)):
            if x == y:
                continue
            fx, fy = _number(x), _number(y)
            if fx is None or fy is None:
                rel = dif = float("inf")
            else:
                rel, dif = _relative(fx, fy), abs(fx - fy)
            out["max_relative_difference"] = max(out["max_relative_difference"], rel)
            out["max_absolute_difference"] = max(out["max_absolute_difference"], dif)
            if not (rel <= rel_tol or dif <= abs_tol) and out["within_tolerance"]:
                column = header[j] if j < len(header) else str(j)
                out.update(within_tolerance=False, first_difference=f"row {i}, column {column}: {x} against {y}")
    return out


class Archived:
    def __init__(self, folder: Path):
        folder = Path(folder)
        self.rows: dict[tuple[str, str, str], list[str]] = {}
        with open(folder / "per_scenario.csv", encoding="utf-8", newline="") as f:
            head = f.readline().rstrip("\n").split(",")
            if head[:3] != ["dataset", "model", "seed"]:
                raise RescoreError(f"{folder / 'per_scenario.csv'}: unexpected columns")
            self.header = ",".join(head[3:])
            for line in f:
                parts = line.rstrip("\n").split(",", 3)
                self.rows.setdefault(tuple(parts[:3]), []).append(parts[3])
        with open(folder / "per_run.csv", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            self.summary_columns = [c for c in reader.fieldnames or [] if c not in ("dataset", "model", "seed")]
            self.summaries = {(r["dataset"], r["model"], r["seed"]): r for r in reader}

    def compare(self, labels: list[str], model: str, seed: int, ps_csv: str, summary: dict) -> dict:
        key = next((k for k in ((label, model, str(seed)) for label in labels) if k in self.rows), None)
        if key is None:
            return {"archived": False, "ok": False}
        lines = ps_csv.rstrip("\n").split("\n")
        archived = self.rows[key]
        same = lines[0] == self.header and lines[1:] == archived
        worst = 0.0 if same else compare_csv_text("\n".join(lines) + "\n",
                                                  "\n".join([self.header] + archived) + "\n")["max_relative_difference"]
        row = self.summaries.get(key)
        if row is None:
            return {"archived": True, "archived_label": key[0], "per_scenario_identical": same,
                    "summary_identical": False, "max_relative_difference": worst, "ok": False,
                    "note": "the archived per_run.csv has no row of this run"}
        absent = [k for k in self.summary_columns if k not in summary]
        summary_same, summary_worst, words_equal = not absent, 0.0, True
        for k in self.summary_columns:
            if k in absent:
                continue
            v, text = summary[k], row[k]
            summary_same &= text == _csv_text(v)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                words_equal &= text == _csv_text(v)
                continue
            number = _number(text) if text != "" else float("nan")
            summary_worst = max(summary_worst, float("inf") if number is None else _relative(number, float(v)))
        worst = max(worst, summary_worst)
        out = {"archived": True, "archived_label": key[0], "per_scenario_identical": same,
               "summary_identical": summary_same, "max_relative_difference": worst,
               "ok": not absent and words_equal and worst <= TOLERANCE}
        if absent:
            out["summary_columns_not_scored"] = absent
        return out


def _csv_text(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return ""
    if isinstance(v, float):
        return repr(v)
    return str(v)


class _Archives:
    def __init__(self, source, sums: dict | None):
        self.source, self.sums = source, sums
        self.loaded: dict[str, Archived | None] = {}
        self.problems: list[str] = []

    def get(self, folder: str) -> Archived | None:
        if folder not in self.loaded:
            files = [f"results/{folder}/per_run.csv", f"results/{folder}/per_scenario.csv"]
            try:
                root = self.source.fetch(files)
                for rel in files:
                    p = _listed(self.sums, rel, root / rel)
                    if p:
                        self.problems.append(p)
                self.loaded[folder] = Archived(root / "results" / folder)
            except (SourceError, RescoreError, OSError) as exc:
                self.problems.append(f"results/{folder}: the archived scores cannot be read: {exc}")
                self.loaded[folder] = None
        return self.loaded[folder]


def no_change_error(P, key: pd.DataFrame) -> float:
    l1 = key[P.KEY].drop_duplicates().assign(predicted_units=1.0)
    deltas = key[["intervention_id"] + P.KEY].assign(predicted_delta_units=0.0)
    return P.summarize(P.score_scenarios(key, l1, deltas))["mean_accuracy_cf_wmape"]


def _present(source, run_sets: Iterable[RunSet]) -> set[str]:
    out = set()
    for rs in run_sets:
        out.update(p.rsplit("/", 1)[0] for p in source.glob(rs.record_pattern()))
    return out


def _score_run(P, run: PlannedRun, run_dir: Path, key, fkey, archives: _Archives, sums, scorer_sha: str,
               on_run: Callable | None):
    ds = run.dataset
    check = {"run": run.path, "run_set": run.run_set.name, "dataset": ds.id, "model": run.model, "seed": ds.seed}
    problems, _ = check_run_record(run_dir, run.path, release_name=ds.release_name, release_path=ds.release_path,
                                   model=run.model, foundation=run.run_set.foundation,
                                   tuned=run.run_set.settings == "tuned", sums=sums, scorer_sha256=scorer_sha)
    check["files_match_record"] = not problems
    if problems:
        return {**check, "status": None, "archived": [], "ok": False, "problems": problems}, None, None
    try:
        l1 = P.load_layer1(run_dir / FORECAST)
        dl = P.load_deltas(run_dir / SCENARIO)
        ps = P.score_scenarios(key, l1, dl)
        sm = P.run_summary(ps, fkey, l1)
    except Exception as exc:
        problems.append(f"{run.path}: cannot be scored: {type(exc).__name__}: {exc}")
        return {**check, "status": None, "archived": [], "ok": False, "problems": problems}, None, None
    check["status"] = sm["status"]
    if sm["status"] != "valid":
        problems.append(f"{run.path}: the score is invalid: {sm['invalid_reasons']}")
    ps_csv = ps.to_csv(index=False, lineterminator="\n")
    compared = []
    for folder, labels in run.run_set.comparisons(ds):
        arch = archives.get(folder)
        if arch is None:
            problems.append(f"{run.path}: the archived scores in results/{folder} cannot be read")
            continue
        res = {"folder": f"results/{folder}", **arch.compare(labels, run.model, ds.seed, ps_csv, sm)}
        compared.append(res)
        if not res["archived"]:
            problems.append(f"{run.path}: no archived score in results/{folder} (looked for {', '.join(labels)})")
        elif not res["ok"]:
            problems.append(f"{run.path}: the score differs from the archived score in results/{folder} "
                            f"(largest relative difference {res['max_relative_difference']:.1e})")
    if on_run is not None:
        on_run({"run_set": run.run_set.name, "dataset": ds, "model": run.model, "run": run.path, "run_dir": run_dir,
                "key": key.copy(), "forecast_truth": fkey.copy(), "forecast": l1.copy(), "scenarios": dl.copy(),
                "per_scenario": ps.copy(), "summary": dict(sm)})
    return {**check, "archived": compared, "ok": not problems, "problems": problems}, ps, sm


def _scorer_record(P) -> dict:
    return {"path": "metrics/panel_scorer.py", "version": P.SCORER_VERSION, "sha256": module_sha256(P)}


def _versions() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}


def _progress(i: int, n: int, quiet: bool) -> None:
    if quiet:
        return
    text = f"Checked {i} of {count_text(n, 'dataset')}."
    if sys.stdout.isatty():
        print(f"\r{text}", end="\n" if i == n else "", flush=True)
    elif i == n:
        print(text, flush=True)


def rescore_runs(run_sets, source, out: Path | None = None, datasets=None, models=None, *,
                 on_run: Callable | None = None, quiet: bool = False) -> dict:
    P = panel_scorer()
    scorer = _scorer_record(P)
    planned = plan_runs(run_sets, datasets, models)
    if not planned:
        raise RescoreError("no released run matches the selection")
    sums = release_sums(source)
    archives = _Archives(source, sums)
    present = _present(source, {r.run_set for r in planned})
    checks, rows, scen, nochange = [], [], [], []
    by_dataset: dict[Dataset, list[PlannedRun]] = {}
    for r in planned:
        if r.path in present:
            by_dataset.setdefault(r.dataset, []).append(r)
        else:
            checks.append({"run": r.path, "run_set": r.run_set.name, "dataset": r.dataset.id, "model": r.model,
                           "seed": r.dataset.seed, "files_match_record": False, "status": None, "archived": [],
                           "ok": False, "problems": [f"{r.path}: no run folder with a run record in the release"]})
    tmp = None
    if out is not None:
        out = output_folder(Path(out))
        cache = out / "_ground_truth"
    else:
        tmp = tempfile.mkdtemp(prefix="causaldemand_rescore_")
        cache = Path(tmp)
    try:
        for i, (ds, runs) in enumerate(by_dataset.items(), 1):
            try:
                root = source.fetch([f"{r.path}/*" for r in runs])
                truth = ground_truth_csv(source, ds, cache)
                key = P.load_answer_key(truth["counterfactual"])
                fkey = P.load_answer_key(truth["forecast"])
            except (SourceError, KeyError, OSError, ValueError) as exc:
                why = f"{ds.release_path}: the runs or the ground truth cannot be read: {type(exc).__name__}: {exc}"
                for r in runs:
                    checks.append({"run": r.path, "run_set": r.run_set.name, "dataset": ds.id, "model": r.model,
                                   "seed": ds.seed, "files_match_record": False, "status": None, "archived": [],
                                   "ok": False, "problems": [why]})
                _progress(i, len(by_dataset), quiet)
                continue
            nochange.append({"dataset": ds.id, "release_path": ds.release_path, "variant": ds.variant,
                             "setting": ds.setting, "seed": ds.seed,
                             "mean_accuracy_cf_wmape": no_change_error(P, key)})
            for r in runs:
                check, ps, sm = _score_run(P, r, root / r.path, key, fkey, archives, sums, scorer["sha256"], on_run)
                checks.append(check)
                if sm is None:
                    continue
                labels = {"run_set": r.run_set.name, "dataset": ds.id, "release_path": ds.release_path,
                          "model": r.model, "seed": ds.seed}
                rows.append({**labels, **sm})
                labelled = ps.copy()
                for k, v in reversed(list(labels.items())):
                    labelled.insert(0, k, v)
                scen.append(labelled)
            shutil.rmtree(cache / ds.release_path, ignore_errors=True)
            _progress(i, len(by_dataset), quiet)
    finally:
        shutil.rmtree(cache, ignore_errors=True)
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)

    per_run = pd.DataFrame(rows)
    per_scenario = pd.concat(scen, ignore_index=True) if scen else pd.DataFrame()
    no_change = pd.DataFrame(nochange, columns=["dataset", "release_path", "variant", "setting", "seed",
                                                "mean_accuracy_cf_wmape"])
    names = list(_selection(run_sets))
    by_run_set = {n: per_run[per_run["run_set"] == n].reset_index(drop=True) for n in names if len(per_run)}
    ps_by_run_set = {n: per_scenario[per_scenario["run_set"] == n].reset_index(drop=True)
                     for n in names if len(per_scenario)}
    problems = archives.problems + list(dict.fromkeys(p for c in checks for p in c["problems"]))
    counts = {}
    for n in names:
        cs = [c for c in checks if c["run_set"] == n]
        counts[n] = {"expected": len(cs), "scored": sum(c["status"] is not None for c in cs),
                     "ok": sum(c["ok"] for c in cs),
                     "identical": sum(bool(c["archived"]) and all(a.get("per_scenario_identical") and
                                                                  a.get("summary_identical") for a in c["archived"])
                                      for c in cs)}
    if out is not None:
        per_run.to_csv(out / "per_run.csv", index=False, lineterminator="\n")
        per_scenario.to_csv(out / "per_scenario.csv", index=False, lineterminator="\n")
        no_change.to_csv(out / "no_change.csv", index=False, lineterminator="\n")
    return {"per_run": per_run, "per_scenario": per_scenario, "by_run_set": by_run_set,
            "per_scenario_by_run_set": ps_by_run_set, "no_change": no_change, "checks": checks,
            "problems": problems, "ok": not problems, "counts": counts, "sha256sums": _sums_state(sums),
            "scorer": scorer, "versions": _versions()}


def _real_dataset(dataset) -> Dataset:
    ds = dataset if isinstance(dataset, Dataset) else by_id(str(dataset))
    if not ds.is_real:
        raise RescoreError(f"{ds.id} is not an observed-sales dataset (cereal, snack-crackers)")
    return ds


def real_title(ds: Dataset) -> str:
    return f"Forecast WMAPE and own-price sign accuracy, {REAL_DATASETS[ds.category][1]}"


REAL_NOTES = (
    "Forecast WMAPE over the holdout rows with sales.",
    "Sign correct, price up: share of +10% price rises with predicted lower sales, zero predicted changes left out.",
    "95% CI over 1,000 bootstrap draws of the stores.",
    "LightGBM, XGBoost, random forest and Double ML are tuned for forecast accuracy; the other models use default "
    "settings. TabPFN and TabFM: mean over five sampler seeds.",
)


def real_table(ds: Dataset, rows) -> T.Table:
    return T.real_table(rows, real_title(ds), REAL_NOTES)


def _same_numpy(spec: dict) -> bool:
    return np.__version__.split(".")[:2] == str(spec.get("numpy_version", "")).split(".")[:2]


def _csv_rows(text: str) -> tuple[list[str], list[list[str]]]:
    rows = list(csv.reader(io.StringIO(text)))
    return (rows[0], rows[1:]) if rows else ([], [])


def _csv_rows_text(header: list[str], rows: list[list[str]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def real_run_path(name: str, run_dir: str) -> str:
    parts = run_dir.strip().split("/")
    return f"runs/dominicks/{parts[0]}/{name}/{parts[-1]}"


def _row_runs(fname: str, header: list[str], name: str) -> Callable:
    if fname == "per_run.csv":
        i = header.index("run_dir")
        return lambda r: tuple(real_run_path(name, d) for d in r[i].split(";") if d.strip())
    ia, im, ir = header.index("arm"), header.index("model"), header.index("run")
    return lambda r: (f"runs/dominicks/{r[ia]}/{name}/{r[ir] or r[im]}",)


def compare_real_runs(fname: str, text: str, archived: str, name: str, exact: bool) -> dict[str, list[str]] | None:
    header, new = _csv_rows(text)
    old_header, old = _csv_rows(archived)
    if header != old_header:
        return None
    try:
        runs_of = _row_runs(fname, header, name)
        groups: dict[tuple, tuple[list, list]] = {}
        for which, rows in ((0, new), (1, old)):
            for r in rows:
                groups.setdefault(runs_of(r), ([], []))[which].append(r)
    except (ValueError, IndexError):
        return None
    out: dict[str, list[str]] = {}
    for runs, (a, b) in groups.items():
        if not b:
            why = f"{fname}: the archived scores have no row of this run"
        elif not a:
            why = f"{fname}: the re-scored file has no row of this run"
        else:
            c = compare_csv_text(_csv_rows_text(header, a), _csv_rows_text(header, b), abs_tol=TOLERANCE)
            if c["identical"] or (not exact and c["within_tolerance"]):
                continue
            why = (f"{fname}: not identical (largest relative difference {c['max_relative_difference']:.1e})"
                   if c["within_tolerance"] else f"{fname}: {c['first_difference']}")
        for run in runs:
            out.setdefault(run, []).append(why)
    return out


def compare_real(scores: Path, archived: Path, name: str, known: dict) -> dict:
    label = known["archived_label"]
    spec_old = json.loads((archived / "bootstrap_spec.json").read_text(encoding="utf-8"))
    spec_new = json.loads((scores / "bootstrap_spec.json").read_text(encoding="utf-8"))
    exact = _same_numpy(spec_old)
    out = {"byte_identity_required": exact, "numpy_of_archived_scores": spec_old.get("numpy_version")}
    problems, dataset_level, by_run = [], [], {}
    for fname, rename in (("per_scenario.csv", False), ("per_run.csv", True)):
        text = (scores / fname).read_text(encoding="utf-8")
        if rename:
            text = text.replace(f"/{name}/", f"/{label}/")
        old = (archived / fname).read_text(encoding="utf-8")
        c = compare_csv_text(text, old, abs_tol=TOLERANCE)
        ok = c["identical"] or (not exact and c["within_tolerance"])
        out[fname] = {**c, "ok": ok}
        if ok:
            continue
        why = "not byte-identical" if exact and c["within_tolerance"] else (c["first_difference"] or "differs")
        problems.append(f"results/dominicks/{name}/{fname}: the scores differ ({why})")
        runs = compare_real_runs(fname, text, old, name, exact)
        if not runs:
            dataset_level.append(f"results/dominicks/{name}/{fname}: the scores differ, and the difference lies "
                                 "in no single run's rows")
        for run, whys in (runs or {}).items():
            by_run.setdefault(run, []).extend(whys)
    spec_problems = []
    for k in ("B", "seed", "n_stores", "store_order", "draw_matrix_sha256", "scorer_version", "dataset"):
        if spec_new.get(k) != spec_old.get(k):
            spec_problems.append(f"results/dominicks/{name}/bootstrap_spec.json: {k} differs")
    sha_new, sha_old = spec_new.get("dataset_sha256") or {}, spec_old.get("dataset_sha256") or {}
    for k in ("train", "panel"):
        if sha_new.get(k) != sha_old.get(k):
            spec_problems.append(f"results/dominicks/{name}/bootstrap_spec.json: the {k} checksum differs")
    if sha_new.get("forecast_truth_csv") != known["forecast_truth_csv_sha256"]:
        spec_problems.append(f"dominicks/{name}: the forecast ground truth built from the Kilts Center files is not "
                             "the archived one")
    problems += spec_problems
    dataset_level += spec_problems
    out.update(problems=problems, ok=not problems, runs=by_run, dataset_level=dataset_level)
    return out


def rescore_real(dataset, source, out: Path | None = None, *, quiet: bool = False) -> dict:
    from causaldemand import realdata as R

    ds = _real_dataset(dataset)
    say = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
    name = ds.release_name
    known = R.REAL_BOOTSTRAP[name]
    scorer_sha = module_sha256(panel_scorer())
    sums = release_sums(source)
    planned = [(arm, run, f"runs/dominicks/{arm}/{name}/{run}") for arm, runs in REAL_ARMS for run in runs]
    present = {p.rsplit("/", 1)[0] for p in source.glob(f"runs/dominicks/*/{name}/*/{RECORD}")}
    found = [p for p in planned if p[2] in present]
    root = source.fetch([f"{rel}/*" for _, _, rel in found]) if found else None
    checks = []
    for arm, run, rel in planned:
        check = {"run": rel, "arm": arm, "model": run, "files_match_record": False, "scores_match_archive": None,
                 "status": None, "ok": False}
        if rel not in present:
            check["problems"] = [f"{rel}: no run folder with a run record in the release"]
        else:
            probs, _ = check_run_record(root / rel, rel, release_name=name, release_path=ds.release_path,
                                        model=run, foundation=arm == "foundation_models", tuned=arm == "cpu_tuned",
                                        sums=sums, scorer_sha256=scorer_sha)
            check.update(files_match_record=not probs, problems=probs)
        checks.append(check)
    problems = [p for c in checks for p in c["problems"]]
    results = f"results/dominicks/{name}"
    files = [f"{results}/{f}" for f in ("per_run.csv", "per_scenario.csv", "bootstrap_spec.json")]
    archive_problems = []
    try:
        rroot = source.fetch(files)
        archive_problems = [p for p in (_listed(sums, f, rroot / f) for f in files) if p]
    except SourceError as exc:
        rroot = None
        archive_problems = [f"{results}: the archived scores cannot be read: {exc}"]
    problems += archive_problems
    result = {"dataset": ds.id, "release_name": name, "sha256sums": _sums_state(sums), "checks": checks,
              "comparison": None}

    def not_scored(reason: str) -> dict:
        for c in checks:
            if not c["problems"]:
                c["note"] = f"not scored, because {reason}"
        rr = {**result, "problems": problems, "ok": False, "not_scored": reason}
        return rr

    bad = sum(not c["files_match_record"] for c in checks)
    if bad:
        return not_scored(f"{bad} of its {count_text(len(checks), 'run')} {'fails' if bad == 1 else 'fail'} "
                          "a file check")
    if archive_problems:
        return not_scored("the archived scores fail their checks")

    from causaldemand import dominicks
    work = Path(tempfile.mkdtemp(prefix=f"_work_{name}_", dir=output_folder(Path(out)) if out else None))
    try:
        folder = dominicks.built_dataset(ds, source=source, say=None if quiet else say, indent="    ")
        res = R.score_reference_runs(folder, root / "runs" / "dominicks", work / "scores",
                                     dataset_folder=name, label=known["archived_label"], quiet=True)
        comparison = compare_real(work / "scores", rroot / results, name, known)
        if out is not None:
            for f in REAL_OUTPUT_FILES:
                shutil.copyfile(work / "scores" / f, Path(out) / f)
    except (dominicks.KiltsError, DiskSpaceError) as exc:
        problems.append(f"dominicks/{name}: the dataset cannot be built from the Kilts Center files: {exc}")
        return not_scored("the dataset cannot be built from the Kilts Center files")
    except (SourceError, R.RealDataError, OSError, ValueError, KeyError) as exc:
        problems.append(f"dominicks/{name}: the scoring stopped on an error: {type(exc).__name__}: {exc}")
        return not_scored(f"the scoring stopped on an error ({type(exc).__name__}: {exc})")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    unscored = {m.split(" (", 1)[0]: m for m in res["missing"] if m not in REAL_NOT_RELEASED}
    scored = {f"runs/dominicks/{arm}/{name}/{run}": row for (arm, run), row in res["runs"].items()}
    problems += comparison["problems"]
    for c in checks:
        own = []
        row = scored.get(c["run"])
        if row is None:
            own.append(f"{c['run']}: not scored: {unscored.get(c['arm'] + '/' + c['model'], 'no score')}")
        else:
            c["status"] = row["status"]
            if row["status"] != "valid":
                own.append(f"{c['run']}: the score is invalid: {row.get('invalid_reasons')}")
        differences = [f"{c['run']}: the score differs from the archived score in {results}/{w}"
                       for w in comparison["runs"].get(c["run"], [])]
        problems += [p for p in own + differences if p not in problems]
        if comparison["dataset_level"]:
            differences.append(f"{c['run']}: not confirmed, because the scores of the dataset differ from the "
                               f"archived scores in {results} as a whole")
        c["scores_match_archive"] = row is not None and not differences
        c["problems"] = own + differences
        c["ok"] = not c["problems"]
    rows = R.real_table_rows(res["per_run"])
    rr = {**result, "comparison": comparison, "problems": problems, "ok": not problems, "rows": rows,
          "table": real_table(ds, rows), "per_run": res["per_run"], "per_scenario": res["per_scenario"],
          "runs_scored": len(res["runs"])}
    return rr


def run_set_frames(result: dict, prefix: str) -> dict[str, pd.DataFrame]:
    return {k: v for k, v in result["by_run_set"].items() if k.startswith(prefix + "/")}


def no_change_rows(result: dict, variant: str) -> pd.DataFrame:
    nc = result["no_change"]
    return nc[nc["variant"] == variant].reset_index(drop=True)


def main_tables(result: dict) -> list[T.Table]:
    return T.main_table(run_set_frames(result, "main"), no_change_rows(result, "main"))


def tuned_tables(result: dict) -> list[T.Table]:
    return T.tuned_table(run_set_frames(result, "main"), "tissue")


def yogurt_tables(result: dict) -> list[T.Table]:
    return T.yogurt_table(run_set_frames(result, "yogurt"), no_change_rows(result, "yogurt"))


def switching_tables(result: dict) -> list[T.Table]:
    return T.switching_table(run_set_frames(result, "switching"), no_change_rows(result, "switching"))


def strength_frame(result: dict) -> pd.DataFrame:
    f = result["by_run_set"]
    need = ("main/cpu_default", "main/foundation_models", "dose_response/cpu", "dose_response/foundation_models")
    absent = [n for n in need if n not in f]
    if absent:
        raise T.TableError(f"the confounding-strength table needs the run sets {', '.join(absent)}")
    return T.strength_frame(*(f[n] for n in need))


def strength_tables(frame: pd.DataFrame) -> list[T.Table]:
    return [T.strength_table(frame, T.CF)]


TABLE_RUNS = {
    "main": {f"main/{arm}": models for arm, models in DEFAULT_RUNS},
    "tuned": {"main/cpu_default": TUNED_MODELS, "main/cpu_tuned": TUNED_MODELS},
    "confounding_strength": {"main/cpu_default": CPU_MODELS, "main/foundation_models": FM_RUNS,
                             "dose_response/cpu": CPU_MODELS, "dose_response/foundation_models": FM_RUNS},
    "switching": {f"switching/{arm}": models for arm, models in DEFAULT_RUNS},
    "yogurt": {f"yogurt/{arm}": models for arm, models in SETTINGS_RUNS},
}
NO_CHANGE_CHECK_NOTE = ("No change is computed here from the ground truth of each dataset; the release holds no "
                        "archived No-change scores, so this row is not compared with one.")


def table_failures(result: dict, table: str) -> tuple[int, int]:
    need = TABLE_RUNS[table]
    mine = [c for c in result["checks"]
            if c.get("run_set") in need and ("model" not in c or c["model"] in need[c["run_set"]])]
    return len(mine), sum(not c["ok"] for c in mine)


def _real_not_built(rr: dict) -> str:
    n = len(rr["checks"])
    if rr.get("comparison") is None:
        return f"the runs are not scored, because {rr.get('not_scored') or 'files fail their checks'}"
    bad = sum(not c["ok"] for c in rr["checks"])
    return f"{bad} of its {count_text(n, 'run')} {'fails' if bad == 1 else 'fail'} a check"


_CELL = re.compile(r"^(?P<value>[+-]?\d+(?:\.\d+)?)%?(?: ± (?P<hw>\d+(?:\.\d+)?))?"
                   r"(?: \[(?P<lo>[+-]?\d+(?:\.\d+)?), (?P<hi>[+-]?\d+(?:\.\d+)?)\])?")
CSV_COLUMNS = ["table", "panel", "row", "column", "cell", "value", "half_width", "lower", "upper"]


def table_cells(tables: Iterable[T.Table]) -> list[dict]:
    out = []
    for t in tables:
        header = t.header_rows()
        cols = ([f"{u}, {lo}" if u and lo else (u or lo) for u, lo in zip(*header)] if len(header) == 2
                else list(header[0]))
        panel = ""
        for r, cells in zip(t.rows, T.label_cells(t)):
            if not r:
                continue
            if len(r) == 1:
                panel = r[0]
                continue
            label = " ".join(c for c in cells if c)
            for column, text in zip(cols[t.label_columns:], r[t.label_columns:]):
                m = _CELL.match(text)
                parts = m.groupdict() if m else {}
                out.append({"table": t.title, "panel": panel, "row": label, "column": column, "cell": text,
                            "value": parts.get("value") or "", "half_width": parts.get("hw") or "",
                            "lower": parts.get("lo") or "", "upper": parts.get("hi") or ""})
    return out


def write_tables(folder: Path, stem: str, tables: list[T.Table]) -> list[str]:
    folder = Path(folder)
    (folder / f"{stem}.md").write_text("\n".join(T.to_markdown(t, heading="##") for t in tables), encoding="utf-8")
    with open(folder / f"{stem}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(table_cells(tables))
    return [f"{stem}.md", f"{stem}.csv"]


OUTPUT_FILES = ("results.md", "main.md", "main.csv", "tuned.md", "tuned.csv", "switching.md", "switching.csv",
                "yogurt.md", "yogurt.csv",
                "confounding_strength.md", "confounding_strength.csv", "confounding_strength.pdf", "cereal.md",
                "cereal.csv",
                "snack-crackers.md", "snack-crackers.csv", "per_run.csv", "per_scenario.csv", "no_change.csv",
                "bootstrap_spec.json", "checks.json")
REAL_OUTPUT_FILES = ("per_run.csv", "per_scenario.csv", "bootstrap_spec.json")


def _clear(folder: Path) -> None:
    for name in OUTPUT_FILES:
        p = folder / name
        if p.is_file():
            p.unlink()
    for sub in REAL_DATASETS:
        for name in REAL_OUTPUT_FILES:
            p = folder / sub / name
            if p.is_file():
                p.unlink()
    for p in folder.glob("_*"):
        if p.is_dir() and (p.name == "_ground_truth" or p.name.startswith("_work_")):
            shutil.rmtree(p, ignore_errors=True)


def _what(category: str, tissue_dose: bool, tissue_switching: bool) -> str:
    if category == "tissue":
        extra = [w for w, on in (("--tissue-dose", tissue_dose), ("--tissue-switching", tissue_switching)) if on]
        return "tissue" + "".join(f" {e}" for e in extra)
    return category


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return str(x)
    return x


def _print_tables(tables: list[T.Table]) -> None:
    for t in tables:
        print()
        print(T.to_text(t), end="")


def build_tables(result: dict | None, real_results: list, category: str, tissue_dose: bool = False,
                 tissue_switching: bool = False) -> dict:
    if category == "all":
        tissue_dose = tissue_switching = True
    tables, not_built, strength = [], {}, (None, None, None)

    def build(name: str, fn):
        n, bad = table_failures(result, name)
        if bad:
            not_built[name] = f"{bad:,} of its {count_text(n, 'run')} {'fails' if bad == 1 else 'fail'} a check"
            return None
        try:
            return fn()
        except ValueError as exc:
            not_built[name] = str(exc) or type(exc).__name__
            return None

    def strength_parts():
        frame = strength_frame(result)
        points = F.strength_points(frame)
        return frame, strength_tables(frame), points

    if result is not None:
        if category in ("tissue", "all"):
            tabs = build("main", lambda: main_tables(result))
            if tabs:
                tables.append(("main", T.merge_notes(tabs)))
            tabs = build("tuned", lambda: tuned_tables(result))
            if tabs:
                tables.append(("tuned", T.merge_notes(tabs)))
        if tissue_dose:
            strength = build("confounding_strength", strength_parts) or strength
            if strength[1]:
                tables.append(("confounding_strength", T.merge_notes(strength[1])))
        if tissue_switching:
            tabs = build("switching", lambda: switching_tables(result))
            if tabs:
                tables.append(("switching", T.merge_notes(tabs)))
        if category in ("yogurt", "all"):
            tabs = build("yogurt", lambda: yogurt_tables(result))
            if tabs:
                tables.append(("yogurt", T.merge_notes(tabs)))
    for rr in real_results:
        if rr.get("ok") and "table" in rr:
            tables.append((rr["dataset"], T.merge_notes([rr["table"]])))
        else:
            not_built[rr["dataset"]] = _real_not_built(rr)
    return {"tables": tables, "not_built": not_built, "strength_frame": strength[0], "strength_points": strength[2]}


ALL_SECTIONS = (
    ("main", "Main results"),
    ("confounding_strength", "Counterfactual WMAPE by confounding strength"),
    ("rank", "Agreement of the model rankings across seeds and scenarios"),
    ("seeds", "WMAPE by seed"),
    ("scenario_error", "Counterfactual WMAPE by scenario"),
    ("scenario_bias", "Counterfactual bias by scenario"),
    ("tuned", "WMAPE with default and forecast-tuned settings"),
    ("switching", "WMAPE with product distances from household switching"),
    ("yogurt", "WMAPE on a second product category, yogurt"),
    ("real", "Forecast WMAPE and direction of the price effect on real store sales"),
)


def write_results(folder: Path, sections: dict, figure: str | None) -> None:
    parts = ["# CausalDemand reference results", ""]
    for key, heading in ALL_SECTIONS:
        tables = sections.get(key)
        if not tables:
            continue
        parts += [f"## {heading}", ""]
        if key == "confounding_strength" and figure:
            parts += [f"Figure: {figure}", ""]
        parts += [T.to_markdown(t) for t in T.merge_notes(tables)]
    (Path(folder) / "results.md").write_text("\n".join(parts), encoding="utf-8")


def run_rescore(category: str, tissue_dose: bool = False, tissue_switching: bool = False, source=None) -> int:
    from causaldemand import appendix as A

    selection = category_run_sets(category, tissue_dose, tissue_switching)
    everything = category == "all"
    if everything:
        tissue_dose = tissue_switching = True
    source = source if source is not None else make_source(None)
    reals = real_datasets(category)
    n_runs = len(plan_runs(selection)) if selection else 0
    n_runs += sum(len(runs) for _, runs in REAL_ARMS) * len(reals)
    out = output_folder(Path(OUTPUT_ROOT)) / category
    out.mkdir(exist_ok=True)
    _clear(out)
    what = _what(category, tissue_dose, tissue_switching)
    print(f"causaldemand rescore {what}: {n_runs:,} released run{'' if n_runs == 1 else 's'} from "
          f"{source.describe()}.", flush=True)

    t0 = time.time()
    boot = A.StoreBootstrap(TABLE_RUNS["main"]) if everything else None
    keep = None if everything else out
    result = rescore_runs(selection, source, keep, on_run=boot) if selection else None
    real_results = []
    for ds in reals:
        target = out if category != "all" else None
        real_results.append(rescore_real(ds, source, target))

    built = build_tables(result, real_results, category, tissue_dose, tissue_switching)
    printed, not_built, figure_note = built["tables"], built["not_built"], None
    points = built["strength_points"]
    if points is not None:
        try:
            F.strength_figure(points, out / "confounding_strength.pdf")
            figure_note = "confounding_strength.pdf"
        except ImportError:
            figure_note = None
            print("The figure is not drawn: matplotlib is not installed (pip install matplotlib).")
    appendix_problems = []
    if everything:
        sections = {name: tabs for name, tabs in printed if name in dict(ALL_SECTIONS)}
        sections["real"] = [tab for name, tabs in printed if name in REAL_DATASETS for tab in tabs]
        if result is not None:
            extra, appendix_problems = A.appendix_tables(result, boot, source, release_sums(source))
            sections.update(extra)
        write_results(out, sections, figure_note)
    else:
        for stem, tabs in printed:
            _print_tables(tabs)
            write_tables(out, stem, tabs)

    problems = (result["problems"] if result else []) + [p for rr in real_results for p in rr["problems"]]
    problems += appendix_problems
    synthetic_checks = result["checks"] if result else []
    real_checks = [c for rr in real_results for c in rr["checks"]]
    n_checked = len(synthetic_checks) + len(real_checks)
    n_ok = sum(c["ok"] for c in synthetic_checks + real_checks)
    states = ([result.get("sha256sums")] if result else []) + [rr.get("sha256sums") for rr in real_results]
    sums_state = "absent" if "absent" in states else "checked"
    record = {"causaldemand_version": __version__, "category": category, "tissue_dose": tissue_dose,
              "tissue_switching": tissue_switching, "source": source.record(), **_versions(),
              "scorer": _scorer_record(panel_scorer()), "tolerance": TOLERANCE, "sha256sums": sums_state,
              "runs": n_runs, "runs_checked": n_checked, "runs_passing_every_check": n_ok,
              "ok": not problems and not not_built, "problems": problems,
              "tables_built": [name for name, _ in printed], "tables_not_built": not_built,
              "no_change": ({"compared_with_archived_scores": False, "note": NO_CHANGE_CHECK_NOTE}
                            if result is not None else None),
              "figure": figure_note, "seconds": round(time.time() - t0, 1),
              "synthetic": ({"counts": result["counts"], "sha256sums": result["sha256sums"], "runs": synthetic_checks}
                            if result else None),
              "observed_sales": [{k: v for k, v in rr.items()
                                  if k not in ("table", "per_run", "per_scenario", "rows")} for rr in real_results]}
    (out / "checks.json").write_text(json.dumps(_plain(record), indent=1, default=str) + "\n", encoding="utf-8")

    print()
    bad = n_checked - n_ok
    if not problems:
        print(f"All {count_text(n_checked, 'run')} match the archived scores.")
    else:
        print(f"{bad:,} of {count_text(n_checked, 'run')} fail a check:" if bad else
              f"{count_text(len(problems), 'check')} failed:")
        for p in problems[:20]:
            print(f"  {p}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more (checks.json lists every one)")
    for name, why in not_built.items():
        print(f"Table not built: {name}: {why}")
    if sums_state == "absent":
        print(SUMS_ABSENT_NOTE)
    figure = f" (figure: {figure_note})" if figure_note else ""
    print(f"Saved to {show(out / 'results.md') if everything else show(out) + os.sep}{figure}")
    return 0 if record["ok"] else 1
