from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from causaldemand import rescore as RS
from causaldemand._code import output_folder, panel_scorer
from causaldemand.download import OUT, ground_truth_csv, show
from causaldemand.names import Dataset, UnknownNameError, count_text, select
from causaldemand.sources import make_source

OUTPUT_ROOT = "causaldemand_reruns"
MODELS = RS.CPU_MODELS + RS.SAMPLED_MODELS + ("chronos2",)
PYTHON = sys.executable
CPU_MODULES = {"numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn", "scipy": "scipy",
               "statsmodels": "statsmodels", "lightgbm": "lightgbm", "xgboost": "xgboost"}
FM_MODULES = {"torch": "torch", "tabpfn": "tabpfn", "tabfm": "tabfm", "chronos": "chronos-forecasting"}
MODEL_ENV = ("CARD_KEEP_PROMO_COST", "CARD_PRICE_ENDING_CENT")
CPU_IMPORTS = ("sklearn", "statsmodels", "lightgbm", "xgboost")
DONE = (RS.FORECAST, RS.SCENARIO, RS.RECORD)
UNITS_TOLERANCE = 1e-9
MIN_PYTHON = (3, 12)


class RerunError(RuntimeError):
    pass


@dataclass(frozen=True)
class Job:
    dataset: Dataset
    released: str
    run_set: str
    run: str

    @property
    def model(self) -> str:
        return RS.base_model(self.run)

    @property
    def tuned(self) -> bool:
        return self.run_set.endswith("cpu_tuned")

    @property
    def foundation(self) -> bool:
        return self.run_set.endswith("foundation_models")

    @property
    def folder(self) -> Path:
        return Path(OUTPUT_ROOT) / self.run_set / self.dataset.id / self.run

    def label(self) -> str:
        seed = RS._SAMPLER_RUN.match(self.run)
        return self.dataset.id + (" (tuned)" if self.tuned else "") + (f", sampler seed {seed.group(2)}" if seed else "")


def models_folder() -> Path:
    here = Path(__file__).resolve().parent
    for folder in (here / "models", here.parent / "models"):
        if (folder / "run_panel_model.py").is_file():
            return folder
    raise RerunError("the reference models are not installed with this package")


def plan(category: str, model: str, tissue_dose: bool = False, tissue_switching: bool = False) -> list[Job]:
    if model not in MODELS:
        raise UnknownNameError(f"{model!r} is not a reference model; models: {', '.join(MODELS)}")
    datasets = select(category, tissue_dose, tissue_switching)
    if all(ds.is_real for ds in datasets):
        ds = datasets[0]
        return [Job(ds, f"runs/dominicks/{arm}/{ds.release_name}/{run}", f"dominicks/{arm}", run)
                for arm, runs in RS.REAL_ARMS for run in runs if RS.base_model(run) == model]
    runs = RS.plan_runs(RS.category_run_sets(category, tissue_dose, tissue_switching), models=(model,))
    return [Job(r.dataset, r.path, r.run_set.name, r.model) for r in runs]


INSPECT = """\
import importlib, importlib.metadata, importlib.util, json, platform, shutil, sys
mods, dists, imports = json.loads(sys.argv[1]), json.loads(sys.argv[2]), json.loads(sys.argv[3])
def version(d):
    try:
        return importlib.metadata.version(d)
    except importlib.metadata.PackageNotFoundError:
        return None
missing = [m for m in mods if importlib.util.find_spec(m) is None]
broken = {}
for m in imports:
    if m in missing:
        continue
    try:
        importlib.import_module(m)
    except Exception as exc:
        broken[m] = f"{type(exc).__name__}: {exc}"
print(json.dumps({"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
                  "missing": missing, "broken": broken, "nvidia_smi": shutil.which("nvidia-smi") is not None,
                  "versions": {d: version(d) for d in dists}}))
"""


def inspect_python(modules: dict, dists, imports=()) -> dict:
    out = subprocess.run([PYTHON, "-c", INSPECT, json.dumps(list(modules)), json.dumps(sorted(dists)),
                          json.dumps(list(imports))], capture_output=True, text=True)
    if out.returncode != 0:
        raise RerunError(f"{PYTHON} cannot be run: {out.stderr.strip()[-300:]}")
    return json.loads(out.stdout)


def has_gpu() -> bool:
    out = subprocess.run([PYTHON, "-c", "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)"],
                         capture_output=True, text=True)
    if out.returncode not in (0, 3):
        raise RerunError(f"{PYTHON} cannot import torch: {out.stderr.strip()[-300:]}")
    return out.returncode == 0


def released_versions(record: dict, foundation: bool) -> tuple[str | None, dict]:
    if foundation:
        versions = dict(record.get("versions") or {})
        return versions.pop("python", None), versions
    return record.get("python"), {CPU_MODULES.get(k, k): v for k, v in (record.get("packages") or {}).items()}


def _computer(system: str, machine: str) -> str:
    return f"{'macOS' if system in ('Darwin', 'macOS') else system} {machine}"


def version_differences(info: dict, python: str | None, versions: dict, hardware: dict | None = None) -> list[str]:
    out = []
    if hardware and hardware.get("platform") and hardware.get("machine") and info.get("system"):
        released = _computer(hardware["platform"].split("-", 1)[0], hardware["machine"])
        if released != _computer(info["system"], info["machine"]):
            out.append(released)
    if python and python != info["python"]:
        out.append(f"Python {python}")
    return out + [f"{d} {v}" for d, v in sorted(versions.items())
                  if info["versions"].get(d) is not None and info["versions"][d] != v]


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def command(job: Job, models: Path, record: dict, tuning: dict | None = None) -> list[str]:
    data = str(Path(OUT) / job.dataset.id)
    if job.foundation:
        context = record.get("context") or {}
        args = [PYTHON, str(models / "foundation" / "run_fm.py"), data, data, str(job.folder), job.model]
        if context.get("sampler_seed") is not None or context.get("store_weeks") is not None:
            args.append(str(context.get("sampler_seed") or 0))
        if context.get("store_weeks") is not None:
            args.append(str(context["store_weeks"]))
        return args
    out_root = str(job.folder.parent.parent)
    if job.tuned:
        return [PYTHON, str(models / "tune_panel_model.py"), data, data, out_root, job.model,
                str((tuning or {})["n_trials"]), str(job.folder / "trials.jsonl")]
    return [PYTHON, str(models / "run_panel_model.py"), data, data, out_root, job.model]


def environment(record: dict) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in MODEL_ENV}
    env.update({k: str(v) for k, v in (record.get("env") or {}).items()})
    return env


def _key(line: str, width: int) -> str:
    return ",".join(line.split(",", width)[:width])


def _gap(line: str, old: str) -> float:
    try:
        return abs(float(line.rsplit(",", 1)[1]) - float(old.rsplit(",", 1)[1]))
    except (IndexError, ValueError):
        return float("inf")


def kept_rows(released: Path, rerun: Path, width: int, out: Path) -> tuple[int, int, float]:
    with gzip.open(released, "rt", newline="") as fh:
        lines = fh.read().splitlines()
    want = {_key(line, width): line for line in lines[1:]}
    same = kept = 0
    gap = 0.0
    with gzip.open(rerun, "rt", newline="") as fh, gzip.open(out, "wt", newline="") as w:
        head = fh.readline().rstrip("\r\n")
        w.write(head + "\n")
        for line in fh:
            line = line.rstrip("\r\n")
            old = want.get(_key(line, width))
            if old is not None:
                kept += 1
                if line == old:
                    same += 1
                else:
                    gap = max(gap, _gap(line, old))
                w.write(line + "\n")
    if head != lines[0] or kept != len(want):
        same, gap = min(same, len(want) - 1), float("inf")
    return same, len(want), gap


def _launch(args: list[str], env: dict, log: Path) -> int:
    with open(log, "w", encoding="utf-8") as fh:
        return subprocess.run(args, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode


def _scores(job: Job, source, kept: Path, cache: Path) -> dict | None:
    if job.dataset.is_real:
        return None
    P = panel_scorer()
    truth = ground_truth_csv(source, job.dataset, cache)
    key, fkey = P.load_answer_key(truth["counterfactual"]), P.load_answer_key(truth["forecast"])
    l1 = P.load_layer1(kept / RS.FORECAST)
    sm = P.run_summary(P.score_scenarios(key, l1, P.load_deltas(kept / RS.SCENARIO)), fkey, l1)
    return {"counterfactual": sm["mean_accuracy_cf_wmape"], "forecast": sm["forecast_wmape"]}


def rerun_one(job: Job, source, models: Path, cache: Path) -> dict:
    root = source.fetch([f"{job.released}/*"])
    released = root / job.released
    record = json.loads((released / RS.RECORD).read_text(encoding="utf-8"))
    tuning = json.loads((released / RS.TUNING).read_text(encoding="utf-8")) if job.tuned else None
    if not all((job.folder / f).is_file() for f in DONE):
        shutil.rmtree(job.folder, ignore_errors=True)
        job.folder.mkdir(parents=True)
        status = _launch(command(job, models, record, tuning), environment(record), job.folder / "log.txt")
        if status != 0 or not all((job.folder / f).is_file() for f in DONE):
            return {"job": job, "ok": False, "problem": f"the run failed (see {show(job.folder / 'log.txt')})",
                    "verdict": None}
    work = Path(tempfile.mkdtemp(dir=cache))
    try:
        same_fc, n_fc, gap_fc = kept_rows(released / RS.FORECAST, job.folder / RS.FORECAST, 3, work / RS.FORECAST)
        same_sc, n_sc, gap_sc = kept_rows(released / RS.SCENARIO, job.folder / RS.SCENARIO, 4, work / RS.SCENARIO)
        scores = _scores(job, source, work, cache)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    rows, differ, gap = n_fc + n_sc, n_fc + n_sc - same_fc - same_sc, max(gap_fc, gap_sc)
    if not differ:
        verdict = "identical to the released run"
    elif gap <= UNITS_TOLERANCE:
        verdict = f"the same as the released run ({differ:,} of {rows:,} rows differ by at most {gap:.0e} units)"
    else:
        verdict = f"{differ:,} of {rows:,} released rows differ" + (f" (by up to {gap:.3g} units)"
                                                                   if gap != float("inf") else "")
    return {"job": job, "ok": gap <= UNITS_TOLERANCE, "scores": scores, "rows": rows, "differ": differ,
            "verdict": verdict}


def line(res: dict) -> str:
    job, s = res["job"], res.get("scores")
    numbers = f": counterfactual WMAPE {s['counterfactual']:.4f}, forecast WMAPE {s['forecast']:.4f}" if s else ""
    return f"  {job.label()}{numbers}, {res.get('verdict') or res['problem']}"


def missing_datasets(jobs: list[Job]) -> list[str]:
    return sorted({j.dataset.id for j in jobs if not (Path(OUT) / j.dataset.id / "public").is_dir()})


def run_rerun(category: str, model: str, tissue_dose: bool = False, tissue_switching: bool = False,
              source=None, jobs: list[Job] | None = None) -> int:
    jobs = jobs if jobs is not None else plan(category, model, tissue_dose, tissue_switching)
    if not jobs:
        raise UnknownNameError(f"the paper has no {model} runs on {category}")
    foundation = jobs[0].foundation
    models = models_folder()
    modules = FM_MODULES if foundation else CPU_MODULES
    info = inspect_python(modules, set(modules.values()) | set(FM_MODULES.values() if foundation else ()),
                          () if foundation else CPU_IMPORTS)
    if tuple(int(x) for x in info["python"].split(".")[:2]) < MIN_PYTHON:
        raise RerunError(f"rerun needs Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or later; this is Python {info['python']}")
    if foundation and (info["system"] != "Linux" or ("torch" in info["missing"] and not info["nvidia_smi"])):
        where = (_computer(info["system"], info["machine"]) if info["system"] != "Linux"
                 else "this computer, which has no nvidia-smi")
        raise RerunError(f"{model} needs Linux with an NVIDIA GPU (CUDA); there is none on {where}")
    if model == "tabpfn" and not os.environ.get("TABPFN_TOKEN"):
        raise RerunError("tabpfn needs a TABPFN_TOKEN, which Prior Labs issues once its TabPFN model license is "
                         "accepted: export TABPFN_TOKEN=<your token>")
    absent = [modules[m] for m in info["missing"]]
    if absent:
        req = models / ("requirements-fm.txt" if foundation else "requirements.txt")
        raise RerunError(f"{model} needs {', '.join(absent)} in this Python environment: pip install -r {req}")
    for m, err in info.get("broken", {}).items():
        if m in ("lightgbm", "xgboost") and info["system"] == "Darwin" and "libomp" in err:
            raise RerunError(f"{modules[m]} cannot be loaded without the OpenMP library on macOS: brew install libomp")
        raise RerunError(f"{modules[m]} is installed but cannot be imported in {PYTHON}: {err[-300:]}")
    if foundation and not has_gpu():
        raise RerunError(f"{model} needs an NVIDIA GPU with CUDA; torch finds none on this computer")
    absent = missing_datasets(jobs)
    if absent:
        raise RerunError(f"{count_text(len(absent), 'dataset')} not downloaded ({absent[0]}"
                         f"{', ...' if len(absent) > 1 else ''}): run causaldemand download {category} first")
    source = source if source is not None else make_source(None)
    output_folder(Path(OUTPUT_ROOT))
    print(f"causaldemand rerun {category} {model}: {count_text(len(jobs), 'run')}, written to {OUTPUT_ROOT}/.",
          flush=True)
    results, noted = [], set()
    cache = Path(tempfile.mkdtemp(prefix="causaldemand_rerun_"))
    try:
        for job in jobs:
            root = source.fetch([f"{job.released}/{RS.RECORD}"])
            record = json.loads((root / job.released / RS.RECORD).read_text(encoding="utf-8"))
            python, versions = released_versions(record, foundation)
            if set(versions) - set(info["versions"]):
                info = {**info, "versions": {**info["versions"], **inspect_python({}, versions)["versions"]}}
            diff = version_differences(info, python, versions, record.get("hardware"))
            note = f"note: the released runs used {_and(diff)}; a few predictions can differ." if diff else ""
            if note and note not in noted:
                print(note, flush=True)
                noted.add(note)
            res = rerun_one(job, source, models, cache)
            results.append(res)
            print(line(res), flush=True)
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    bad = sum(not r["ok"] for r in results)
    print()
    print(f"All {count_text(len(results), 'run')} reproduce the released predictions." if not bad else
          f"{bad:,} of {count_text(len(results), 'run')} do not reproduce the released predictions.")
    print(f"Saved to {OUTPUT_ROOT}{os.sep}")
    return 0 if not bad else 1
