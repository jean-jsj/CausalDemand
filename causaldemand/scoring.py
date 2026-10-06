from __future__ import annotations

import contextlib
import gzip
import json
import math
import os
import shutil
import sys
import tempfile
import warnings
import zlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from causaldemand._code import csv_text, elasticity, file_sha256, output_folder, panel_scorer
from causaldemand.names import (CATEGORY_TITLES, UnknownNameError, by_generator_name, by_id, by_release_name,
                                count_text, select)
from causaldemand.sources import SourceError, make_source, read_sha256sums


OUT_ROOT = "causaldemand_scores"
OUTPUT_FILES = ("table.md", "table.csv", "per_dataset.csv", "per_scenario.csv", "summary.json")
FIGURE = "confounding_strength.pdf"
PREDICTION_FILES = {
    "forecast": ("forecast.csv", "forecast.csv.gz", "layer1_predictions.csv.gz", "layer1_predictions.csv"),
    "scenarios": ("scenarios.csv", "scenarios.csv.gz", "panel_deltas.csv.gz", "panel_deltas.csv"),
    "elasticities": ("elasticities.csv", "elasticities.csv.gz"),
}
PREDICTION_TEXT = {"forecast": "forecast predictions", "scenarios": "scenario predictions",
                   "elasticities": "elasticities"}
REFERENCE_SCORES = {
    "tissue": {"main/cpu_default": "results/main/cpu_default/per_run.csv",
               "main/foundation_models": "results/main/foundation_models/per_run.csv"},
    "yogurt": {"yogurt/default": "results/yogurt/default/per_run.csv",
               "yogurt/tuned": "results/yogurt/tuned/per_run.csv",
               "yogurt/foundation_models": "results/yogurt/foundation_models/per_run.csv"},
    "switching": {"switching/cpu_default": "results/switching/cpu_default/per_run.csv",
                  "switching/foundation_models": "results/switching/foundation_models/per_run.csv"},
    "strength": {"dose_response/cpu": "results/dose_response/cpu/per_run.csv",
                 "dose_response/foundation_models": "results/dose_response/foundation_models/per_run.csv",
                 "main/cpu_default": "results/main/cpu_default/per_run.csv",
                 "main/foundation_models": "results/main/foundation_models/per_run.csv"},
}
REAL_SCORES = "results/dominicks/{name}/per_run.csv"
REAL_BOOTSTRAP_SPEC = "results/dominicks/{name}/bootstrap_spec.json"
TRAIN = "inputs/transactions_train_public.parquet"
CF, FC = "mean_accuracy_cf_wmape", "forecast_wmape"
TABLE_COLUMNS = ["dataset", "seed", "n_scenarios", "complete_panel", "rows_missing_prediction",
                 "forecast_rows_missing_prediction", "status", CF, FC]
PREDICTION_COLUMNS = {"forecast": ["product_id", "store_id", "week", "predicted_units"],
                      "scenarios": ["intervention_id", "product_id", "store_id", "week", "predicted_delta_units"]}
NUMBER_COLUMNS = {"forecast": ("week", "predicted_units"), "scenarios": ("week", "predicted_delta_units")}
READ_ERRORS = (OSError, EOFError, zlib.error, TypeError)
SAME_DRAWS_NOTE = "The submitted method's intervals use the same bootstrap draws of the stores as the reference runs."
SUBMITTED = " (submitted)"


class ScoringInputError(ValueError):
    pass


@contextlib.contextmanager
def _quiet_numpy():
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def _read_error_text(exc: BaseException) -> str:
    if isinstance(exc, EOFError):
        return f"the compressed file ends early; it may be truncated ({exc})"
    if isinstance(exc, zlib.error):
        return f"the compressed data are damaged ({exc})"
    if isinstance(exc, gzip.BadGzipFile):
        if str(exc).startswith("Not a gzipped file"):
            return f"it is not a gzip file, although its name ends in .gz ({exc})"
        return f"the compressed data are damaged ({exc})"
    if isinstance(exc, UnicodeDecodeError):
        return f"it is not a text file in UTF-8 ({exc})"
    return f"{type(exc).__name__}: {exc}"


def prediction_problem(path: Path, kind: str) -> str | None:
    path = Path(path)
    name = f"{path.parent.name}/{path.name}"
    try:
        df = pd.read_csv(path, dtype=str)
    except (*READ_ERRORS, ValueError) as exc:
        return f"{name} cannot be read: {_read_error_text(exc)}"
    missing = [c for c in PREDICTION_COLUMNS[kind] if c not in df]
    if missing:
        return f"{name} lacks the column{'s' if len(missing) > 1 else ''} {', '.join(missing)}"
    for col in NUMBER_COLUMNS[kind]:
        values = df[col].str.strip()
        parsed = pd.to_numeric(values, errors="coerce")
        text = values[parsed.isna() & values.notna() & (values != "")]
        if len(text):
            return (f"{name}: {col} holds {count_text(len(text), 'value')} that "
                    f"{'is not a number' if len(text) == 1 else 'are not numbers'}, for example {text.iloc[0]!r}")
        whole = parsed.dropna()
        if col == "week" and (whole % 1 != 0).any():
            return (f"{name}: week holds a value that is not a whole number, for example "
                    f"{values[parsed.notna() & (parsed % 1 != 0)].iloc[0]!r}")
    return None


def _prediction_problems(forecast: Path, scenarios: Path) -> str | None:
    found = [p for p in (prediction_problem(forecast, "forecast"), prediction_problem(scenarios, "scenarios")) if p]
    return "; ".join(found) or None


def score_files(counterfactual_truth: Path, forecast_truth: Path, forecast: Path, scenarios: Path,
                label: str | None = None):
    P = panel_scorer()
    for p, what in ((forecast, "forecast predictions"), (scenarios, "scenario predictions")):
        if not Path(p).is_file():
            raise ScoringInputError(f"{p}: no such file ({what})")
    try:
        with _quiet_numpy():
            ps, summary = P.score_run(counterfactual_truth, forecast_truth, forecast, scenarios)
    except ValueError as exc:
        text = str(exc)
        if "Usecols" in text or "usecols" in text or "columns expected" in text:
            raise ScoringInputError(
                "a prediction file lacks a column. Forecast predictions need product_id, store_id, week, "
                "predicted_units; scenario predictions need intervention_id, product_id, store_id, week, "
                f"predicted_delta_units ({text})") from exc
        if "one-to-one" in text or "one_to_one" in text:
            raise ScoringInputError("the scenario predictions repeat an (intervention_id, product_id, store_id, "
                                    f"week) key ({text})") from exc
        raise ScoringInputError(_prediction_problems(forecast, scenarios) or text) from exc
    except READ_ERRORS as exc:
        problem = _prediction_problems(forecast, scenarios)
        if problem is None:
            raise
        raise ScoringInputError(problem) from exc
    if label is not None:
        summary = {"dataset": label, **summary}
    return ps, summary


def no_change_error(counterfactual_truth: Path) -> float:
    P = panel_scorer()
    key = P.load_answer_key(counterfactual_truth)
    layer1 = key[P.KEY].drop_duplicates().assign(predicted_units=1.0)
    deltas = key[["intervention_id"] + P.KEY].assign(predicted_delta_units=0.0)
    with _quiet_numpy():
        return P.summarize(P.score_scenarios(key, layer1, deltas))[CF]


def score_elasticities(dataset_dir: Path | None, submission: Path, truth: Path | None = None,
                       train: Path | None = None) -> dict:
    E = elasticity()
    try:
        return E.score_elasticities(dataset_dir, submission, truth_csv=truth, train_csv=train)
    except (E.SubmissionFormatError, FileNotFoundError) as exc:
        raise ScoringInputError(str(exc)) from exc


def _selection_text(category: str, tissue_dose: bool = False, tissue_switching: bool = False) -> str:
    options = [o for o, on in (("--tissue-dose", tissue_dose), ("--tissue-switching", tissue_switching)) if on]
    return category + "".join(f" {o}" for o in options)


def _folder_hint(name: str, category: str) -> str:
    try:
        return by_release_name(name).id
    except UnknownNameError:
        pass
    for variant in {"tissue": ("main", "switching"), "yogurt": ("yogurt",)}.get(category, ()):
        try:
            return by_generator_name(name, variant).id
        except UnknownNameError:
            continue
    return ""


def find_predictions(predictions_dir: Path, datasets: list) -> tuple[dict, list[str], list[str]]:
    predictions_dir = Path(predictions_dir)
    wanted = {ds.id for ds in datasets}
    files, problems, notes = {}, [], []
    for ds in datasets:
        folder = predictions_dir / ds.id
        if not folder.is_dir():
            problems.append(f"{ds.id}: no folder")
            continue
        found = {}
        for kind, names in PREDICTION_FILES.items():
            present = [n for n in names if (folder / n).is_file()]
            if len(present) > 1:
                problems.append(f"{ds.id}: holds {' and '.join(present)}; keep one file of {PREDICTION_TEXT[kind]}")
            elif present:
                found[kind] = folder / present[0]
            elif kind != "elasticities":
                problems.append(f"{ds.id}: no {PREDICTION_TEXT[kind]} ({', '.join(names[:2])} or {names[2]})")
        if "elasticities" in found and ds.is_real:
            notes.append(f"{ds.id}/{found.pop('elasticities').name} is not scored: the observed-sales datasets "
                         "have no true elasticities.")
        files[ds.id] = found
    others: dict[str, list[str]] = {}
    unknown = []
    for name in sorted(p.name for p in predictions_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if name in wanted:
            continue
        try:
            ds = by_id(name)
        except UnknownNameError:
            hint = _folder_hint(name, datasets[0].category)
            if hint in wanted:
                problems.append(f"{name}: folders are named by dataset ID; this one is {hint}")
            else:
                unknown.append(name)
            continue
        option = {"dose": "--tissue-dose", "switching": "--tissue-switching"}.get(ds.variant, "")
        others.setdefault((ds.category, option), []).append(name)
    for (category, option), names in others.items():
        add = f" Add {option} to score them." if option and category == datasets[0].category else ""
        notes.append(f"{count_text(len(names), 'folder')} of the {category}{' ' if option else ''}{option} datasets "
                     f"{'is' if len(names) == 1 else 'are'} not part of this selection and not scored.{add}")
    if unknown:
        notes.append(f"{count_text(len(unknown), 'folder')} not named by a dataset ID "
                     f"{'is' if len(unknown) == 1 else 'are'} ignored: {', '.join(unknown)}.")
    loose = [n for kind in ("forecast", "scenarios") for n in PREDICTION_FILES[kind] if (predictions_dir / n).is_file()]
    if problems and loose:
        problems.append(f"{predictions_dir.name} holds {' and '.join(loose)} itself; put the files of each dataset "
                        "in a folder named by its dataset ID")
    return files, problems, notes


def release_file(source, rel: str) -> Path:
    root = source.fetch([rel] + (["SHA256SUMS"] if source.kind == "huggingface" else []))
    path = root / rel
    sums = read_sha256sums(root)
    if sums is not None:
        if rel not in sums:
            raise SourceError(f"{rel} is not listed in SHA256SUMS")
        if file_sha256(path) != sums[rel]:
            raise SourceError(f"{rel}: SHA-256 differs from SHA256SUMS")
    return path


def reference_scores(source, table: str) -> dict[str, pd.DataFrame]:
    return {key: pd.read_csv(release_file(source, rel)) for key, rel in REFERENCE_SCORES[table].items()}


def train_revenue_csv(source, ds, folder: Path) -> Path:
    import pyarrow.parquet as pq

    from causaldemand.download import read_manifest

    CT = csv_text()
    rec = next((r for r in read_manifest(source, ds)["files"] if r["path"] == TRAIN), None)
    if rec is None:
        raise SourceError(f"{ds.release_path}: MANIFEST.json lists no {TRAIN}")
    src = source.fetch([f"{ds.release_path}/{TRAIN}"]) / ds.release_path / TRAIN
    if file_sha256(src) != rec["sha256"]:
        raise SourceError(f"{ds.release_path}/{TRAIN}: SHA-256 differs from MANIFEST.json")
    forms = {c["name"]: c["csv_text"] for c in rec["columns"]}
    table = pq.read_table(src, columns=["product_id", "dollars"])
    target = Path(folder) / ds.release_path / "train_revenue.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="") as f:
        w = CT.csv_writer(f)
        w.writerow(["product_id", "dollars"])
        w.writerows(zip(CT.format_column(table.column("product_id"), forms["product_id"]),
                        CT.format_column(table.column("dollars"), forms["dollars"])))
    return target


def _flatten(d: dict, prefix: str) -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}_{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif v is None or isinstance(v, (bool, int, float)):
            out[key] = v
    return out


def _number(x, digits: int = 4) -> str:
    try:
        return f"{float(x):.{digits}f}" if math.isfinite(float(x)) else "n/a"
    except (TypeError, ValueError):
        return "n/a"


def score_synthetic(source, ds, files: dict, cache: Path, method: str) -> dict:
    from causaldemand.download import ground_truth_csv

    kinds = ("counterfactual", "forecast") + (("elasticities",) if "elasticities" in files else ())
    truth = ground_truth_csv(source, ds, cache, kinds)
    for kind in ("counterfactual", "forecast"):
        if kind not in truth:
            raise SourceError(f"{ds.release_path}: the release holds no {kind} ground truth")
    row = {"dataset": ds.id, "release_path": ds.release_path, "method": method, "demand_model": ds.demand_model,
           "confounding": ds.confounding, "strength": ds.strength, "seed": ds.seed}
    out = {"per_scenario": None, "no_change": None, "elasticities": None}
    try:
        ps, summary = score_files(truth["counterfactual"], truth["forecast"], files["forecast"], files["scenarios"])
    except ScoringInputError as exc:
        row.update(status="invalid", invalid_reasons=f"the predictions cannot be scored: {exc}")
        line = f"invalid score: {row['invalid_reasons']}"
    else:
        row.update(summary)
        out["per_scenario"] = ps
        line = f"counterfactual WMAPE {_number(summary[CF])}, forecast WMAPE {_number(summary[FC])}"
        if summary["status"] != "valid":
            line += f"; invalid score: {summary['invalid_reasons']}"
    row.update(forecast_file=files["forecast"].name, scenarios_file=files["scenarios"].name)
    if ds.variant != "dose":
        out["no_change"] = no_change_error(truth["counterfactual"])
    if "elasticities" in files:
        row["elasticities_file"] = files["elasticities"].name
        if "elasticities" not in truth:
            row["elasticity_error"] = "the release holds no true elasticities of this dataset"
        else:
            try:
                e = score_elasticities(None, files["elasticities"], truth=truth["elasticities"],
                                       train=train_revenue_csv(source, ds, cache))
            except ScoringInputError as exc:
                row["elasticity_error"] = str(exc)
            else:
                out["elasticities"] = e
                row.update(_flatten(e, "elasticity"))
        if "elasticity_error" in row:
            line += f"\n      elasticity diagnostic not scored: {row['elasticity_error']}"
        else:
            e = out["elasticities"]
            line += (f"\n      elasticity diagnostic: own-price sign accuracy "
                     f"{_number(e['own_price']['sign_accuracy'], 3)}, own-price WMAPE "
                     f"{_number(e['own_price']['wmape'], 3)}, cross-price NDCG {_number(e['cross_price']['ndcg'], 3)}")
    out.update(row=row, line=line)
    return out


def score_real(source, ds, files: dict, cache: Path, method: str, spec: Path) -> dict:
    from causaldemand import dominicks, realdata

    folder = dominicks.built_dataset(ds, source=source, say=lambda *a: print(*a, flush=True))
    row = {"dataset": ds.id, "release_path": ds.release_path, "method": method}
    out = {"per_scenario": None, "no_change": None, "elasticities": None}
    try:
        scores, ps = realdata.submission_scores(folder, files["forecast"], files["scenarios"], reference=spec,
                                                label=method)
    except (ScoringInputError, *READ_ERRORS) as exc:
        problem = _prediction_problems(files["forecast"], files["scenarios"])
        if problem is None and isinstance(exc, ScoringInputError):
            problem = str(exc)
        if problem is None:
            raise
        row.update(status="invalid", invalid_reasons=f"the predictions cannot be scored: {problem}")
        line = f"invalid score: {row['invalid_reasons']}"
    else:
        row.update(scores)
        out["per_scenario"] = ps
        wmape, sign = realdata.table_cells(scores)
        line = f"forecast WMAPE {wmape}, sign correct (price up) {sign}"
        if scores["status"] != "valid":
            line += f"; invalid score: {scores['invalid_reasons']}"
    row.update(forecast_file=files["forecast"].name, scenarios_file=files["scenarios"].name)
    out.update(row=row, line=line)
    return out


def _method_frame(results: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame([r["row"] for _, r in results]).reindex(columns=TABLE_COLUMNS)


def _no_change(results: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame([{"dataset": ds.release_name, "seed": ds.seed, CF: r["no_change"]} for ds, r in results])


def _by_variant(results: list[tuple]) -> dict[str, list[tuple]]:
    out: dict[str, list[tuple]] = {}
    for ds, r in results:
        out.setdefault(ds.variant, []).append((ds, r))
    return out


def synthetic_tables(category: str, method: str, results: list[tuple], references: dict) -> list:
    from causaldemand import tables as T

    parts = _by_variant(results)
    main = parts["yogurt" if category == "yogurt" else "main"]
    table = T.yogurt_table if category == "yogurt" else T.main_table
    out = T.merge_notes(_reference_note(table(references[category], _no_change(main),
                                              method=(method, _method_frame(main)))))
    if "dose" in parts:
        frame = T.strength_frame(*references["strength"].values())
        mine = (method, _method_frame(main + parts["dose"]))
        out += T.merge_notes([T.strength_table(frame, T.CF, method=mine)])
    if "switching" in parts:
        switching = parts["switching"]
        out += T.merge_notes(_reference_note(T.switching_table(references["switching"], _no_change(switching),
                                                               method=(method, _method_frame(switching)))))
    return out


def _reference_note(tables: list) -> list:
    from causaldemand import tables as T

    for t in tables:
        t.notes = ["Reference models at default settings." if n == T.DEFAULT_SET_NOTE else n for n in t.notes]
    return tables


def real_table(category: str, method: str, row: dict, reference_runs: pd.DataFrame):
    from causaldemand import realdata
    from causaldemand import rescore
    from causaldemand import tables as T

    ds = select(category)[0]
    rows = [(method, *realdata.table_cells(row))] + realdata.real_table_rows(reference_runs)
    t = T.real_table(rows, rescore.real_title(ds), [T.METHOD_NOTE, *rescore.REAL_NOTES, SAME_DRAWS_NOTE])
    t.rows.insert(1, [])
    return t


def _repeats_label(tables: list, label: str) -> bool:
    for t in tables:
        seen = 0
        for r in t.rows:
            if len(r) == 1:
                seen = 0
            elif r and r[0] == label:
                seen += 1
                if seen > 1:
                    return True
    return False


def labelled_tables(build, method: str) -> tuple[list, str]:
    tables = build(method)
    if _repeats_label(tables, method):
        return build(method + SUBMITTED), method + SUBMITTED
    return tables, method


def method_numbers(results: list[tuple]) -> dict:
    from causaldemand import tables as T

    if results[0][0].is_real:
        row = results[0][1]["row"]
        keys = ("forecast_wmape", "sign_up_excl", "sign_down_excl", "sign_both_excl", "sign_up_wrong",
                "zero_share_up")
        return {k: {"value": row.get(k), "lo": row.get(f"{k}_lo"), "hi": row.get(f"{k}_hi")} for k in keys
                if k in row}

    def by_setting(part, col):
        out = {}
        for setting in sorted({ds.setting for ds, _ in part}):
            values = {ds.seed: r["row"][col] for ds, r in part if ds.setting == setting}
            mean, half = T.interval([values[s] for s in sorted(values)])
            out[setting] = {"mean": mean, "half_width": half, "seeds": {str(s): values[s] for s in sorted(values)}}
        return out

    metrics = ((CF, "counterfactual_wmape"), (FC, "forecast_wmape"))
    parts = _by_variant(results)
    out = {}
    for name, part in (("main", parts.get("main", []) + parts.get("yogurt", [])),
                       ("switching", parts.get("switching", []))):
        if part:
            out[name] = {what: by_setting(part, col) for col, what in metrics}
    if "dose" in parts:
        part = parts["main"] + parts["dose"]
        panels = {}
        for model in sorted({ds.model for ds, _ in part}):
            sub = [(ds.strength, ds.seed, r["row"][CF]) for ds, r in part if ds.model == model]
            cells = {}
            for s in sorted({s for s, _, _ in sub}):
                mean, half = T.interval([v for st, _, v in sorted(sub) if st == s])
                cells[f"{s:.2f}"] = {"mean": mean, "half_width": half}
            panels[model] = cells
        out["strength"] = {"counterfactual_wmape": panels}
    return out


def table_cells(tables: list) -> pd.DataFrame:
    from causaldemand import tables as T

    out = []
    for t in tables:
        header = t.header_rows()
        cols = ([f"{u}, {lo}" if u and lo else (u or lo) for u, lo in zip(*header)] if len(header) == 2
                else list(header[0]))
        panel = ""
        for r, cells in zip(t.rows, T.label_cells(t)):
            if len(r) == 1:
                panel = r[0]
            elif r:
                label = " / ".join(c for c in cells if c)
                out += [{"table": t.title, "panel": panel, "row": label, "column": c, "value": v}
                        for c, v in zip(cols[t.label_columns:], r[t.label_columns:])]
    return pd.DataFrame(out, columns=["table", "panel", "row", "column", "value"])


def _json_value(x):
    if isinstance(x, dict):
        return {str(k): _json_value(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_value(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    return x


def _strength_figure(path: Path, references: dict) -> bool:
    from causaldemand import figure as F
    from causaldemand import tables as T

    try:
        F.strength_figure(F.strength_points(T.strength_frame(*references["strength"].values())), path)
    except ImportError:
        print("The figure is not drawn: matplotlib is not installed (pip install matplotlib).")
        return False
    return True


def write_results(out: Path, tables: list, per_dataset: pd.DataFrame, per_scenario: pd.DataFrame, summary: dict,
                  markdown: str) -> None:
    out = output_folder(Path(out))
    for name in OUTPUT_FILES + (FIGURE,):
        if (out / name).exists():
            (out / name).unlink()
    (out / "table.md").write_text(markdown, encoding="utf-8")
    table_cells(tables).to_csv(out / "table.csv", index=False)
    per_dataset.to_csv(out / "per_dataset.csv", index=False)
    per_scenario.to_csv(out / "per_scenario.csv", index=False)
    (out / "summary.json").write_text(json.dumps(_json_value(summary), indent=1) + "\n", encoding="utf-8")


def _per_scenario(results: list[tuple], method: str) -> pd.DataFrame:
    frames = []
    for ds, r in results:
        ps = r["per_scenario"]
        if ps is None:
            continue
        ps = ps.drop(columns=[c for c in ("arm", "model") if c in ps])
        ps.insert(0, "method", method)
        ps.insert(0, "dataset", ds.id)
        frames.append(ps)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["dataset", "method"])


def earlier_results(out: Path) -> str | None:
    out = Path(out)
    if not any((out / name).exists() for name in OUTPUT_FILES):
        return None
    try:
        s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        return _selection_text(s["category"], s.get("tissue_dose", False), s.get("tissue_switching", False))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return ""


def _invalid_text(n_invalid: int, n: int) -> str:
    if n == 1:
        return "the score is invalid"
    return f"{n_invalid} of {n} scores {'is' if n_invalid == 1 else 'are'} invalid"


def _replaced_line(method: str, earlier: str | None) -> str | None:
    if earlier is None:
        return None
    return f"Replaced the earlier results of {method}" + (f" ({earlier})." if earlier else ".")


def run_score(category: str, predictions_dir: Path, tissue_dose: bool = False, tissue_switching: bool = False,
              source=None) -> int:
    from causaldemand import __version__
    from causaldemand import tables as T
    from causaldemand.download import show

    if category == "all":
        raise UnknownNameError("score takes one category per command: tissue, yogurt, cereal, snack-crackers.")
    datasets = select(category, tissue_dose, tissue_switching)
    source = make_source(None) if source is None else source
    predictions_dir = Path(predictions_dir)
    method = Path(os.path.abspath(predictions_dir)).name
    if not method:
        raise UnknownNameError(f"{predictions_dir} cannot name a method; give the folder of the predictions by its "
                               "name, for example: causaldemand score tissue my_method")
    selection = _selection_text(category, tissue_dose, tissue_switching)
    real = datasets[0].is_real
    what = f"{count_text(len(datasets))} of {CATEGORY_TITLES[category]}"
    print(f"Scoring the method {method} on {selection}: {what}; release: {source.describe()}", flush=True)
    if not predictions_dir.is_dir():
        raise ScoringInputError(f"{predictions_dir}: no such folder (the folder of {method}'s predictions)")
    out = Path.cwd() / OUT_ROOT / method / category
    earlier = earlier_results(out)
    release = source.describe() if source.kind == "huggingface" else "a local copy of the release"
    head = [f"# Scores of {method}: {selection}", "", f"{what}; release: {release}; causaldemand {__version__}.", ""]
    summary = {"method": method, "row_label": None, "category": category, "tissue_dose": bool(tissue_dose),
               "tissue_switching": bool(tissue_switching), "package_version": __version__,
               "scorer_version": panel_scorer().SCORER_VERSION, "source": source.record(),
               "datasets": len(datasets)}

    files, problems, notes = find_predictions(predictions_dir, datasets)
    for note in notes:
        print(f"note: {note}", flush=True)
    if problems:
        why = "" if real else ", since every number is a mean over the five seeds"
        print(f"The submission is incomplete. Every dataset of {selection} needs a folder named by its dataset ID "
              f"with {PREDICTION_FILES['forecast'][0]} and {PREDICTION_FILES['scenarios'][0]}{why}. "
              f"{count_text(len(problems), 'problem')}:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        markdown = "\n".join(head + [f"No table: the submission is incomplete; nothing is scored. "
                                      f"{count_text(len(problems), 'problem')}:", ""]
                             + [f"- {p}" for p in problems] + [""])
        per_dataset = pd.DataFrame([{"dataset": ds.id, "release_path": ds.release_path, "method": method,
                                     "status": "not scored",
                                     "problem": "; ".join(p[len(ds.id) + 2:] for p in problems
                                                          if p.startswith(f"{ds.id}: "))} for ds in datasets])
        summary.update(datasets_valid=0, status="incomplete", exit_status=1, problems=problems, invalid={},
                       table_error=None, elasticity_errors={}, method_scores=None, elasticities={}, tables=[])
        output_folder(out.parent.parent)
        write_results(out, [], per_dataset, pd.DataFrame(columns=["dataset", "method"]), summary, markdown)
        print("Nothing was scored.", file=sys.stderr)
        replaced = _replaced_line(method, earlier)
        if replaced:
            print(replaced, file=sys.stderr)
        print(f"Saved the list of problems to {show(out)}{os.sep}: {', '.join(OUTPUT_FILES)}", file=sys.stderr,
              flush=True)
        return 1

    references, reference_runs, spec = {}, None, None
    if real:
        name = datasets[0].release_name
        reference_runs = pd.read_csv(release_file(source, REAL_SCORES.format(name=name)))
        spec = release_file(source, REAL_BOOTSTRAP_SPEC.format(name=name))
    else:
        needed = [category] + ["strength"] * bool(tissue_dose) + ["switching"] * bool(tissue_switching)
        references = {t: reference_scores(source, t) for t in needed}

    if real:
        from causaldemand import dominicks
        from causaldemand.download import DiskSpaceError
        try:
            dominicks.built_dataset(datasets[0], source=source, say=lambda *a: print(*a, flush=True))
        except (dominicks.KiltsError, DiskSpaceError) as exc:
            reason = f"the dataset cannot be built from the Kilts Center files: {exc}"
            print(f"{datasets[0].id} is not scored: {reason}", file=sys.stderr)
            markdown = "\n".join(head + [f"No table: {datasets[0].id} is not scored: {reason}", ""])
            per_dataset = pd.DataFrame([{"dataset": ds.id, "release_path": ds.release_path, "method": method,
                                         "status": "not scored", "problem": reason} for ds in datasets])
            summary.update(datasets_valid=0, status="not scored", exit_status=1, problems=[reason], invalid={},
                           table_error=None, elasticity_errors={}, method_scores=None, elasticities={}, tables=[])
            output_folder(out.parent.parent)
            write_results(out, [], per_dataset, pd.DataFrame(columns=["dataset", "method"]), summary, markdown)
            print("Nothing was scored.", file=sys.stderr)
            replaced = _replaced_line(method, earlier)
            if replaced:
                print(replaced, file=sys.stderr)
            print(f"Saved the reason to {show(out)}{os.sep}: {', '.join(OUTPUT_FILES)}", file=sys.stderr, flush=True)
            return 1

    results = []
    width = len(str(len(datasets)))
    with tempfile.TemporaryDirectory(prefix="causaldemand-score-") as tmp:
        cache = Path(tmp)
        for i, ds in enumerate(datasets, 1):
            if real:
                r = score_real(source, ds, files[ds.id], cache, method, spec)
            else:
                r = score_synthetic(source, ds, files[ds.id], cache, method)
            for p in cache.iterdir():
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
            results.append((ds, r))
            print(f"[{i:>{width}}/{len(datasets)}] {ds.id}: {r['line']}", flush=True)

    invalid = {ds.id: r["row"].get("invalid_reasons", "") for ds, r in results if r["row"].get("status") != "valid"}
    elasticity_errors = {ds.id: r["row"]["elasticity_error"] for ds, r in results if "elasticity_error" in r["row"]}
    tables, table_error, label = [], None, None
    if not invalid:
        try:
            if real:
                tables, label = labelled_tables(
                    lambda name: [real_table(category, name, results[0][1]["row"], reference_runs)], method)
            else:
                tables, label = labelled_tables(
                    lambda name: synthetic_tables(category, name, results, references), method)
        except T.TableError as exc:
            table_error = str(exc)
    status = 1 if invalid or table_error or elasticity_errors else 0
    if real:
        T.merge_notes(tables or [])

    if tables:
        markdown = "\n".join(head) + "\n" + "\n".join(T.to_markdown(t) for t in tables)
    elif invalid:
        markdown = "\n".join(head + [
            f"No table: {_invalid_text(len(invalid), len(datasets))}, and the table needs a valid score on every "
            "dataset. per_dataset.csv gives the reasons.", ""])
    else:
        markdown = "\n".join(head + [f"No table: {table_error}", ""])
    summary.update(
        row_label=label, datasets_valid=len(datasets) - len(invalid), status="invalid" if invalid else "valid",
        exit_status=status, problems=[], invalid=invalid, table_error=table_error,
        elasticity_errors=elasticity_errors, method_scores=None if invalid else method_numbers(results),
        elasticities={ds.id: r["elasticities"] for ds, r in results if r["elasticities"] is not None},
        tables=[asdict(t) for t in tables])
    output_folder(out.parent.parent)
    write_results(out, tables, pd.DataFrame([r["row"] for _, r in results]), _per_scenario(results, method),
                  summary, markdown)

    for t in tables:
        print("")
        print(T.to_text(t), end="")
    print("")
    if label and label != method:
        print(f"The method's row is labelled {label}, since a reference row is labelled {method}.")
    if invalid:
        print(f"{_invalid_text(len(invalid), len(datasets)).capitalize()}; no table is built, since the table needs "
              "a valid score on every dataset:")
        for ds_id, reasons in invalid.items():
            print(f"  {ds_id}: {reasons}")
    if table_error:
        print(f"The table cannot be built: {table_error}")
    if elasticity_errors:
        print(f"The elasticity diagnostic of {count_text(len(elasticity_errors))} is not scored:")
        for ds_id, reasons in elasticity_errors.items():
            print(f"  {ds_id}: {reasons}")
    saved = list(OUTPUT_FILES)
    if tables and tissue_dose and _strength_figure(out / FIGURE, references):
        saved.append(FIGURE)
    replaced = _replaced_line(method, earlier)
    if replaced:
        print(replaced)
    print(f"Saved to {show(out)}{os.sep}: {', '.join(saved)}", flush=True)
    return status
