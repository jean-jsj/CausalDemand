from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from causaldemand import __version__
from causaldemand._code import csv_text, file_sha256, output_folder
from causaldemand.names import Dataset, by_release_path, count_text
from causaldemand.sources import SourceError, read_sha256sums

OUT = "causaldemand_data"
FOLDER_OUT = {"inputs": "public"}
ANSWER_KEYS = {"counterfactual": "ground_truth/ground_truth_counterfactual.parquet",
               "forecast": "ground_truth/ground_truth_forecast.parquet",
               "elasticities": "ground_truth/true_elasticities.parquet"}
RECORD = "release/download.json"
MANIFEST = "release/MANIFEST.json"
HEADROOM = 1.10

_WIDTHS = {
    ("intervention_id", "string"): 34.5, ("product_id", "string"): 4.0, ("store_id", "string"): 5.0,
    ("week", "int"): 4.0, ("scope", "string"): 7.0, ("direction", "string"): 6.5,
    ("baseline_price", "fixed6"): 8.0, ("intervention_price", "fixed6"): 8.0, ("q", "fixed6"): 9.0,
    ("q_cf", "fixed6"): 9.0, ("q_realized", "fixed6"): 8.7, ("q_cf_realized", "fixed6"): 8.7,
    ("priced_product_id", "string"): 4.0, ("affected_product_id", "string"): 4.0,
    ("epsilon_star", "repr"): 19.5, ("epsilon_star_conditional", "repr"): 19.5, ("support", "bool"): 4.1,
    ("baseline_price", "repr"): 4.0, ("intervention_price", "repr"): 4.0, ("promo_cost", "repr"): 3.9,
    ("product_text", "string"): 215.5, ("brand_code", "string"): 3.0, ("chain", "string"): 4.0,
    ("household_count", "repr"): 5.4, ("price", "repr"): 4.0, ("promo_flag", "int"): 1.0,
    ("supply_cost_proxy", "repr"): 5.9, ("units", "float_of_int"): 3.8, ("dollars", "repr"): 4.9,
    ("seasonality_index", "repr"): 18.6,
}
_DEFAULT_WIDTHS = {"string": 8.0, "int": 4.0, "bool": 5.0, "repr": 18.0, "fixed6": 9.0, "sci18": 24.0,
                   "float_of_int": 4.0}


class DiskSpaceError(RuntimeError):
    pass


def local_path(released_path: str) -> str:
    folder, rest = released_path.split("/", 1)
    folder = FOLDER_OUT.get(folder, folder)
    if rest.endswith(".parquet"):
        rest = rest[: -len(".parquet")] + ".csv"
    return f"{folder}/{rest}"


def select_files(manifest: dict) -> list[dict]:
    return [rec for rec in manifest["files"]
            if rec["path"].split("/", 1)[0] in ("inputs", "release") and rec["path"] != MANIFEST]


def show(path: Path) -> str:
    try:
        rel = os.path.relpath(Path(path).resolve(), Path.cwd().resolve())
    except ValueError:
        return str(path)
    return str(path) if rel.startswith("..") else rel


def human(n: float) -> str:
    for unit, size in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= size:
            value = n / size
            digits = 2 if value < 10 else 1 if value < 100 else 0
            return f"{value:.{digits}f} {unit}"
    return f"{int(n)} B"


def estimate_csv_bytes(rec: dict) -> int:
    if "columns" not in rec:
        return int(rec.get("size_bytes", 0))
    cols = rec["columns"]
    row = sum(_WIDTHS.get((c["name"], c["csv_text"]), _DEFAULT_WIDTHS.get(c["csv_text"], 18.0)) + 1 for c in cols)
    header = sum(len(c["name"]) + 1 for c in cols)
    return int(rec["rows"] * row + header)


def _existing_parent(path: Path) -> Path:
    path = Path(path).expanduser().absolute()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(_existing_parent(path)).free


def _same_disk(a: Path, b: Path) -> bool:
    try:
        return os.stat(_existing_parent(a)).st_dev == os.stat(_existing_parent(b)).st_dev
    except OSError:
        return True


_SUMS: dict = {}


def _sha256sums(root: Path) -> dict | None:
    p = Path(root) / "SHA256SUMS"
    if not p.is_file():
        return None
    st = p.stat()
    key = (str(p), st.st_mtime_ns, st.st_size)
    if key not in _SUMS:
        _SUMS.clear()
        _SUMS[key] = read_sha256sums(Path(root))
    return _SUMS[key]


def read_manifest(source, ds: Dataset) -> dict:
    if ds.is_real:
        raise SourceError(f"{ds.id} is not in the release: it is built from the Kilts Center files "
                          f"(causaldemand download {ds.id})")
    root = source.fetch([f"{ds.release_path}/{MANIFEST}"] + (["SHA256SUMS"] if source.kind == "huggingface" else []))
    mpath = root / ds.release_path / MANIFEST
    sums = _sha256sums(root)
    if sums is not None:
        rel = f"{ds.release_path}/{MANIFEST}"
        if rel not in sums:
            raise SourceError(f"{rel} is not listed in SHA256SUMS")
        if file_sha256(mpath) != sums[rel]:
            raise SourceError(f"{rel}: SHA-256 differs from SHA256SUMS")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if manifest.get("dataset") != ds.release_name:
        raise SourceError(f"{ds.release_path}: MANIFEST.json names dataset {manifest.get('dataset')!r}, "
                          f"not {ds.release_name!r}")
    if manifest.get("dose_level") != ds.dose_level:
        raise SourceError(f"{ds.release_path}: MANIFEST.json names level {manifest.get('dose_level')!r}")
    if manifest.get("format") != "parquet":
        raise SourceError(f"{ds.release_path}: the release format is {manifest.get('format')!r}, not parquet")
    return manifest


def _previous(record_path: Path) -> dict:
    if not record_path.is_file():
        return {}
    try:
        return {f["path"]: f for f in json.loads(record_path.read_text(encoding="utf-8"))["files"]}
    except (ValueError, KeyError, TypeError):
        return {}


@dataclass
class Plan:
    ds: Dataset
    dest: Path
    manifest: dict
    wanted: list = field(default_factory=list)
    write: list = field(default_factory=list)
    kept: dict = field(default_factory=dict)
    write_bytes: int = 0
    fetch_bytes: int = 0


def plan_dataset(source, ds: Dataset, out: Path) -> Plan:
    manifest = read_manifest(source, ds)
    dest = Path(out) / ds.id
    plan = Plan(ds, dest, manifest, select_files(manifest))
    previous = _previous(dest / RECORD)
    for rec in plan.wanted:
        lp = local_path(rec["path"])
        target = dest / lp
        old = previous.get(lp)
        if (old and old.get("from_sha256") == rec["sha256"] and target.is_file()
                and target.stat().st_size == old.get("size_bytes") and file_sha256(target) == old.get("sha256")):
            plan.kept[lp] = old
        else:
            plan.write.append(rec)
            plan.write_bytes += estimate_csv_bytes(rec)
    missing = set(source.missing([f"{ds.release_path}/{rec['path']}" for rec in plan.write]))
    plan.fetch_bytes = sum(int(rec.get("size_bytes", 0)) for rec in plan.write
                           if f"{ds.release_path}/{rec['path']}" in missing)
    return plan


def check_disk(plans: list, out: Path, source, real_plans: list = ()) -> dict:
    from causaldemand import dominicks

    real = dominicks.check_cache_disk(list(real_plans), out)
    need = sum(p.write_bytes for p in plans) + sum(p.write_bytes for p in real_plans)
    if real["same_disk"]:
        need += real["cache_needed"]
    fetch = sum(p.fetch_bytes for p in plans)
    free = free_bytes(out)
    cache = source.cache_folder()
    same_disk = cache is None or _same_disk(out, cache)
    checks = {"needed": need, "free": free, "fetch": fetch, "dominicks_cache": real["cache_needed"]}
    if fetch and not same_disk:
        cache_free = free_bytes(cache)
        checks["cache_free"] = cache_free
        if fetch * HEADROOM > cache_free:
            raise DiskSpaceError(f"the files to fetch need about {human(fetch)} in the Hugging Face cache "
                                 f"({cache}), which has {human(cache_free)} free. Free space there.")
    elif fetch:
        need += fetch
        checks["needed"] = need
    if need * HEADROOM > free:
        what = ("as CSV, with the files fetched into the Hugging Face cache" if fetch and same_disk else
                "as CSV, with the Kilts Center files and the builds kept in the cache"
                if real["cache_needed"] and real["same_disk"] else "as CSV")
        raise DiskSpaceError(f"the download needs about {human(need)} {what}, and the disk of {show(out)} has "
                             f"{human(free)} free. Free some disk space, or run the command in a folder on another "
                             "disk.")
    return checks


def _write_record(dest: Path, ds: Dataset, source, files: dict) -> None:
    record = {"dataset": ds.id, "release_name": ds.release_name, "release_path": ds.release_path,
              "causaldemand_version": __version__, "source": source.record(),
              "files": [files[p] for p in sorted(files)]}
    tmp = dest / (RECORD + ".partial")
    tmp.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, dest / RECORD)


def write_dataset(source, plan: Plan, *, quiet: bool = False, prefix: str = "") -> dict:
    CT = csv_text()
    ds, dest = plan.ds, plan.dest
    say = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
    output_folder(dest.parent)
    dest.mkdir(parents=True, exist_ok=True)
    root = source.fetch([f"{ds.release_path}/{MANIFEST}"] + [f"{ds.release_path}/{rec['path']}" for rec in plan.write])
    (dest / MANIFEST).parent.mkdir(exist_ok=True)
    shutil.copyfile(root / ds.release_path / MANIFEST, dest / MANIFEST)
    kept = f", {len(plan.kept)} kept from an earlier download" if plan.kept else ""
    say(f"{prefix}{ds.id}: {count_text(len(plan.write), 'file')} to write{kept}")
    files = dict(plan.kept)
    _write_record(dest, ds, source, files)
    for rec in plan.write:
        src = root / ds.release_path / rec["path"]
        lp = local_path(rec["path"])
        target = dest / lp
        if file_sha256(src) != rec["sha256"]:
            raise SourceError(f"{ds.release_path}/{rec['path']}: SHA-256 differs from MANIFEST.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".partial")
        entry = {"path": lp, "from": rec["path"], "from_sha256": rec["sha256"]}
        if rec["path"].endswith(".parquet"):
            sink = CT.HashingWriter(tmp)
            try:
                rows = CT.write_csv_text(src, sink, rec["columns"])
            except BaseException:
                sink.close()
                tmp.unlink()
                raise
            digest = sink.close()
            if rows != rec["rows"]:
                tmp.unlink()
                raise SourceError(f"{ds.release_path}/{rec['path']}: {rows} rows, MANIFEST.json says {rec['rows']}")
            entry.update(rows=rows, sha256=digest)
        else:
            shutil.copyfile(src, tmp)
            entry["sha256"] = rec["sha256"]
        os.replace(tmp, target)
        entry["size_bytes"] = target.stat().st_size
        files[lp] = entry
        _write_record(dest, ds, source, files)
        if "rows" in entry:
            say(f"  {entry['path']}  {entry['rows']:,} rows")
    return {"dataset": ds.id, "folder": dest, "files": len(files), "written": len(plan.write)}


def download(source, datasets: list, *, quiet: bool = False) -> list[dict]:
    from causaldemand import dominicks

    say = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
    out = Path(OUT)
    synthetic = [ds for ds in datasets if not ds.is_real]
    if synthetic:
        source.fetch([f"{ds.release_path}/{MANIFEST}" for ds in synthetic]
                     + (["SHA256SUMS"] if source.kind == "huggingface" else []))
    earlier = sum((out / ds.id / RECORD).is_file() for ds in synthetic)
    if earlier:
        say(f"Checking the SHA-256 of the files of {count_text(earlier)} written by an earlier download.")
    plans ={ds.id: plan_dataset(source, ds, out) for ds in synthetic}
    real_plans = {ds.id: dominicks.plan_download(ds, out, source=source) for ds in datasets if ds.is_real}
    checks = check_disk(list(plans.values()), out, source, list(real_plans.values()))
    if plans:
        say(f"Writing {count_text(len(plans))} from {source.describe()} into {show(out)}: about "
            f"{human(sum(p.write_bytes for p in plans.values()))} as CSV ({human(checks['free'])} free).")
    results = []
    for i, ds in enumerate(datasets, 1):
        prefix = f"[{i}/{len(datasets)}] " if len(datasets) > 1 or not ds.is_real else ""
        if ds.is_real:
            results.append(dominicks.write_download(real_plans[ds.id], quiet=quiet, prefix=prefix))
        else:
            results.append(write_dataset(source, plans[ds.id], quiet=quiet, prefix=prefix))
    if len(results) == 1 and datasets[0].is_real:
        say(f"Done: {show(out)}/{datasets[0].id}/")
    else:
        written = sum(r["written"] for r in results)
        say(f"Done: {count_text(len(results))} in {show(out)} ({count_text(written, 'file')} written).")
    return results


def ground_truth_csv(source, dataset, cache: Path, kinds=("counterfactual", "forecast")) -> dict[str, Path]:
    CT = csv_text()
    ds = dataset if isinstance(dataset, Dataset) else by_release_path(str(dataset))
    unknown = [k for k in kinds if k not in ANSWER_KEYS]
    if unknown:
        raise ValueError(f"unknown ground-truth kind {unknown[0]!r}; kinds: {', '.join(ANSWER_KEYS)}")
    if ds.is_real:
        from causaldemand import dominicks
        return {"forecast": dominicks.forecast_truth(ds, source=source)} if "forecast" in kinds else {}
    manifest = read_manifest(source, ds)
    recs = {r["path"]: r for r in manifest["files"]}
    out = {}
    for kind in kinds:
        rel = ANSWER_KEYS[kind]
        rec = recs.get(rel)
        if rec is None:
            continue
        target = Path(cache) / ds.release_path / local_path(rel)
        done = target.with_name(target.name + ".sha256")
        if target.is_file() and done.is_file() and done.read_text().split()[:1] == [rec["sha256"]]:
            out[kind] = target
            continue
        root = source.fetch([f"{ds.release_path}/{rel}"])
        src = root / ds.release_path / rel
        if file_sha256(src) != rec["sha256"]:
            raise SourceError(f"{ds.release_path}/{rel}: SHA-256 differs from MANIFEST.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = CT.parquet_to_csv(src, target, rec["columns"])
        if rows != rec["rows"]:
            raise SourceError(f"{ds.release_path}/{rel}: {rows} rows, MANIFEST.json says {rec['rows']}")
        done.write_text(rec["sha256"] + "\n")
        out[kind] = target
    return out
