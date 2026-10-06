from __future__ import annotations

import hashlib
import http.client
import json
import os
import shutil
import sys
import tempfile
import textwrap
import time
import types
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from causaldemand import __version__
from causaldemand._code import file_sha256, output_folder
from causaldemand.download import HEADROOM, DiskSpaceError, free_bytes, human, show
from causaldemand.names import REAL_DATASETS, Dataset, UnknownNameError, by_id, count_text
from causaldemand.sources import SourceError

KILTS_PAGE = "https://www.chicagobooth.edu/research/kilts/research-data/dominicks"
_BASE = ("https://www.chicagobooth.edu/research/kilts/research-data/-/media/enterprise/centers/kilts/datasets/"
         "dominicks-dataset")
NOTICE = ("The Dominick's data are distributed by the Kilts Center for Marketing, University of Chicago Booth School "
          "of Business, for academic research only; publications must acknowledge the Kilts Center.")
SOURCE_KIND = "built from the Kilts Center files"
PAUSE = 2.0
TIMEOUT = 300
STALE = 6 * 3600
KILTS_FOLDER, BUILT_FOLDER = "kilts_files", "built"
BUILD_RECORD = "build.json"
DOWNLOAD_RECORD = "release/download.json"
SCORING_PARAMS = "release/scoring_params.json"
FORECAST_TRUTH = "ground_truth/ground_truth_forecast.csv"
INPUTS = ("public/counterfactual_sweep_context_panel.csv", "public/products_public.csv", "public/stores_public.csv",
          "public/transactions_holdout_context_public.csv", "public/transactions_train_public.csv")
DOWNLOADED = tuple(sorted(INPUTS + (SCORING_PARAMS,)))
SCORING_PARAMS_TEXT = json.dumps({"benchmark_family": {"active_cell": {"model_family": "real_data"}},
                                  "simulation": {"counterfactual_eval_weeks": 16}}, indent=2) + "\n"


@dataclass(frozen=True)
class KiltsFile:
    name: str
    url: str
    size: int
    sha256: str


KILTS_FILES = {f.name: f for f in (
    KiltsFile("wcer.zip", f"{_BASE}/movement_csv-files/wcer.zip", 42_402_094,
              "2d4a59f88e4b97257c566f7c62fcccfdf9d5971891a617316ab21ea61c8d24c2"),
    KiltsFile("upccer.csv", f"{_BASE}/upc_csv-files/upccer.csv", 25_932,
              "affa589f080ab18a2edb2fa8f4cfb52b9212dec353bb49778e57f2b2403c5c6c"),
    KiltsFile("wsna.zip", f"{_BASE}/movement_csv-files/wsna.zip", 27_789_565,
              "9a1d3f078efa12625ef9b4b3c9d4482aa643634a7740d83d0a8f56222a1d5be6"),
    KiltsFile("upcsna.csv", f"{_BASE}/upc_csv-files/upcsna.csv", 22_496,
              "5ce04b56bfd5c9c71d7e3e0f799acc4fcd7253326ae3c8443bd307e0b9518f12"),
    KiltsFile("demo_stata.zip",
              "https://www.chicagobooth.edu/boothsitecore/docs/dff/store-demos-customer-count/demo_stata.zip", 168_854,
              "d5e1464fecbb14d68f5168a222fdd133466b4d17b01c9e14724788d079a8ed9a"),
)}


@dataclass(frozen=True)
class KiltsDataset:
    name: str
    code: str
    subject: str
    n_stores: int
    n_products: int
    weeks: tuple
    outputs: dict = field(default_factory=dict)

    @property
    def files(self) -> tuple:
        return (f"w{self.code}.zip", f"upc{self.code}.csv", "demo_stata.zip")

    @property
    def title(self) -> str:
        return f"{self.subject}, observed sales of Dominick's Finer Foods"


KILTS_DATASETS = {
    "cereal": KiltsDataset("cereal", "cer", "ready-to-eat cereals", 81, 60, (63, 218), {
        "public/counterfactual_sweep_context_panel.csv":
            ("8a51520894fd2cee36609258019b3729fc9f270f82bcfc1539c5c2c9b352025d", 156_077_370),
        "public/products_public.csv": ("5f4665117e03e50d36d14a1b7efdb004563bc64ae6ba638d312d19fb5590ba37", 1_046),
        "public/stores_public.csv": ("caece2de06e1b4a8aa4e516ab4114c9aba121564db64910adb64a7998eb9f220", 1_327),
        "public/transactions_holdout_context_public.csv":
            ("e4e0bb51137ec12011486acfcd8428be811a89df86575f08acee1fd82cc22617", 2_570_460),
        "public/transactions_train_public.csv":
            ("bc1246d91a545c2d6b76a48587acfbe9edef0c131a2db507444cf3fd15d92afd", 27_904_208),
        FORECAST_TRUTH: ("f6f5128e5192739483da63c678015d179e24b3ef80b032231b6c2c96c6c9b960", 1_849_717)}),
    "snack_crackers": KiltsDataset("snack_crackers", "sna", "snack crackers", 82, 60, (63, 218), {
        "public/counterfactual_sweep_context_panel.csv":
            ("c0154314e0336fda590ea3f7f3fa95c4936d71e1468079def2f0d35741ac336a", 154_001_477),
        "public/products_public.csv": ("a3a841401b8540a1802cfbaabbb1460aab4061183e26b1949fd71f31204e01f8", 924),
        "public/stores_public.csv": ("c9820d2f5d5a0a19dc21c679347e0f81c26b3c7a3c71c0c1eb5ea869088176d9", 1_343),
        "public/transactions_holdout_context_public.csv":
            ("5913eb518f681d94190786525454e78fec74afcbe01ce925b42f1ecd92aa75f6", 2_535_172),
        "public/transactions_train_public.csv":
            ("bfcf9c29ec31c326e799bcaef9e0f8895b4dee7dadbf91f84a83c8517001e315", 24_882_336),
        FORECAST_TRUTH: ("3dc7280f601bd81f5641e5107ca2f434068cdf712ac78210e17b5f10b5d21765", 1_760_183)}),
}
CHECKED_WITH = "Python 3.9 (pandas 2.3, numpy 1.24) and Python 3.12 (pandas 3.0, numpy 2.5)"


class KiltsError(SourceError):
    pass


def kilts_dataset(dataset) -> KiltsDataset:
    if isinstance(dataset, KiltsDataset):
        return dataset
    if isinstance(dataset, Dataset):
        name = dataset.release_name if dataset.is_real else None
    else:
        text = str(dataset)
        name = REAL_DATASETS[text][0] if text in REAL_DATASETS else text
    if name not in KILTS_DATASETS:
        raise UnknownNameError(f"{dataset!s} is not built from the Kilts Center files (cereal, snack-crackers)")
    return KILTS_DATASETS[name]


def cache_folder(cache_dir=None) -> Path:
    if cache_dir:
        return Path(cache_dir).expanduser() / "dominicks"
    return Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache").expanduser() / "causaldemand" / "dominicks"


def source_cache(source=None, cache_dir=None) -> Path:
    return cache_folder(cache_dir if cache_dir is not None else getattr(source, "cache_dir", None))


class _Screen:
    def __init__(self, say, indent: str = "  "):
        self.say, self.indent, self.open = say, indent, False

    def line(self, text: str) -> None:
        if self.say is not None:
            self.close()
            self.say(textwrap.fill(text, width=110, initial_indent=self.indent, subsequent_indent=self.indent,
                                   break_on_hyphens=False, break_long_words=False))

    def begin(self, text: str) -> None:
        if self.say is not None:
            self.close()
            print(f"{self.indent}{text} ...", end="", flush=True)
            self.open = True

    def end(self, text: str) -> None:
        if self.say is not None and self.open:
            print(f" {text}", flush=True)
            self.open = False

    def close(self) -> None:
        if self.open:
            self.end("")


_notice_shown = False


def _notice(screen: _Screen) -> None:
    global _notice_shown
    if screen.say is not None and not _notice_shown:
        screen.line(NOTICE)
        _notice_shown = True


_checked: dict = {}
_failed: dict = {}
_last_request: list = []


def _age(path: Path) -> float:
    times = [path.stat().st_mtime]
    if path.is_dir():
        times += [p.stat().st_mtime for p in path.rglob("*")]
    return time.time() - max(times)


def _remove_stale(paths) -> None:
    for p in paths:
        try:
            if _age(p) <= STALE:
                continue
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
        except OSError:
            pass


def _sha256_of(path: Path) -> str:
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _checked:
        _checked[key] = file_sha256(path)
    return _checked[key]


def _has(path: Path, sha256: str, size: int) -> bool:
    try:
        return path.is_file() and path.stat().st_size == size and _sha256_of(path) == sha256
    except OSError:
        return False


def cached_kilts_files(names, cache: Path) -> list[str]:
    return [n for n in names if _has(cache / KILTS_FOLDER / n, KILTS_FILES[n].sha256, KILTS_FILES[n].size)]


def _urlopen(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": f"causaldemand/{__version__} (research software)"})
    return urllib.request.urlopen(request, timeout=TIMEOUT)


def _download(kf: KiltsFile, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f"{target.name}.", suffix=".partial", dir=target.parent)
    tmp = Path(name)
    h, size = hashlib.sha256(), 0
    try:
        with open(fd, "wb") as f, _urlopen(kf.url) as response:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                h.update(chunk)
                f.write(chunk)
                size += len(chunk)
    except (OSError, urllib.error.URLError, http.client.HTTPException, ValueError) as exc:
        tmp.unlink(missing_ok=True)
        raise KiltsError(f"{kf.name}: the download from the Kilts Center website failed ({exc}).") from exc
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        _last_request[:] = [time.monotonic()]
    digest = h.hexdigest()
    if size != kf.size or digest != kf.sha256:
        tmp.unlink(missing_ok=True)
        raise KiltsError(f"{kf.name}: the file from the Kilts Center website is not the file the datasets are "
                         f"built from ({size:,} bytes, SHA-256 {digest[:16]}...; expected {kf.size:,} bytes, "
                         f"SHA-256 {kf.sha256[:16]}...). The Kilts Center may have changed or moved it; the "
                         "original file is needed.")
    os.replace(tmp, target)
    _checked[(str(target), target.stat().st_size, target.stat().st_mtime_ns)] = digest


def _wait(pause: float) -> None:
    if pause and _last_request:
        rest = pause - (time.monotonic() - _last_request[0])
        if rest > 0:
            time.sleep(rest)


def _by_hand(names, folder: Path) -> str:
    files = "; ".join(f"{n} from {KILTS_FILES[n].url}" for n in names)
    return (f"Save {'this file' if len(names) == 1 else 'these files'} into {folder} and run the command again: "
            f"{files} (the Kilts Center page: {KILTS_PAGE}).")


def fetch(names, cache: Path, *, say=None, indent: str = "  ", pause: float | None = None) -> Path:
    pause = PAUSE if pause is None else pause
    folder = Path(cache) / KILTS_FOLDER
    names = list(dict.fromkeys(names))
    have = set(cached_kilts_files(names, cache))
    missing = [n for n in names if n not in have]
    screen = _Screen(say, indent)
    if not missing:
        screen.begin(f"the Kilts Center files are in the cache: {', '.join(names)}")
        screen.end("checked")
        return folder
    known = [_failed[("file", str(folder), n)] for n in missing if ("file", str(folder), n) in _failed]
    if known:
        raise KiltsError(f"{known[0]} {_by_hand(missing, folder)}")

    def sized(n: str) -> str:
        size = KILTS_FILES[n].size
        return f"{n} ({size / 1e6:.0f} MB)" if size >= 1e6 else n
    in_cache = f"; in the cache: {', '.join(n for n in names if n in have)}" if have else ""
    if folder.is_dir():
        _remove_stale(folder.glob("*.partial"))
    screen.begin(f"downloading from the Kilts Center website: {', '.join(sized(n) for n in missing)}")
    try:
        for n in missing:
            _wait(pause)
            try:
                _download(KILTS_FILES[n], folder / n)
            except KiltsError as exc:
                _failed[("file", str(folder), n)] = str(exc)
                lacking = [m for m in missing if not _has(folder / m, KILTS_FILES[m].sha256, KILTS_FILES[m].size)]
                raise KiltsError(f"{exc} {_by_hand(lacking, folder)}") from exc
    except BaseException:
        screen.end("failed")
        raise
    screen.end(f"checked{in_cache}")
    return folder


def check_built(folder: Path, dataset) -> dict:
    kd = kilts_dataset(dataset)
    out = {}
    for rel, (expected, size) in kd.outputs.items():
        p = Path(folder) / rel
        got = _sha256_of(p) if p.is_file() else None
        out[rel] = {"sha256": got, "expected": expected, "size_bytes": p.stat().st_size if got else None,
                    "identical": got == expected}
    return out


def _checked_build(folder: Path, kd: KiltsDataset) -> dict | None:
    if not (Path(folder) / BUILD_RECORD).is_file():
        return None
    files = check_built(folder, kd)
    return files if all(c["identical"] for c in files.values()) else None


def _run_build(kd: KiltsDataset, raw: Path, out: Path) -> dict:
    from causaldemand import _dff_build

    args = types.SimpleNamespace(category=kd.code, raw=str(raw), out=str(out), n_products=kd.n_products,
                                 last_week=kd.weeks[1], brands=None)
    try:
        return _dff_build.build(args)
    except SystemExit as exc:
        raise KiltsError(f"the build of {kd.name} from the Kilts Center files stopped: {exc}") from None
    except Exception as exc:
        memory = " The build needs about 4 GB of memory." if isinstance(exc, MemoryError) else ""
        raise KiltsError(f"the build of {kd.name} from the Kilts Center files stopped on an error "
                         f"({type(exc).__name__}: {exc}; {_versions()}).{memory}") from exc


def _versions() -> str:
    import numpy as np
    import pandas as pd
    return f"Python {sys.version.split()[0]}, pandas {pd.__version__}, numpy {np.__version__}"


def _check_space(kd: KiltsDataset, cache: Path) -> None:
    need = cache_needs([kd], cache)
    free = free_bytes(cache)
    if need and need * HEADROOM > free:
        raise DiskSpaceError(f"building {kd.name} needs about {human(need)} in the Dominick's cache ({show(cache)}: "
                             f"the Kilts Center files and the build), and its disk has {human(free)} free. Free "
                             "space there.")


def _install(new: Path, folder: Path, kd: KiltsDataset, work: Path) -> None:
    for i in range(3):
        try:
            os.replace(new, folder)
            return
        except OSError:
            if _checked_build(folder, kd) is not None:
                return
            try:
                os.replace(folder, work / f"replaced_{i}")
            except OSError:
                pass
    raise KiltsError(f"the checked build of {kd.name} cannot be moved to {folder}")


def prepare(dataset, *, source=None, cache_dir=None, cache: Path | None = None, say=None, indent: str = "  ",
            brief: bool = True) -> dict:
    kd = kilts_dataset(dataset)
    cache = Path(cache) if cache is not None else source_cache(source, cache_dir)
    folder = cache / BUILT_FOLDER / kd.name
    screen = _Screen(say, indent)
    t0 = time.time()
    if folder.parent.is_dir():
        _remove_stale(folder.parent.glob(".build_*"))
    files = _checked_build(folder, kd)
    if files is not None:
        if not brief:
            screen.begin("using the dataset built earlier (in the cache); checking it against the files the "
                         "reference models used")
            screen.end("identical")
        return {"folder": folder, "cache": cache, "reused": True, "downloaded": [], "seconds": time.time() - t0,
                "files": files, "ok": True}
    key = ("build", str(cache), kd.name)
    if key in _failed:
        raise KiltsError(_failed[key])
    _check_space(kd, cache)
    _notice(screen)
    try:
        downloaded = _build(kd, cache, folder, screen, say, indent)
    except KiltsError as exc:
        _failed[key] = str(exc)
        raise
    return {"folder": folder, "cache": cache, "reused": False, "downloaded": downloaded, "seconds": time.time() - t0,
            "files": check_built(folder, kd), "ok": True}


def _build(kd: KiltsDataset, cache: Path, folder: Path, screen: _Screen, say, indent: str) -> list:
    before = set(cached_kilts_files(kd.files, cache))
    raw = fetch(kd.files, cache, say=say, indent=indent)
    downloaded = [n for n in kd.files if n not in before]
    built = cache / BUILT_FOLDER
    built.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".build_{kd.name}_", dir=built))
    try:
        screen.begin(f"building the dataset ({kd.n_stores} stores, {kd.n_products} products, weeks "
                     f"{kd.weeks[0]}-{kd.weeks[1]})")
        t = time.time()
        try:
            _run_build(kd, raw, work / "dataset")
        except BaseException:
            screen.end("failed")
            raise
        screen.end(f"done in {time.time() - t:.0f} s")
        screen.begin("checking the built files against the files the reference models used")
        files = check_built(work / "dataset", kd)
        differ = [rel for rel, c in files.items() if not c["identical"]]
        if differ:
            screen.end("they differ")
            raise KiltsError(f"the {kd.name} dataset built from the Kilts Center files differs from the files the "
                             f"reference models used in {', '.join(differ)} ({_versions()}); the build is not used. "
                             f"The builds were checked with {CHECKED_WITH}.")
        screen.end("identical")
        record = {"dataset": kd.name, "causaldemand_version": __version__, "versions": _versions(),
                  "kilts_files": {n: KILTS_FILES[n].sha256 for n in kd.files},
                  "files": {rel: c["sha256"] for rel, c in files.items()}}
        (work / "dataset" / BUILD_RECORD).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        _install(work / "dataset", folder, kd, work)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return downloaded


def built_dataset(dataset, *, source=None, cache_dir=None, say=None, indent: str = "  ") -> Path:
    return prepare(dataset, source=source, cache_dir=cache_dir, say=say, indent=indent, brief=True)["folder"]


def forecast_truth(dataset, *, source=None, cache_dir=None, say=None) -> Path:
    return built_dataset(dataset, source=source, cache_dir=cache_dir, say=say) / FORECAST_TRUTH


@dataclass
class Plan:
    ds: Dataset
    kd: KiltsDataset
    dest: Path
    cache: Path
    wanted: list = field(default_factory=list)
    write: list = field(default_factory=list)
    kept: list = field(default_factory=list)
    write_bytes: int = 0
    build_cached: bool = False
    kilts_missing: list = field(default_factory=list)
    build_bytes: int = 0


def _expected(kd: KiltsDataset, rel: str) -> tuple:
    if rel == SCORING_PARAMS:
        data = SCORING_PARAMS_TEXT.encode("utf-8")
        return hashlib.sha256(data).hexdigest(), len(data)
    return kd.outputs[rel]


def plan_download(dataset, out: Path, *, source=None, cache_dir=None) -> Plan:
    ds = dataset if isinstance(dataset, Dataset) else by_id(str(dataset))
    kd = kilts_dataset(ds)
    cache = source_cache(source, cache_dir)
    dest = Path(out) / ds.id
    plan = Plan(ds, kd, dest, cache, list(DOWNLOADED))
    for rel in plan.wanted:
        sha, size = _expected(kd, rel)
        if _has(dest / rel, sha, size):
            plan.kept.append(rel)
        else:
            plan.write.append(rel)
            plan.write_bytes += size
    if any(rel != SCORING_PARAMS for rel in plan.write):
        folder = cache / BUILT_FOLDER / kd.name
        plan.build_cached = (folder / BUILD_RECORD).is_file() and all(
            c["identical"] for c in check_built(folder, kd).values())
        if not plan.build_cached:
            plan.kilts_missing = [n for n in kd.files if n not in cached_kilts_files(kd.files, cache)]
            plan.build_bytes = sum(size for _, size in kd.outputs.values())
    else:
        plan.build_cached = True
    return plan


def cache_needs(datasets, cache: Path) -> int:
    names, builds = set(), 0
    for ds in datasets:
        kd = kilts_dataset(ds)
        folder = Path(cache) / BUILT_FOLDER / kd.name
        if (folder / BUILD_RECORD).is_file() and all(c["identical"] for c in check_built(folder, kd).values()):
            continue
        names.update(n for n in kd.files if n not in cached_kilts_files(kd.files, cache))
        builds += sum(size for _, size in kd.outputs.values())
    return sum(KILTS_FILES[n].size for n in names) + builds


def cache_bytes(plans: list) -> int:
    kilts = {n for p in plans for n in p.kilts_missing}
    return sum(KILTS_FILES[n].size for n in kilts) + sum(p.build_bytes for p in plans)


def _write_record(plan: Plan, files: dict) -> None:
    record = {"dataset": plan.ds.id, "release_name": plan.ds.release_name, "causaldemand_version": __version__,
              "source": {"kind": SOURCE_KIND, "kilts_center": KILTS_PAGE, "notice": NOTICE,
                         "kilts_files": [{"name": n, "url": KILTS_FILES[n].url, "size_bytes": KILTS_FILES[n].size,
                                          "sha256": KILTS_FILES[n].sha256} for n in plan.kd.files],
                         "build": "causaldemand/_dff_build.py"},
              "files": [files[p] for p in sorted(files)]}
    target = plan.dest / DOWNLOAD_RECORD
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".partial")
    tmp.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, target)


def _entry(rel: str, path: Path, sha: str) -> dict:
    entry = {"path": rel, "sha256": sha, "size_bytes": path.stat().st_size}
    if rel.endswith(".csv"):
        with open(path, "rb") as f:
            entry["rows"] = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 20), b"")) - 1
    return entry


def write_download(plan: Plan, *, quiet: bool = False, prefix: str = "") -> dict:
    say = None if quiet else (lambda *a: print(*a, flush=True))
    screen = _Screen(say, "  ")
    ds, kd, dest = plan.ds, plan.kd, plan.dest
    if say:
        say(f"{prefix}{ds.id}: {kd.title}")
    _notice(screen)
    folder = None
    if any(rel != SCORING_PARAMS for rel in plan.write):
        folder = prepare(ds, cache=plan.cache, say=say, brief=False)["folder"]
    elif plan.kept:
        screen.line(f"{count_text(len(plan.kept), 'file')} written by an earlier download and unchanged (kept)")
    output_folder(dest.parent)
    dest.mkdir(parents=True, exist_ok=True)
    previous = {}
    try:
        previous = {f["path"]: f for f in json.loads((dest / DOWNLOAD_RECORD).read_text(encoding="utf-8"))["files"]}
    except (OSError, ValueError, KeyError, TypeError):
        pass
    files = {rel: previous.get(rel) or _entry(rel, dest / rel, _expected(kd, rel)[0]) for rel in plan.kept}
    _write_record(plan, files)
    for rel in plan.write:
        sha, size = _expected(kd, rel)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".partial")
        if rel == SCORING_PARAMS:
            tmp.write_bytes(SCORING_PARAMS_TEXT.encode("utf-8"))
        else:
            shutil.copyfile(folder / rel, tmp)
        if file_sha256(tmp) != sha:
            tmp.unlink()
            raise KiltsError(f"{ds.id}/{rel}: the written file differs from the checked build")
        os.replace(tmp, target)
        files[rel] = _entry(rel, target, sha)
        _write_record(plan, files)
    return {"dataset": ds.id, "folder": dest, "files": len(files), "written": len(plan.write)}


def check_cache_disk(plans: list, out: Path) -> dict:
    from causaldemand.download import DiskSpaceError, _same_disk

    need = cache_bytes(plans)
    if not plans or not need:
        return {"cache_needed": 0, "same_disk": True}
    cache = plans[0].cache
    same = _same_disk(out, cache)
    if not same:
        free = free_bytes(cache)
        if need * HEADROOM > free:
            raise DiskSpaceError(f"the Kilts Center files and the builds need about {human(need)} in the cache "
                                 f"({show(cache)}), which has {human(free)} free. Free space there.")
    return {"cache_needed": need, "same_disk": same}
