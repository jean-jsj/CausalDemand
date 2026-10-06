from __future__ import annotations

import hashlib
import importlib
import importlib.util
from pathlib import Path
from types import ModuleType

_FOLDER = {"metrics": "metrics"}
_cache: dict[str, ModuleType] = {}


def _load(subpackage: str, module: str) -> ModuleType:
    key = f"{subpackage}.{module}"
    if key in _cache:
        return _cache[key]
    try:
        mod = importlib.import_module(f"causaldemand.{subpackage}.{module}")
    except ModuleNotFoundError:
        path = Path(__file__).resolve().parents[1] / _FOLDER[subpackage] / f"{module}.py"
        if not path.is_file():
            raise
        spec = importlib.util.spec_from_file_location(f"causaldemand_repo_{subpackage}_{module}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    _cache[key] = mod
    return mod


def panel_scorer() -> ModuleType:
    return _load("metrics", "panel_scorer")


def elasticity() -> ModuleType:
    return _load("metrics", "elasticity")


def csv_text() -> ModuleType:
    return importlib.import_module("causaldemand.csv_text")


def output_folder(folder: Path) -> Path:
    folder = Path(folder)
    if not folder.exists():
        folder.mkdir(parents=True)
        (folder / ".gitignore").write_text("# Output of causaldemand: git ignores this folder.\n*\n",
                                           encoding="utf-8")
    return folder


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def module_sha256(mod: ModuleType) -> str:
    return file_sha256(Path(mod.__file__))
