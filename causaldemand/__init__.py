import importlib.abc
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

__version__ = "2.0.0"

_SUBPACKAGES = {"metrics": "panel_scorer.py"}
_HERE = Path(__file__).resolve().parent
_REPOSITORY_FOLDERS = {f"{__name__}.{name}": _HERE.parent / name for name, marker in _SUBPACKAGES.items()
                       if not (_HERE / name).is_dir() and (_HERE.parent / name / marker).is_file()}


class _RepositoryFolders(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        folder = _REPOSITORY_FOLDERS.get(fullname)
        if folder is None:
            return None
        init = folder / "__init__.py"
        if init.is_file():
            return importlib.util.spec_from_file_location(fullname, init, submodule_search_locations=[str(folder)])
        spec = importlib.machinery.ModuleSpec(fullname, None, is_package=True)
        spec.submodule_search_locations = [str(folder)]
        return spec


if _REPOSITORY_FOLDERS and not any(type(f).__name__ == "_RepositoryFolders" for f in sys.meta_path):
    sys.meta_path.append(_RepositoryFolders())
