from __future__ import annotations

import fnmatch
from pathlib import Path

DEFAULT_REPO_ID = "jean-jsj/CausalDemand"


class SourceError(RuntimeError):
    pass


class LocalSource:
    kind = "local folder"

    def __init__(self, root: str | Path, cache_dir: str | None = None):
        self.root = Path(root).expanduser()
        self.cache_dir = cache_dir
        if not self.root.is_dir():
            raise SourceError(f"{root}: no such folder")

    def fetch(self, paths: list[str]) -> Path:
        for p in paths:
            if any(ch in p for ch in "*?["):
                continue
            if not (self.root / p).exists():
                raise SourceError(f"{p} is not in the release folder")
        return self.root

    def glob(self, pattern: str) -> list[str]:
        return sorted(p.relative_to(self.root).as_posix() for p in self.root.glob(pattern) if p.is_file())

    def record(self) -> dict:
        return {"kind": self.kind}

    def describe(self) -> str:
        return f"the local release folder {self.root}"

    def missing(self, paths: list[str]) -> list[str]:
        return []

    def cache_folder(self) -> Path | None:
        return None


def gate_text(repo_id: str, exc: Exception) -> str | None:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if type(exc).__name__ != "GatedRepoError" and status not in (401, 403):
        return None
    return (f"the Hugging Face dataset {repo_id} is gated: accept its access conditions with a free Hugging Face "
            f"account at https://huggingface.co/datasets/{repo_id}, then set HF_TOKEN to an access token of that "
            "account")


def hub_error(repo_id: str, what: str, exc: Exception) -> SourceError:
    return SourceError(gate_text(repo_id, exc) or f"{what}: {type(exc).__name__}: {exc}")


class HubSource:
    kind = "huggingface"

    def __init__(self, repo_id: str = DEFAULT_REPO_ID, revision: str | None = None, cache_dir: str | None = None):
        self.repo_id = repo_id
        self.revision = revision
        self.cache_dir = cache_dir
        self.commit: str | None = None
        self._files: list[str] | None = None

    def _snapshot(self, patterns: list[str]) -> Path:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise SourceError("downloading from Hugging Face needs the package huggingface_hub") from exc
        try:
            folder = Path(snapshot_download(repo_id=self.repo_id, repo_type="dataset",
                                            revision=self.commit or self.revision, allow_patterns=patterns,
                                            cache_dir=self.cache_dir))
        except Exception as exc:
            raise hub_error(self.repo_id, f"download from the Hugging Face repository {self.repo_id} failed",
                            exc) from exc
        if self.commit is not None and folder.name != self.commit:
            raise SourceError(f"the download from {self.repo_id} returned commit {folder.name}, not {self.commit}")
        self.commit = folder.name
        return folder

    def _resolve(self) -> str:
        if self.commit is None:
            from huggingface_hub import HfApi
            try:
                self.commit = HfApi().repo_info(self.repo_id, repo_type="dataset", revision=self.revision).sha
            except Exception as exc:
                raise hub_error(self.repo_id, f"the Hugging Face repository {self.repo_id} could not be reached",
                                exc) from exc
        return self.commit

    def fetch(self, paths: list[str]) -> Path:
        folder = self._snapshot(paths)
        for p in paths:
            if any(ch in p for ch in "*?["):
                continue
            if not (folder / p).exists():
                raise SourceError(f"{p} is not in {self.repo_id} at revision {self.revision or 'main'} "
                                  f"(commit {self.commit})")
        return folder

    def files(self) -> list[str]:
        if self._files is None:
            commit = self._resolve()
            from huggingface_hub import HfApi
            try:
                self._files = sorted(HfApi().list_repo_files(self.repo_id, repo_type="dataset", revision=commit))
            except Exception as exc:
                raise hub_error(self.repo_id, f"listing the Hugging Face repository {self.repo_id} failed",
                                exc) from exc
        return self._files

    def glob(self, pattern: str) -> list[str]:
        return [f for f in self.files() if fnmatch.fnmatchcase(f, pattern)]

    def record(self) -> dict:
        return {"kind": self.kind, "repo_id": self.repo_id, "revision": self.revision or "main", "commit": self.commit}

    def describe(self) -> str:
        commit = f", commit {self.commit[:7]}" if self.commit else ""
        return f"Hugging Face {self.repo_id} (revision {self.revision or 'main'}{commit})"

    def cache_folder(self) -> Path:
        if self.cache_dir:
            return Path(self.cache_dir).expanduser()
        try:
            from huggingface_hub import constants
            return Path(constants.HF_HUB_CACHE)
        except (ImportError, AttributeError):
            return Path("~/.cache/huggingface/hub").expanduser()

    def missing(self, paths: list[str]) -> list[str]:
        if self.commit is None:
            return list(paths)
        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError:
            return list(paths)
        out = []
        for p in paths:
            try:
                found = try_to_load_from_cache(self.repo_id, p, cache_dir=self.cache_dir, revision=self.commit,
                                               repo_type="dataset")
            except Exception:
                found = None
            if not isinstance(found, str):
                out.append(p)
        return out


def make_source(source: str | None, repo_id: str = DEFAULT_REPO_ID, revision: str | None = None,
                cache_dir: str | None = None):
    if source:
        return LocalSource(source, cache_dir)
    return HubSource(repo_id, revision, cache_dir)


def read_sha256sums(folder: Path) -> dict[str, str] | None:
    p = folder / "SHA256SUMS"
    if not p.is_file():
        return None
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, path = line.split(None, 1)
            out[path.strip()] = digest
    return out
