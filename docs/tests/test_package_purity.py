"""Guard: `causaldemand` is pure — importable without the pipeline.

The package is the single source of truth for the metric math and is distributed standalone (pip). It must never grow an import of `benchmark_pipeline`, which would drag the hidden DGP into the public distribution.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_package_imports_without_benchmark_pipeline() -> None:
    """Importing the package must not load any benchmark_pipeline module."""
    code = (
        "import sys\n"
        "import causaldemand\n"
        "loaded = [m for m in sys.modules if m.startswith('benchmark_pipeline')]\n"
        "assert not loaded, f'package pulled in pipeline modules: {loaded}'\n"
        "assert causaldemand.__version__\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_package_sources_do_not_reference_pipeline() -> None:
    """No source file in the package may import benchmark_pipeline (even lazily)."""
    package_dir = REPO_ROOT / "causaldemand"
    offenders = [
        path.name
        for path in package_dir.glob("*.py")
        if "import benchmark_pipeline" in path.read_text(encoding="utf-8")
        or "from benchmark_pipeline" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"package modules import the pipeline: {offenders}"
