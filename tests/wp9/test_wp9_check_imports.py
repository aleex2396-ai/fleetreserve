"""`tools/wp9_check.py` exits 2 (could not run), never Python's own 1, when its imports fail."""

import os
import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parents[2] / "tools" / "wp9_check.py"


def test_the_tool_exits_2_when_its_imports_fail(tmp_path):
    shadow = tmp_path / "shadow" / "fleetreserve"
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text("")  # a package without the modules the tool imports
    environment = {**os.environ, "PYTHONPATH": str(shadow.parent)}
    done = subprocess.run(
        [sys.executable, str(TOOL), "abstain", "--top", str(tmp_path)],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert done.returncode == 2
    assert done.stdout.startswith("COULD NOT RUN: the check cannot start: ModuleNotFoundError")
