"""A skipped, expected-to-fail or deselected test never counts as a pass."""

import os
import subprocess
import sys
from pathlib import Path

CONFTEST = Path(__file__).resolve().parent / "conftest.py"

PASSING = "def test_ok():\n    assert True\n"
FAILING = "def test_bad():\n    assert False\n"
_MARK = "import pytest\n\n\n@pytest.mark.{mark}\ndef test_x():\n    {body}\n"
SKIPPED = _MARK.format(mark="skip", body="pass")
XFAILED = _MARK.format(mark="xfail", body="assert False")
XPASSED = _MARK.format(mark="xfail", body="assert True")


def run_pytest_full(folder: Path, files: dict[str, str], *args: str):
    """Run pytest in `folder` on made-up test files plus a copy of the conftest rule."""
    (folder / "conftest.py").write_text(CONFTEST.read_text())
    (folder / "pytest.ini").write_text("[pytest]\n")
    for name, text in files.items():
        (folder / name).write_text(text)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args],
        cwd=folder,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def run_pytest(folder: Path, files: dict[str, str], *args: str) -> int:
    return run_pytest_full(folder, files, *args).returncode


def run_pytest_output(folder: Path, files: dict[str, str], *args: str) -> str:
    return run_pytest_full(folder, files, *args).stdout


def test_all_pass_is_0(tmp_path):
    assert run_pytest(tmp_path, {"test_a.py": PASSING}) == 0


def test_one_skipped_is_2(tmp_path):
    assert run_pytest(tmp_path, {"test_a.py": PASSING, "test_b.py": SKIPPED}) == 2


def test_one_xfail_is_2(tmp_path):
    assert run_pytest(tmp_path, {"test_a.py": PASSING, "test_b.py": XFAILED}) == 2


def test_one_xpass_is_2(tmp_path):
    assert run_pytest(tmp_path, {"test_a.py": PASSING, "test_b.py": XPASSED}) == 2


def test_failing_plus_skipped_is_1(tmp_path):
    assert run_pytest(tmp_path, {"test_a.py": FAILING, "test_b.py": SKIPPED}) == 1


def test_deselected_is_2(tmp_path):
    files = {"test_a.py": PASSING, "test_b.py": PASSING.replace("test_ok", "test_other")}
    assert run_pytest(tmp_path, files, "-k", "test_ok") == 2


def test_no_tests_is_not_0_and_not_1(tmp_path):
    status = run_pytest(tmp_path, {})
    assert status not in (0, 1)


def test_module_level_skip_is_2(tmp_path):
    skip = "import pytest\n\npytest.skip('made up', allow_module_level=True)\n"
    assert run_pytest(tmp_path, {"test_a.py": PASSING, "test_b.py": skip}) == 2


def test_importorskip_is_2(tmp_path):
    skip = "import pytest\n\npytest.importorskip('no_such_module_for_this_test')\n"
    assert run_pytest(tmp_path, {"test_a.py": PASSING, "test_b.py": skip}) == 2


def test_collect_only_is_2(tmp_path):
    assert run_pytest(tmp_path, {"test_a.py": PASSING}, "--collect-only") == 2


def test_message_is_on_its_own_line_and_names_a_skip_once(tmp_path):
    out = run_pytest_output(tmp_path, {"test_a.py": PASSING, "test_b.py": SKIPPED}, "-q")
    lines = [line for line in out.splitlines() if line.startswith("could not run all tests")]
    assert lines == ["could not run all tests (1 skipped): exit status 2"]


def test_early_exit_with_status_0_is_2(tmp_path):
    first = "import pytest\n\n\ndef test_first():\n    pytest.exit('stop', returncode=0)\n"
    assert run_pytest(tmp_path, {"test_a.py": first, "test_b.py": PASSING}) == 2


def test_early_exit_in_the_last_test_is_2(tmp_path):
    last = "import pytest\n\n\ndef test_last():\n    pytest.exit('stop', returncode=0)\n"
    assert run_pytest(tmp_path, {"test_a.py": PASSING, "test_b.py": last}) == 2


# A test file that reads a file kept only on this machine: left out of a whole-suite run when that
# file is missing (so a fresh checkout still passes), never left out when named on the command line.
LOCAL_TEST = "tests/wp1/test_earlier_offer_matrices.py"
LOCAL_FILE = "reference/part1/src/fleetoffer/offer.py"
EXITS_2 = "import pytest\n\npytest.exit('missing', returncode=2)\n\n\ndef test_x():\n    pass\n"


def run_local_layout(folder: Path, with_file: bool, *args: str):
    (folder / "tests" / "wp1").mkdir(parents=True)
    (folder / "tests" / "conftest.py").write_text(CONFTEST.read_text())
    (folder / "pytest.ini").write_text("[pytest]\n")
    (folder / "tests" / "test_other.py").write_text(PASSING)
    local_test = PASSING.replace("test_ok", "test_local") if with_file else EXITS_2
    (folder / LOCAL_TEST).write_text(local_test)
    if with_file:
        (folder / LOCAL_FILE).parent.mkdir(parents=True)
        (folder / LOCAL_FILE).write_text("# stands in for the local file\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args],
        cwd=folder,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_local_file_missing_whole_suite_leaves_the_test_out_and_is_0(tmp_path):
    done = run_local_layout(tmp_path, False)
    assert done.returncode == 0
    assert "1 passed" in done.stdout


def test_local_file_missing_named_test_is_2(tmp_path):
    assert run_local_layout(tmp_path, False, LOCAL_TEST).returncode == 2


def test_local_file_present_whole_suite_runs_the_test(tmp_path):
    done = run_local_layout(tmp_path, True)
    assert done.returncode == 0
    assert "2 passed" in done.stdout
