"""The check runner library gives three outcomes: passed (0), failed (1), could not run (2)."""

import subprocess
from pathlib import Path

LIB = Path(__file__).resolve().parents[1] / "tools" / "check_lib.sh"


def run_checks(body: str) -> tuple[int, list[str]]:
    """Source the library in a fresh bash, run `body`, return exit status and printed lines."""
    script = f'source "{LIB}" || exit 99\n{body}\ncheck_finish\n'
    done = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=60, check=False
    )
    return done.returncode, done.stdout.splitlines()


def test_all_pass():
    status, lines = run_checks('check "one" true\ncheck "two" true')
    assert status == 0
    assert lines == ["passed: one", "passed: two", "all 2 checks passed"]


def test_one_fails():
    status, lines = run_checks('check "one" true\ncheck "two" false')
    assert status == 1
    assert lines == ["passed: one", "FAILED: two"]


def test_one_cannot_run_exit_2():
    status, lines = run_checks('check "one" true\ncheck "two" bash -c "exit 2"')
    assert status == 2
    assert lines == ["passed: one", "COULD NOT RUN: two (exit 2)"]


def test_missing_command_cannot_run_exit_127():
    status, lines = run_checks('check "one" no_such_command_for_this_test')
    assert status == 2
    assert lines == ["COULD NOT RUN: one (exit 127)"]


def test_fail_and_cannot_run_together_is_failed():
    status, lines = run_checks(
        'check "one" bash -c "exit 2"\ncheck "two" false\ncheck "three" true'
    )
    assert status == 1
    assert lines == [
        "COULD NOT RUN: one (exit 2)",
        "FAILED: two",
        "passed: three",
    ]


def test_no_checks_listed_cannot_run():
    status, lines = run_checks("")
    assert status == 2
    assert lines == ["COULD NOT RUN: no checks are listed in check.sh yet"]


def test_checked_command_does_not_see_callers_variables():
    body = 'export CALLER_SET_THIS=1\ncheck "one" bash -c \'[ -z "${CALLER_SET_THIS-}" ]\''
    status, lines = run_checks(body)
    assert status == 0
    assert lines == ["passed: one", "all 1 checks passed"]


def test_checked_command_still_has_path_and_home():
    body = 'check "one" bash -c \'[ -n "${HOME-}" ] && [ -x "$(command -v ls)" ]\''
    status, lines = run_checks(body)
    assert status == 0


def test_exported_function_named_env_does_not_hide_a_failure():
    body = 'env() { return 0; }\nexport -f env\ncheck "one" bash -c "exit 1"'
    status, lines = run_checks(body)
    assert status == 1
    assert lines == ["FAILED: one"]


# --- the learn lane in check.sh ------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
CHECK_SH = ROOT / "check.sh"
CHECK_SLOW_SH = ROOT / "tools" / "check-slow.sh"


def check_lines(script: Path = CHECK_SH) -> list[str]:
    return [line for line in script.read_text().splitlines() if line.startswith("check ")]


def test_check_sh_runs_the_learn_tests_in_an_environment_with_the_learn_group():
    (line,) = [line for line in check_lines() if "--lane learn" in line]
    assert "uv run --frozen --group learn python tools/run_tests.py --lane learn" in line


def test_check_sh_default_tests_run_without_the_learn_group():
    (line,) = [line for line in check_lines() if "run_tests.py" in line and "learn" not in line]
    assert "--group" not in line and "--lane" not in line


def test_check_sh_runs_the_fast_set_of_each_lane_and_never_the_slow_set():
    lines = [line for line in check_lines() if "run_tests.py" in line]
    assert len(lines) == 2
    assert all(line.endswith("--set fast") for line in lines)
    assert not [
        line for line in check_lines(CHECK_SH) if "--set slow" in line or "--set all" in line
    ]


def test_check_slow_sh_runs_the_slow_set_of_every_lane_that_has_slow_files():
    lines = [line for line in check_lines(CHECK_SLOW_SH) if "run_tests.py" in line]
    assert all(line.endswith("--set slow") for line in lines)
    listed = [
        line.split("#")[0].strip()
        for line in (ROOT / "tests" / "slow.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    learn = [line for line in lines if "--lane learn" in line]
    default = [line for line in lines if "--lane" not in line]
    assert len(default) == int(any(not name.startswith("tests/learn/") for name in listed))
    assert len(learn) == int(any(name.startswith("tests/learn/") for name in listed))
    assert all("--group learn" in line for line in learn)
    assert len(default) + len(learn) == len(lines) > 0


def test_check_slow_sh_says_when_it_must_run_and_uses_the_same_library():
    text = CHECK_SLOW_SH.read_text()
    header = text.split("set -u")[0]
    assert "once before the tag" in header
    assert "held-out data, the seal or the sealed-run path" in header
    assert "Exit 0:" in header and "Exit 1:" in header and "Exit 2:" in header
    assert "source tools/check_lib.sh" in text and text.rstrip().endswith("check_finish")


def test_a_learn_environment_that_cannot_be_set_up_is_could_not_run_not_failed():
    command = "uv run --frozen --offline --group no_such_group_for_this_test true"
    status, lines = run_checks(f'check "learn lane" {command}')
    assert status == 2
    assert lines == ["COULD NOT RUN: learn lane (exit 2)"]
