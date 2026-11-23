"""The parallel test runner gives the result of one plain whole-suite pytest call.

Each case builds a small made-up suite in the test's temporary folder (with a copy of the real
conftest rule, so a skipped test is not a pass there either) and compares the runner's exit
status with the status of one plain pytest call over the same folder. One case collects the
real suite and checks the split without running it.
"""

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "run_tests.py"
CONFTEST = Path(__file__).resolve().parent / "conftest.py"
PASSING = "def test_ok():\n    assert True\n"
FAILING = "def test_bad():\n    assert False\n"
SKIPPED = "import pytest\n\n\n@pytest.mark.skip\ndef test_s():\n    pass\n"
XFAILED = "import pytest\n\n\n@pytest.mark.xfail\ndef test_x():\n    assert False\n"
XPASSED = "import pytest\n\n\n@pytest.mark.xfail\ndef test_x():\n    assert True\n"
WARNS = "import warnings\n\n\ndef test_w():\n    warnings.warn('made up')\n"
BROKEN_IMPORT = "import no_such_module_for_this_test\n\n\ndef test_x():\n    pass\n"
# These two act only in a run of the tests, not in the listing, so they reach a group.
NOT_WHILE_LISTING = "import sys\n\nif '--collect-only' not in sys.argv:\n    {action}\n\n\n"
EXIT_2 = NOT_WHILE_LISTING.format(action="import pytest; pytest.exit('stop', returncode=2)")
GROUP_IMPORT_ERROR = NOT_WHILE_LISTING.format(action="raise ImportError('made up')")
SIGNAL = "import os\nimport signal\n\n\ndef test_k():\n    os.kill(os.getpid(), signal.SIGKILL)\n"
NO_SUMMARY = "import os\n\n\ndef test_e():\n    os._exit(0)\n"
FINAL = re.compile(
    r"^run_tests: exit (?P<status>\d): (?P<passed>\d+) passed, (?P<failed>\d+) failed, "
    r"(?P<errors>\d+) errors, (?P<skipped>\d+) skipped, (?P<groups>\d+) groups"
)


def load_tool():
    spec = importlib.util.spec_from_file_location("run_tests_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = load_tool()


def environment() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}


def make_tree(folder: Path, files: dict[str, str]) -> Path:
    """A made-up project: its own pytest settings, the real conftest rule, the given tests."""
    (folder / "tests").mkdir(parents=True, exist_ok=True)
    (folder / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths = ["tests"]\naddopts = "-q"\n'
        'filterwarnings = ["error"]\n'
    )
    (folder / "tests" / "conftest.py").write_text(CONFTEST.read_text())
    for name, text in files.items():
        path = folder / "tests" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return folder


def run_tool(tree: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), "--root", str(tree), "--jobs", "3", *args],
        capture_output=True,
        text=True,
        env=environment(),
        timeout=300,
        check=False,
    )


def plain_status(tree: Path) -> int:
    """The exit status of one plain pytest call over the whole made-up suite."""
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-c", "pyproject.toml", "-p", "no:cacheprovider"],
        cwd=tree,
        capture_output=True,
        text=True,
        env=environment(),
        timeout=300,
        check=False,
    )
    return done.returncode


def final_line(done: subprocess.CompletedProcess[str]) -> str:
    lines = [line for line in done.stdout.splitlines() if line.startswith("run_tests:")]
    assert len(lines) == 1, done.stdout
    assert done.stdout.rstrip("\n").splitlines()[-1] == lines[0]
    return lines[0]


def three_files(**extra: str) -> dict[str, str]:
    return {"test_a.py": PASSING, "test_b.py": PASSING, "test_c.py": PASSING, **extra}


# --- results -----------------------------------------------------------------------------------


def test_all_pass_is_0_and_the_count_is_the_listing(tmp_path):
    files = {
        "test_a.py": "def test_1():\n    pass\n\n\ndef test_2():\n    pass\n",
        "test_b.py": "import pytest\n\n\n@pytest.mark.parametrize('n', [1, 2, 3])\n"
        "def test_3(n):\n    pass\n",
        "sub/test_c.py": PASSING,
    }
    tree = make_tree(tmp_path, files)
    done = run_tool(tree)
    found = FINAL.match(final_line(done))
    assert done.returncode == 0
    assert found["passed"] == "6" and found["failed"] == "0" and found["groups"] == "3"
    assert plain_status(tree) == 0


def test_one_failing_test_is_1_and_names_its_file(tmp_path):
    tree = make_tree(tmp_path, three_files(**{"test_d.py": FAILING}))
    done = run_tool(tree)
    line = final_line(done)
    assert done.returncode == 1
    assert "tests/test_d.py" in line and "group" in line
    assert FINAL.match(line)["failed"] == "1"
    assert plain_status(tree) == 1


def test_failure_and_another_group_that_cannot_run_is_1(tmp_path):
    tree = make_tree(
        tmp_path, {"test_a.py": FAILING, "test_b.py": EXIT_2 + PASSING, "test_c.py": PASSING}
    )
    assert run_tool(tree).returncode == 1


def test_collection_error_is_what_the_plain_run_gives(tmp_path):
    tree = make_tree(tmp_path, three_files(**{"test_d.py": BROKEN_IMPORT}))
    done = run_tool(tree)
    assert plain_status(tree) == 2
    assert done.returncode == 2
    assert "could not run" in final_line(done)


def test_collection_error_met_only_inside_a_group_is_2(tmp_path):
    tree = make_tree(tmp_path, three_files(**{"test_d.py": GROUP_IMPORT_ERROR + PASSING}))
    done = run_tool(tree)
    assert done.returncode == 2
    assert "tests/test_d.py" in final_line(done)


def test_pytest_exit_2_at_import_is_2(tmp_path):
    tree = make_tree(tmp_path, three_files(**{"test_d.py": EXIT_2 + PASSING}))
    done = run_tool(tree)
    assert done.returncode == 2
    assert "tests/test_d.py" in final_line(done)


def test_pytest_exit_2_at_import_in_the_listing_is_what_the_plain_run_gives(tmp_path):
    exits = "import pytest\n\npytest.exit('stop', returncode=2)\n\n\ndef test_x():\n    pass\n"
    tree = make_tree(tmp_path, three_files(**{"test_d.py": exits}))
    assert plain_status(tree) == 2
    assert run_tool(tree).returncode == 2


def test_empty_tree_is_2(tmp_path):
    done = run_tool(make_tree(tmp_path, {}))
    assert plain_status(tmp_path) == 5
    assert done.returncode == 2
    assert "collection ended with exit status 5" in final_line(done)


def test_a_group_killed_by_a_signal_is_2(tmp_path):
    tree = make_tree(tmp_path, three_files(**{"test_d.py": SIGNAL}))
    done = run_tool(tree)
    assert done.returncode == 2
    assert "killed by signal 9" in final_line(done)


def test_a_group_that_ends_with_status_0_and_no_summary_is_2(tmp_path):
    tree = make_tree(tmp_path, three_files(**{"test_d.py": NO_SUMMARY}))
    done = run_tool(tree)
    assert done.returncode == 2
    assert "summary line cannot be read" in final_line(done)


@pytest.mark.parametrize(
    ("name", "text", "expected"),
    [
        ("skipped", SKIPPED, 2),
        ("xfailed", XFAILED, 2),
        ("xpassed", XPASSED, 2),
        ("warning", WARNS, 1),
        ("skip_module", "import pytest\n\npytest.skip('x', allow_module_level=True)\n", 2),
    ],
)
def test_not_a_pass_cases_give_the_plain_runs_status(tmp_path, name, text, expected):
    tree = make_tree(tmp_path, three_files(**{"test_d.py": text}))
    assert plain_status(tree) == expected
    assert run_tool(tree).returncode == expected


def test_failure_beside_a_skip_is_1_like_the_plain_run(tmp_path):
    tree = make_tree(tmp_path, {"test_a.py": FAILING, "test_b.py": SKIPPED, "test_c.py": PASSING})
    assert plain_status(tree) == 1
    assert run_tool(tree).returncode == 1


def test_sequential_flag_gives_the_plain_status(tmp_path):
    ok = make_tree(tmp_path / "ok", three_files())
    bad = make_tree(tmp_path / "bad", three_files(**{"test_d.py": FAILING}))
    skip = make_tree(tmp_path / "skip", three_files(**{"test_d.py": SKIPPED}))
    empty = make_tree(tmp_path / "empty", {})
    assert [run_tool(t, "--sequential").returncode for t in (ok, bad, skip, empty)] == [0, 1, 2, 2]


def test_jobs_below_one_is_2(tmp_path):
    tree = make_tree(tmp_path, three_files())
    assert run_tool(tree, "--jobs", "0").returncode == 2


# --- the learn lane -----------------------------------------------------------------------------

NEEDS_LEARN = "import no_such_module_for_this_test\n\n\ndef test_l():\n    pass\n"


def lane_tree(folder: Path, learn: dict[str, str], **default: str) -> Path:
    """A made-up suite with tests in `tests/learn/` beside the ordinary ones."""
    files = {f"learn/{name}": text for name, text in learn.items()}
    return make_tree(folder, {**three_files(), **default, **files})


def test_the_default_lane_leaves_tests_learn_out_even_when_it_cannot_import(tmp_path):
    tree = lane_tree(tmp_path, {"test_l.py": NEEDS_LEARN})
    done = run_tool(tree)
    assert done.returncode == 0, done.stdout
    assert FINAL.match(final_line(done))["passed"] == "3"
    assert "learn" not in done.stdout
    assert plain_status(tree) == 0


def test_the_learn_lane_runs_only_tests_learn(tmp_path):
    tree = lane_tree(tmp_path, {"test_l.py": PASSING, "test_m.py": PASSING})
    done = run_tool(tree, "--lane", "learn")
    assert done.returncode == 0, done.stdout
    assert FINAL.match(final_line(done))["passed"] == "2"


def test_a_failing_learn_test_is_1_and_names_its_file(tmp_path):
    tree = lane_tree(tmp_path, {"test_l.py": FAILING})
    done = run_tool(tree, "--lane", "learn")
    assert done.returncode == 1
    assert "tests/learn/test_l.py" in final_line(done)
    assert run_tool(tree).returncode == 0  # the default lane never sees it


def test_a_learn_test_that_cannot_import_is_not_a_pass(tmp_path):
    tree = lane_tree(tmp_path, {"test_l.py": NEEDS_LEARN})
    assert run_tool(tree, "--lane", "learn").returncode in (1, 2)


def test_a_skipped_learn_test_is_2(tmp_path):
    tree = lane_tree(tmp_path, {"test_l.py": SKIPPED})
    assert run_tool(tree, "--lane", "learn").returncode == 2


def test_the_learn_lane_with_no_learn_tests_is_2(tmp_path):
    missing = make_tree(tmp_path / "missing", three_files())
    assert run_tool(missing, "--lane", "learn").returncode == 2
    empty = lane_tree(tmp_path / "empty", {"helper.py": "X = 1\n"})
    done = run_tool(empty, "--lane", "learn")
    assert done.returncode == 2
    assert "could not run: collection ended with exit status 5" in done.stdout


def test_an_unknown_lane_is_2(tmp_path):
    tree = make_tree(tmp_path, three_files())
    assert run_tool(tree, "--lane", "no_such_lane").returncode == 2


# --- the split is checked ----------------------------------------------------------------------


def test_a_file_dropped_from_every_group_fails_the_tool(tmp_path, capsys):
    tree = make_tree(tmp_path, three_files())

    def drops_one(files, jobs):
        groups = tool.split_into_groups(files, jobs)
        groups[0].remove("tests/test_a.py")
        return [g for g in groups if g]

    status = tool.main(["--root", str(tree), "--jobs", "3"], grouping=drops_one)
    out = capsys.readouterr().out
    assert status == 2
    assert "collected but in no group: tests/test_a.py" in out


def test_a_file_in_two_groups_is_refused(tmp_path, capsys):
    tree = make_tree(tmp_path, three_files())

    def repeats_one(files, jobs):
        groups = tool.split_into_groups(files, jobs)
        groups[-1].append(groups[0][0])
        return groups

    status = tool.main(["--root", str(tree), "--jobs", "3"], grouping=repeats_one)
    out = capsys.readouterr().out
    assert status == 2
    assert "is in group 1 and group" in out


def test_a_group_with_a_file_that_was_not_collected_is_refused():
    with pytest.raises(tool.CouldNotRun, match="not collected"):
        tool.check_partition({"tests/a.py": 1}, [["tests/a.py", "tests/b.py"]])


def test_a_file_missing_from_the_weights_table_is_still_run(tmp_path):
    tree = make_tree(tmp_path, three_files())
    table = tmp_path / "weights.txt"
    table.write_text("5 tests/test_a.py\n")
    collected, groups = tool.plan(tree, "pyproject.toml", 3, table)
    assert sorted(f for g in groups for f in g) == sorted(collected)
    done = run_tool(tree, "--weights", str(table))
    assert done.returncode == 0
    assert FINAL.match(final_line(done))["passed"] == "3"


def test_a_missing_or_malformed_weights_table_is_2(tmp_path):
    tree = make_tree(tmp_path, three_files())
    assert run_tool(tree, "--weights", str(tmp_path / "none.txt")).returncode == 2
    bad = tmp_path / "bad.txt"
    bad.write_text("heavy tests/test_a.py\n")
    assert run_tool(tree, "--weights", str(bad)).returncode == 2
    bad.write_text("0 tests/test_a.py\n")
    assert run_tool(tree, "--weights", str(bad)).returncode == 2


def test_weight_lookup_prefers_the_file_then_the_longest_folder_then_the_default():
    table = {"tests/": 2.0, "tests/wp1/": 3.0, "tests/wp1/test_x.py": 9.0}
    assert tool.weight_of("tests/wp1/test_x.py", table) == 9.0
    assert tool.weight_of("tests/wp1/test_y.py", table) == 3.0
    assert tool.weight_of("tests/test_z.py", table) == 2.0
    assert tool.weight_of("other/test_z.py", table) == tool.DEFAULT_WEIGHT


def test_split_puts_the_heaviest_file_alone_and_is_repeatable():
    files = {"a.py": 10.0, "b.py": 4.0, "c.py": 3.0, "d.py": 3.0}
    assert tool.split_into_groups(files, 2) == [["a.py"], ["b.py", "c.py", "d.py"]]
    assert tool.split_into_groups(files, 2) == tool.split_into_groups(dict(files), 2)
    assert tool.split_into_groups({"a.py": 1.0}, 6) == [["a.py"]]


# --- reading a summary line --------------------------------------------------------------------


def judged(code, text, expected, files=("tests/a.py",)):
    result = tool.GroupResult(1, list(files), expected, returncode=code, output=text)
    tool.judge(result)
    return result


def test_summary_lines_are_read():
    assert tool.parse_summary("....\n12 passed in 1.50s\n") == {"passed": 12}
    assert tool.parse_summary("== 1 failed, 5 passed, 2 warnings in 0.3s ==") == {
        "failed": 1,
        "passed": 5,
        "warnings": 2,
    }
    assert tool.parse_summary("1 error in 0.17s") == {"errors": 1}
    assert tool.parse_summary("no tests ran in 0.01s") == {}
    assert tool.parse_summary("3 passed in 12.00s (0:00:12)") == {"passed": 3}
    assert tool.parse_summary("nothing here") is None
    assert tool.parse_summary("3 passed and more in 1.0s") is None


def test_a_passed_count_that_differs_from_the_listing_is_2():
    assert judged(0, "3 passed in 0.1s", 3).status == tool.PASSED
    short = judged(0, "3 passed in 0.1s", 4)
    assert short.status == tool.COULD_NOT_RUN and "3 passed, 4 collected" in short.reason
    assert judged(0, "5 passed in 0.1s", 4).status == tool.COULD_NOT_RUN


def test_status_0_with_a_skip_in_the_summary_is_2():
    assert judged(0, "3 passed, 1 skipped in 0.1s", 3).status == tool.COULD_NOT_RUN


@pytest.mark.parametrize("code", [2, 3, 4, 5, -9, None])
def test_other_exit_statuses_are_2(code):
    assert judged(code, "3 passed in 0.1s", 3).status == tool.COULD_NOT_RUN


def test_status_1_is_a_failure_and_names_the_files_in_the_group_only():
    text = "FAILED tests/a.py::test_x - assert\nFAILED tests/other.py::test_y\n1 failed in 0.1s"
    result = judged(1, text, 3, files=("tests/a.py", "tests/b.py"))
    assert result.status == tool.FAILED and result.problem_files == ["tests/a.py"]


def test_the_overall_result_follows_the_group_results():
    ok = judged(0, "3 passed in 0.1s", 3)
    failed = judged(1, "1 failed in 0.1s", 3)
    stuck = judged(2, "", 3)
    assert tool.decide([ok], 3)[0] == 0
    assert tool.decide([ok], 4)[0] == 2
    assert tool.decide([ok], 0)[0] == 2
    assert tool.decide([ok, stuck], 6)[0] == 2
    assert tool.decide([stuck, failed], 6)[0] == 1


# --- totals ------------------------------------------------------------------------------------


def test_the_final_line_totals_add_up_to_the_group_lines(tmp_path):
    files = {"test_a.py": PASSING, "test_b.py": PASSING + FAILING, "test_c.py": PASSING}
    done = run_tool(make_tree(tmp_path, files))
    found = FINAL.match(final_line(done))
    rows = re.findall(r"^group \d+: \d+ files, (\d+) tests, (\d+) passed", done.stdout, re.M)
    assert done.returncode == 1
    assert int(found["groups"]) == len(rows) == 3
    assert sum(int(passed) for _, passed in rows) == int(found["passed"]) == 3
    assert sum(int(listed) for listed, _ in rows) == 4
    assert int(found["failed"]) == 1


def test_group_output_is_printed_in_group_order(tmp_path):
    done = run_tool(make_tree(tmp_path, three_files()))
    numbers = [int(n) for n in re.findall(r"^===== group (\d+):", done.stdout, re.M)]
    assert numbers == [1, 2, 3]


# --- the fast and the slow set -------------------------------------------------------------------


def slow_tree(folder: Path, listed: list[str], files: dict[str, str] | None = None) -> Path:
    """A made-up suite whose `tests/slow.txt` names `listed` (paths from the root), with reasons."""
    tree = make_tree(folder, three_files() if files is None else files)
    write_list(tree, "".join(f"{name}  # made up: takes long\n" for name in listed))
    return tree


def write_list(tree: Path, text: str) -> None:
    (tree / "tests" / "slow.txt").write_text(text)


def passed_of(done: subprocess.CompletedProcess[str]) -> str:
    return FINAL.match(final_line(done))["passed"]


def test_the_fast_set_runs_the_unlisted_files_and_the_slow_set_the_listed_ones(tmp_path):
    tree = slow_tree(tmp_path, ["tests/test_c.py"], three_files(**{"test_c.py": FAILING}))
    fast = run_tool(tree, "--set", "fast")
    assert fast.returncode == 0, fast.stdout
    assert passed_of(fast) == "2" and final_line(fast).endswith(", set fast")
    slow = run_tool(tree, "--set", "slow")
    assert slow.returncode == 1
    assert "tests/test_c.py" in final_line(slow) and ", set slow;" in final_line(slow)
    whole = run_tool(tree)  # the default set is every file, and does not read the list
    assert whole.returncode == 1 and FINAL.match(final_line(whole))["failed"] == "1"


def test_fast_and_slow_passed_counts_add_up_to_the_whole_run(tmp_path):
    files = three_files(**{"sub/test_d.py": PASSING})
    tree = slow_tree(tmp_path, ["tests/test_a.py", "tests/sub/test_d.py"], files)
    fast, slow, whole = (run_tool(tree, *a) for a in (("--set", "fast"), ("--set", "slow"), ()))
    assert [d.returncode for d in (fast, slow, whole)] == [0, 0, 0]
    assert (passed_of(fast), passed_of(slow), passed_of(whole)) == ("2", "2", "4")


def test_a_left_out_file_is_left_out_not_skipped(tmp_path):
    tree = slow_tree(tmp_path, ["tests/test_c.py"])
    done = run_tool(tree, "--set", "fast")
    found = FINAL.match(final_line(done))
    assert done.returncode == 0 and found["skipped"] == "0" and found["passed"] == "2"
    assert "tests/test_c.py" not in done.stdout and "deselected" not in done.stdout


def test_a_skipped_test_in_the_fast_set_or_the_slow_set_is_still_2(tmp_path):
    fast_skip = slow_tree(
        tmp_path / "f", ["tests/test_c.py"], three_files(**{"test_d.py": SKIPPED})
    )
    assert run_tool(fast_skip, "--set", "fast").returncode == 2
    slow_skip = slow_tree(
        tmp_path / "s", ["tests/test_d.py"], three_files(**{"test_d.py": SKIPPED})
    )
    assert run_tool(slow_skip, "--set", "slow").returncode == 2
    assert run_tool(slow_skip, "--set", "fast").returncode == 0


def test_fast_and_slow_cover_every_collected_file_once(tmp_path):
    tree = slow_tree(tmp_path, ["tests/test_b.py"], three_files(**{"sub/test_d.py": PASSING}))
    plan = lambda chosen: tool.plan(tree, "pyproject.toml", 3, tool.WEIGHTS_FILE, test_set=chosen)  # noqa: E731
    whole, fast, slow = plan("all")[0], plan("fast"), plan("slow")
    assert set(fast[0]) | set(slow[0]) == set(whole) and not set(fast[0]) & set(slow[0])
    assert sorted(f for g in fast[1] for f in g) == sorted(fast[0])
    assert sorted(f for g in slow[1] for f in g) == ["tests/test_b.py"]


def test_the_set_chosen_from_a_collection_is_checked():
    collected = {"tests/a.py": 1, "tests/b.py": 2}
    assert tool.select_set(collected, ["tests/b.py"], "default", "fast") == {"tests/a.py": 1}
    assert tool.select_set(collected, ["tests/b.py"], "default", "slow") == {"tests/b.py": 2}
    assert tool.select_set(collected, ["tests/b.py"], "default", "all") == collected
    with pytest.raises(tool.CouldNotRun, match="no test of it was collected"):
        tool.select_set(collected, ["tests/c.py"], "default", "fast")


def test_a_listed_file_that_does_not_exist_is_2_in_both_sets(tmp_path):
    tree = slow_tree(tmp_path, ["tests/test_a.py", "tests/test_gone.py"])
    for chosen in ("fast", "slow"):
        done = run_tool(tree, "--set", chosen)
        assert done.returncode == 2
        assert "slow.txt:2: tests/test_gone.py does not exist" in done.stdout


def test_a_listed_file_with_no_collected_test_is_2(tmp_path):
    tree = slow_tree(tmp_path, ["tests/test_empty.py"], three_files(**{"test_empty.py": "X = 1\n"}))
    for chosen in ("fast", "slow"):
        done = run_tool(tree, "--set", chosen)
        assert done.returncode == 2
        assert "tests/test_empty.py is in the slow list but no test of it" in done.stdout


def test_a_set_with_no_tests_is_2(tmp_path):
    every = ["tests/test_a.py", "tests/test_b.py", "tests/test_c.py"]
    done = run_tool(slow_tree(tmp_path / "a", every), "--set", "fast")
    assert done.returncode == 2 and "the fast set of the default lane holds no tests" in done.stdout
    none_slow = slow_tree(tmp_path / "n", [])
    done = run_tool(none_slow, "--set", "slow")
    assert done.returncode == 2 and "the slow set of the default lane holds no tests" in done.stdout
    assert run_tool(none_slow, "--set", "fast").returncode == 0


EXPECTED = "expected '<path>  # <reason>'"
NOT_A_PATH = "is not a test file path in the root"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("tests/test_a.py\n", f"slow.txt:1: {EXPECTED}"),
        ("tests/test_a.py  #   \n", f"slow.txt:1: {EXPECTED}"),
        ("tests/test_a.py tests/test_b.py  # two\n", EXPECTED),
        ("/tests/test_a.py  # root\n", NOT_A_PATH),
        ("tests/../tests/test_a.py  # up\n", NOT_A_PATH),
        ("tests/  # a folder\n", NOT_A_PATH),
        (
            "tests/test_a.py  # one\ntests/test_a.py  # again\n",
            "slow.txt:2: tests/test_a.py is listed twice",
        ),
    ],
)
def test_a_badly_formed_slow_list_is_2(tmp_path, text, message):
    tree = slow_tree(tmp_path, [])
    write_list(tree, text)
    done = run_tool(tree, "--set", "fast")
    assert done.returncode == 2 and message in done.stdout


def test_a_missing_slow_list_is_2_for_a_set_and_not_read_for_all(tmp_path):
    tree = make_tree(tmp_path, three_files())
    done = run_tool(tree, "--set", "fast")
    assert done.returncode == 2 and "cannot read the slow list slow.txt" in done.stdout
    assert run_tool(tree).returncode == 0


def test_comments_and_blank_lines_in_the_slow_list_are_allowed(tmp_path):
    tree = slow_tree(tmp_path, [])
    write_list(tree, "# the slow files\n\ntests/test_c.py  # takes long  # really\n")
    done = run_tool(tree, "--set", "fast")
    assert done.returncode == 0 and passed_of(done) == "2"


def test_an_unknown_set_is_2_and_runs_nothing(tmp_path):
    tree = slow_tree(tmp_path, ["tests/test_c.py"])
    done = run_tool(tree, "--set", "medium")
    assert done.returncode == 2 and "run_tests: exit" not in done.stdout


def test_a_listed_learn_file_belongs_to_the_learn_lane_only(tmp_path):
    tree = lane_tree(tmp_path, {"test_l.py": FAILING, "test_m.py": PASSING})
    write_list(tree, "tests/learn/test_l.py  # learn: takes long\n")
    default_fast = run_tool(tree, "--set", "fast")
    assert default_fast.returncode == 0 and passed_of(default_fast) == "3"
    default_slow = run_tool(tree, "--set", "slow")  # no default-lane file is listed
    assert default_slow.returncode == 2 and "holds no tests" in default_slow.stdout
    learn_fast = run_tool(tree, "--lane", "learn", "--set", "fast")
    assert learn_fast.returncode == 0 and passed_of(learn_fast) == "1"
    learn_slow = run_tool(tree, "--lane", "learn", "--set", "slow")
    assert learn_slow.returncode == 1 and "tests/learn/test_l.py" in final_line(learn_slow)


def test_a_listed_default_file_is_ignored_by_the_learn_lane(tmp_path):
    tree = lane_tree(tmp_path, {"test_l.py": PASSING})
    write_list(tree, "tests/test_a.py  # takes long\n")
    done = run_tool(tree, "--lane", "learn", "--set", "fast")
    assert done.returncode == 0 and passed_of(done) == "1"
    assert run_tool(tree, "--lane", "learn", "--set", "slow").returncode == 2


def test_the_sequential_flag_follows_the_sets(tmp_path):
    tree = slow_tree(tmp_path, ["tests/test_c.py"], three_files(**{"test_c.py": FAILING}))
    assert run_tool(tree, "--sequential", "--set", "fast").returncode == 0
    assert run_tool(tree, "--sequential", "--set", "slow").returncode == 1
    assert run_tool(tree, "--sequential").returncode == 1
    write_list(tree, "tests/test_gone.py  # missing\n")
    assert run_tool(tree, "--sequential", "--set", "fast").returncode == 2
    write_list(tree, "")
    assert run_tool(tree, "--sequential", "--set", "slow").returncode == 2


def test_a_slow_list_named_on_the_command_line_is_used(tmp_path):
    tree = slow_tree(tmp_path, ["tests/test_a.py"])
    other = tmp_path / "other.txt"
    other.write_text("tests/test_b.py  # takes long\ntests/test_c.py  # takes long\n")
    done = run_tool(tree, "--set", "fast", "--slow-list", str(other))
    assert done.returncode == 0 and passed_of(done) == "1"


# --- this repository ---------------------------------------------------------------------------


def listed_ids() -> list[str]:
    """Every test id of the real suite, read straight from one collection-only call."""
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-c", "pyproject.toml", "--collect-only"],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=environment(),
        timeout=600,
        check=False,
    )
    ids = [line for line in done.stdout.splitlines() if re.match(r"^\S+\.py::\S", line)]
    assert len(ids) > 100, done.stdout[-500:]
    return ids


def test_every_test_of_this_repository_is_in_exactly_one_group():
    collected, groups = tool.plan(REPO, "pyproject.toml", tool.DEFAULT_JOBS, tool.WEIGHTS_FILE)
    ids = listed_ids()
    group_of = {file: number for number, group in enumerate(groups) for file in group}
    assert sum(len(group) for group in groups) == len(group_of) == len(collected)
    assert sorted(group_of) == sorted({i.split("::", 1)[0] for i in ids})
    assert sum(collected.values()) == len(ids)
    per_group = [0] * len(groups)
    for test_id in ids:
        per_group[group_of[test_id.split("::", 1)[0]]] += 1
    assert sum(per_group) == len(ids)
    assert all(per_group)


def test_the_weights_table_names_only_files_and_folders_that_exist():
    table = tool.read_weights(tool.WEIGHTS_FILE)
    assert table
    for name in table:
        assert (REPO / name).exists(), name
        assert name.endswith("/") == (REPO / name).is_dir(), name


def test_the_slow_list_names_only_existing_files_each_with_a_reason():
    listed = tool.read_slow_list(REPO / tool.SLOW_LIST, REPO)
    assert listed
    assert listed == sorted(set(listed))


def test_the_fast_and_the_slow_set_of_this_repository_cover_every_test_once():
    listed = tool.read_slow_list(REPO / tool.SLOW_LIST, REPO)
    args = (REPO, "pyproject.toml", tool.DEFAULT_JOBS, tool.WEIGHTS_FILE)
    whole, fast, slow = (tool.plan(*args, test_set=name)[0] for name in ("all", "fast", "slow"))
    assert set(fast) | set(slow) == set(whole) and not set(fast) & set(slow)
    assert set(slow) == {f for f in listed if tool.lane_of(f) == "default"}
    assert sum(fast.values()) + sum(slow.values()) == sum(whole.values()) == len(listed_ids())
