"""The test helper that runs git names the cause when git fails."""

import json
import os
import subprocess

import pytest
from git_run import GitFailed, run_git


def test_a_failed_git_command_names_its_command_folder_and_error_text(tmp_path):
    with pytest.raises(GitFailed) as caught:
        run_git(
            ["git", "-C", str(tmp_path / "no_such_folder"), "add", "-A"],
            capture_output=True,
            text=True,
            check=True,
        )
    message = str(caught.value)
    assert "exit status 128" in message
    assert "add -A" in message
    assert "no_such_folder" in message
    assert "current folder:" in message
    assert "fatal" in message


def test_the_error_is_still_a_called_process_error(tmp_path):
    with pytest.raises(subprocess.CalledProcessError) as caught:
        run_git(["git", "-C", str(tmp_path / "x"), "status"], capture_output=True, check=True)
    assert caught.value.returncode == 128
    assert b"fatal" in caught.value.stderr
    assert "fatal" in str(caught.value)


def test_a_commit_starts_no_background_upkeep(tmp_path):
    trace = tmp_path / "trace.json"
    repo = tmp_path / "repo"
    environment = {**os.environ, "GIT_TRACE2_EVENT": str(trace)}
    for arguments in (
        ["init", "-q", str(repo)],
        ["-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
        + ["commit", "-q", "--allow-empty", "-m", "first"],
    ):
        run_git(["git", *arguments], capture_output=True, check=True, env=environment)
    started = [
        json.loads(line).get("argv", [])
        for line in trace.read_text().splitlines()
        if json.loads(line).get("event") == "start"
    ]
    assert started, "the trace holds no start event: the check could not see git"
    upkeep = [argv for argv in started if {"maintenance", "gc", "repack"} & set(argv)]
    assert upkeep == []


def test_a_successful_command_returns_its_output(tmp_path):
    done = run_git(
        ["git", "init", "--quiet", str(tmp_path)], capture_output=True, text=True, check=True
    )
    assert done.returncode == 0
