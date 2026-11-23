"""Run git in a made-up repository and name the cause when it fails.

`run_git` takes the arguments of `subprocess.run`. With `check=True` and captured output a failure
raises `GitFailed`, which carries the command, the exit status, git's standard error and standard
output, and the folder it ran in, so a failure that comes by chance names its cause.
`GitFailed` is a `subprocess.CalledProcessError`, so a test that catches that still works.
"""

import os
import subprocess


class GitFailed(subprocess.CalledProcessError):
    def __init__(self, error: subprocess.CalledProcessError, cwd):
        super().__init__(error.returncode, error.cmd, output=error.output, stderr=error.stderr)
        self.cwd = cwd

    def __str__(self):
        def text(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")

        command = " ".join(str(part) for part in self.cmd)
        return (
            f"git failed with exit status {self.returncode}\n"
            f"command: {command}\n"
            f"current folder: {self.cwd or os.getcwd()}\n"
            f"git's error text: {text(self.stderr).strip() or '(empty)'}\n"
            f"git's output: {text(self.output).strip() or '(empty)'}"
        )


QUIET = ("-c", "maintenance.auto=false", "-c", "gc.auto=0")


def run_git(command, **kwargs):
    """Run `command` (a list that starts with "git") with git's own background upkeep off.

    Git 2.54 starts `git maintenance run --auto --detach` after a commit. In a made-up repository
    that process packs the loose objects and deletes them while the test still reads them, which
    fails a test by chance under load ("Could not read <object>", `git add -A` or `git tag` exit
    128). The two settings stop it.
    """
    if command and command[0] == "git":
        command = [command[0], *QUIET, *command[1:]]
    try:
        return subprocess.run(command, **kwargs)
    except subprocess.CalledProcessError as error:
        raise GitFailed(error, kwargs.get("cwd")) from None
