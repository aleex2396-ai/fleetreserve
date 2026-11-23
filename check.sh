#!/usr/bin/env bash
# The one command that checks this project.
# Exit 0: every check passed. Exit 1: a check failed. Exit 2: a check could not run. Only 0 is a pass.
# Add one line per check below. A check that is skipped, missing or empty counts as "could not run".
# Scope: this defends against accidents under an honest caller (ordinary environment variables,
# stray settings files, skipped or unrun tests). It does not defend against deliberate sabotage
# (a rigged shell, fake tools on the search path, test code that ends the process, edits to the
# check files); that is left to review of the change.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 2
# shellcheck source=tools/check_lib.sh
source tools/check_lib.sh || { echo "COULD NOT RUN: tools/check_lib.sh is missing"; exit 2; }

check "settings files" uv run --frozen python tools/settings_files.py
check "repository rules" uv run --frozen python tools/repo_rules.py
check "lockfile" uv lock --check --offline
check "style: lint" uv run --frozen ruff check .
check "style: format" uv run --frozen ruff format --check .
# The run list's start check: its form, its units, its starts at a made-up tag, and the files it reads
# (learn/drift-eta.json among them). It exits 1 when the list would refuse a drive, so such a list
# cannot reach main. It needs the `learn` and `baselines` packages; uv exits 2 if they cannot be set up.
check "run list start check" uv run --frozen --group learn --group baselines python -m fleetreserve.run_list run-lists/run-list.json
# The tests run in two sets (tests/slow.txt names the slow files). This check runs the fast set
# of each lane; the slow set runs in tools/check-slow.sh, which must run before the tag and on any
# change to the held-out data, the seal or the sealed-run path (its header says so).
check "tests" uv run --frozen python tools/run_tests.py --set fast
# The learn lane: tests/learn needs the packages of the `learn` dependency group, so it runs apart
# from the default tests. If that environment cannot be set up (for example with no network), uv
# exits 2 and the result is "could not run", never "failed". The packages install once and stay.
check "tests: learn lane" uv run --frozen --group learn python tools/run_tests.py --lane learn --set fast

check_finish
