#!/usr/bin/env bash
# The slow tests of this project: the files listed in tests/slow.txt, which `bash check.sh` leaves out.
# Run it as `bash tools/check-slow.sh`, from any folder.
# It must run:
#   - once before the tag, on the commit that is tagged, and
#   - on any change to the held-out data, the seal or the sealed-run path, before the change is accepted.
# `bash check.sh` runs every other test on every change; the two together run every test once.
# Exit 0: every slow test passed. Exit 1: one failed. Exit 2: could not run. Only 0 is a pass.
# A slow list that names a missing file, or a set with no test, is "could not run", never a pass.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2
# shellcheck source=tools/check_lib.sh
source tools/check_lib.sh || { echo "COULD NOT RUN: tools/check_lib.sh is missing"; exit 2; }

check "tests: slow set" uv run --frozen python tools/run_tests.py --set slow
# The learn lane needs the packages of the `learn` dependency group (see check.sh).
check "tests: learn lane, slow set" uv run --frozen --group learn python tools/run_tests.py --lane learn --set slow

check_finish
