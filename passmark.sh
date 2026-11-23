#!/usr/bin/env bash
# The pass mark of one work package: "is this work package finished?" (plan section 13).
# Run as `bash passmark.sh <work package>`, for example `bash passmark.sh wp1`.
# The items are listed in passmarks/<work package>.sh, one `check "<item>" <command>` line each.
# Exit 0: every item passed. Exit 1: an item failed. Exit 2: an item could not run, no item is
# listed, or the call was wrong. Only 0 is a pass. A missing test file is "could not run".
# `bash check.sh` is the separate check that guards every change; it does not decide this.
# If FLEETRESERVE_RUN_DIR is set (the run recorder sets it), everything printed for the items and
# the count line is also written to passmark-<work package>.txt in that folder.
# Scope: this defends against accidents under an honest caller (an item forgotten, a missing
# test file read as a pass, an item dropped from the list). It does not defend against deliberate
# tampering.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 2
# shellcheck source=tools/check_lib.sh
source tools/check_lib.sh || { echo "COULD NOT RUN: tools/check_lib.sh is missing"; exit 2; }

usage() {
  local known="" file
  for file in passmarks/*.sh; do
    [ -f "$file" ] || continue
    file="${file#passmarks/}"
    known="$known ${file%.sh}"
  done
  echo "usage: bash passmark.sh <work package>   (known:${known:- none})" >&2
  exit 2
}

name="${1-}"
[ "$#" -eq 1 ] || usage
[[ "$name" =~ ^[a-z0-9_-]+$ ]] || usage
list="passmarks/$name.sh"
[ -f "$list" ] || usage

results=""
if [ -n "${FLEETRESERVE_RUN_DIR-}" ]; then
  results="$FLEETRESERVE_RUN_DIR/passmark-$name.txt"
  : > "$results" || { echo "COULD NOT RUN: cannot write $results"; exit 2; }
fi

emit() { # print a line, and keep it in the results file when there is one
  echo "$1"
  if [ -n "$results" ]; then echo "$1" >> "$results"; fi
}

# Count the outcomes by wrapping the library's check: its printed line says what happened.
n_passed=0; n_failed=0; n_cannot=0
eval "lib_$(declare -f check)"
check() { # check "<item>" <command> [arguments]
  local item="$1" out last
  out=$(mktemp) || { echo "COULD NOT RUN: no temporary file"; exit 2; }
  lib_check "$@" > "$out" 2>&1
  cat "$out"
  if [ -n "$results" ]; then cat "$out" >> "$results"; fi
  last=$(tail -n 1 "$out")
  rm -f "$out"
  case "$last" in
    *"passed: $item") n_passed=$((n_passed + 1)) ;;
    *"FAILED: $item") n_failed=$((n_failed + 1)) ;;
    *) n_cannot=$((n_cannot + 1)) ;;
  esac
}

source "$list" || { echo "COULD NOT RUN: $list could not be read"; exit 2; }

total=$((n_passed + n_failed + n_cannot))
if [ "$total" -eq 0 ]; then emit "COULD NOT RUN: no items are listed in $list"; fi
emit "$n_passed passed, $n_failed failed, $n_cannot could not run of $total items"
if [ "$total" -eq 0 ]; then exit 2; fi
check_finish
