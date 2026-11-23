# Library for check.sh: runs listed checks and turns their exit statuses into one result.
# Source it; do not run it. A check's exit 0 is passed, 1 is failed, anything else is could not run.
ran=0; failed=0; could_not_run=0

check() { # check "<name>" <command> [arguments]
  local name="$1"; shift
  # Each check runs in a scrubbed environment so nothing set by the caller (PYTEST_ADDOPTS,
  # UV_*, VIRTUAL_ENV and the like) can change what is checked. Kept: PATH, to find commands,
  # and HOME, where uv keeps its cache.
  # env is called by full path so that a shell function named env cannot replace it.
  /usr/bin/env -i PATH="${PATH-}" HOME="${HOME-}" "$@"
  local status=$?
  ran=$((ran + 1))
  if [ "$status" -eq 0 ]; then echo "passed: $name"
  elif [ "$status" -eq 1 ]; then echo "FAILED: $name"; failed=1
  else echo "COULD NOT RUN: $name (exit $status)"; could_not_run=1
  fi
}

check_finish() { # print the overall line and exit: 1 if any failed, else 2 if any could not run or none ran, else 0
  if [ "$failed" -eq 1 ]; then exit 1; fi
  if [ "$could_not_run" -eq 1 ]; then exit 2; fi
  if [ "$ran" -eq 0 ]; then echo "COULD NOT RUN: no checks are listed in check.sh yet"; exit 2; fi
  echo "all $ran checks passed"
  exit 0
}
