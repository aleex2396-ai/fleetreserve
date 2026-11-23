# The pass mark of work package 12 (plan section 13, WP12: "Dress rehearsal of held-out scripts on
# development units. Pass: done, its duration logged"), sourced by passmark.sh. The item has
# exactly one line here; it is changed only together with tests/test_passmark.py, which names it.
# tools/rehearsal_check.py reads the one run list committed at its fixed path
# run-lists/run-list.json, from which both the rehearsal and the run from the tag are derived, and
# the newest driver log of its rehearsal (runs.noindex/drives/<UTC start>-rehearsal.jsonl, written
# by `caffeinate -i uv run --frozen --group learn --group baselines python -m
# fleetreserve.run_driver run-lists/run-list.json --purpose rehearsal`, one recorded run per step
# of the rehearsal), and needs the log's hash to
# recompute, the log to be of the checked-out commit or of an earlier commit when every path changed
# since is on run_evidence's keep-list, of exactly the list's committed bytes and of
# the rehearsal's steps as the list instantiates them, every step to have run and exited 0 with no
# timeout, each step's recorded run to verify at the commit the log was written at, and the total
# duration to be logged.
# It prints every difference between the rehearsal and the run from the tag (the units of a
# group, entries of one purpose only, arguments of one purpose only) and judges none of them. A
# logged step that failed fails the item; no run list, no log, a stale log, a drive that stopped or
# broke off, or a run that does not verify: the item could not run and is never a pass.
check "dress rehearsal of the held-out scripts on development units done, its duration logged, in the driver log of the rehearsal of the committed run list run-lists/run-list.json, from which the run from the tag is derived (plan 13 WP12; the differences between the two are listed, not judged)" uv run --frozen python tools/rehearsal_check.py
