# The pass mark of work package 6 (plan section 13, WP6: "T5 on 100% of day-paths"), sourced by
# passmark.sh. The item has exactly one line here; it is changed only together with
# tests/test_passmark.py, which names it. It reads the newest recorded gate run (label wp6-gate,
# `python -m fleetreserve.run_record --name wp6-gate --long --max-seconds <n> -- uv run --frozen
# python -m fleetreserve.gate_run --matrix gate/matrix.json --workers <n>`) and needs that run's
# manifest to verify, on the committed gate/matrix.json, with the run recorded at the checked-out
# commit or read from an earlier commit by the rule of fleetreserve.run_evidence. With no such
# run, a stale run (at another commit and not read by that rule), a run on another matrix, a trial
# or partial run, or no day-path checked, the item could not run and is never a pass (see
# tools/gate_t5_check.py).
check "T5 on 100% of day-paths in the recorded gate run (plan 13 WP6)" uv run --frozen python tools/gate_t5_check.py
