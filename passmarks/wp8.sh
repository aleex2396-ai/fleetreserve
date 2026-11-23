# The pass mark of work package 8 (plan section 13, WP8: "development pinball loss, concordance
# index and offer (WP6 calibration) logged per model"), sourced by passmark.sh. The item has
# exactly one line here; it is changed only together with tests/test_passmark.py, which names it.
# The models are the forecast models plan section 7 names, listed in tools/learn_logged_check.py.
# For each, it reads the newest recorded learn run (label wp8-<model>-gate-matrix,
# `python -m fleetreserve.run_record --name wp8-<model>-gate-matrix --long --max-seconds <n> --
# uv run --frozen --group learn python -m fleetreserve.learn_run --model <model> --matrix
# gate/matrix.json --workers <n>`) and needs that run's manifest to verify, on the committed
# gate/matrix.json, with the run recorded at the checked-out commit or read from an earlier commit
# by the rule of fleetreserve.run_evidence. A model with no module or no such run, a stale run (at
# another commit and not read by that rule), a run on another matrix, a trial or partial run: the
# item could not run and is never a pass. A run read with a value missing fails it.
check "development pinball loss, concordance index and offer (WP6 calibration) logged per model of plan section 7 in its recorded learn run (plan 13 WP8)" uv run --frozen python tools/learn_logged_check.py
