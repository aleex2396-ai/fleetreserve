# The pass mark of work package 10 (plan section 13, WP10: "Baselines, no-pause sensitivity.
# Pass: shared offers, paths, output files"), sourced by passmark.sh. The item has exactly one line
# here; it is changed only together with tests/test_passmark.py, which names it. The methods are
# the 11 that plan section 10 names and a method of the registry implements, listed in
# tools/compare_shared_check.py with the registry name of each, and the model is lgbm_tail, the
# main forecasting model and the one the WP6 gate is read with (the owner's decisions of 16
# November 2026). For each model it reads the newest recorded dial-choice run (label
# wp6-dial-choice-<model>) for its dials, then per dial the newest recorded comparison run (label
# wp10-compare-<model>-eps<dial>, `python -m fleetreserve.run_record --name
# wp10-compare-<model>-eps<dial> --long --max-seconds <n> -- uv run --frozen --group learn python
# -m fleetreserve.compare_run --model <model> --dial <dial> --matrix gate/matrix.json --splits 200
# --workers <n>`) and the same run with
# `--no-pause-share 1/3` (label wp10-compare-<model>-eps<dial>-no-pause), each verified with its
# recorded truth build, every run recorded at the checked-out commit or read from an earlier
# commit by the rule of fleetreserve.run_evidence. A method with no implementation, a requirement
# no command writes, no run, a stale run or one that cannot decide: the item could not run and is
# never a pass. A run read with a method missing, a method on other days, paths or ceilings, or an
# output file missing or misnamed fails it.
check "baselines and comparison methods on shared offers, paths and output files, with the no-pause sensitivity, in the recorded comparison runs (plan 13 WP10)" uv run --frozen python tools/compare_shared_check.py
