# The pass mark of work package 2 (plan section 13, WP2), sourced by passmark.sh.
# The plan's sentence "training and development truth sets; fallback shares logged" has one line
# per part, in its order. A line is changed only together with tests/test_passmark.py, which names
# all three. Each item reads the recorded truth build at the checked-out commit (run
# `python -m fleetreserve.run_record --name wp2-truth -- uv run --frozen python -m
# fleetreserve.truth_build` first) and the tables it wrote. Without data/raw, that run or a
# table, an item is "could not run" (exit 2), never a pass. In item 3 a fallback share written as
# null (an undefined share, 0 of 0) is shown and passes only when its denominator is 0.
check "training truth sets built and consistent (plan 13 WP2)" uv run --frozen python -m fleetreserve.truth_verify train
check "development truth sets built and consistent (plan 13 WP2)" uv run --frozen python -m fleetreserve.truth_verify development
check "fallback shares logged for every unit (plan 13 WP2)" uv run --frozen python tools/wp2_fallback_check.py
