# The pass mark of work package 5 (plan section 13, WP5), sourced by passmark.sh.
# The plan's sentence "diag <= affine <= tree <= hindsight every development day; T2's check;
# two-way corner tests match the LP; T4 log started, split by in-window departure" has one line
# per part, in its order. A line is changed only together with tests/test_passmark.py, which names
# all four. Each item reads the newest recorded run labelled wp5-dev-days (run
# `python -m fleetreserve.run_record --name wp5-dev-days ... -- uv run --frozen python -m
# fleetreserve.dev_day_checks ...` first, with the arguments named on the T2 line); without a
# verified run with the configuration named, recorded at the checked-out commit or read from an
# earlier commit by the rule of fleetreserve.run_evidence, an item could not run, never a pass. A
# timeout is missing, never a pass.
# The T2 line's forecast configuration: rate and energy known was decided by the project owner
# on 16 Nov 2026 and a car with no usable forecast left unsteered on 18 Nov 2026. Kappa 0.1 and
# the departure quantile as planned (not floored) are the stated choice of this check; the kappa
# for the gate is the plan's rule (section 7) and the departure floor has no recorded decision.
# The sample size 100 is the plan's (section 5).
check "diag <= affine <= tree <= hindsight on every development day, hourly and block products (plan 13 WP5)" uv run --frozen python tools/dev_day_check.py --order
check "T2's check on 100 development days: closed-form worst case equals replay of all corner paths, no violation where E' holds, share of days in E' shown (plan 5)" uv run --frozen python tools/dev_day_check.py --t2 --sample-size 100 --energy-bounds known --uncertifiable-car leave-unsteered --kappa 0.1 --departure-floor as-planned
check "two-way corner tests match the LP at B = 5 kWh on 100 development days (plan 9, 13 WP5)" uv run --frozen python tools/dev_day_check.py --two-way --sample-size 100
check "T4 log started on every development day, split by in-window departure (plan 4)" uv run --frozen python tools/dev_day_check.py --t4-log
