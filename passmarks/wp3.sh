# The pass mark of work package 3 (plan section 13, WP3), sourced by passmark.sh.
# The plan's sentence "Pass: 45.6 kWh ± 0.1 kWh, 0.4% and 3.9% ± 0.1 point" has exactly one line
# here per number, in its order. A line is changed only together with the test it is pinned by
# (tests/test_passmark.py names all three). Each item reads the newest recorded re-run (label
# wp3-rerun) and needs that run's manifest to verify. How long the run took is never judged.
# With no such run, or a run that did not finish (failed, or stopped by the recorder's stop
# limit, a guard against a hanging run), an item could not run and is never a pass (see
# tools/earlier_rerun_check.py).
check "re-run perfect-knowledge offer is 45.6 kWh, 0.1 kWh either way (plan 13 WP3)" uv run --frozen python tools/earlier_rerun_check.py --kwh
check "re-run learned share is 0.4%, 0.1 point either way (plan 13 WP3)" uv run --frozen python tools/earlier_rerun_check.py --learned
check "re-run population share is 3.9%, 0.1 point either way (plan 13 WP3)" uv run --frozen python tools/earlier_rerun_check.py --population
