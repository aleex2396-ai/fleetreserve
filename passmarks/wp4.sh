# The pass mark of work package 4 (plan section 13, WP4), sourced by passmark.sh.
# The plan's sentence "T1, flexitroid, Frank 2011 projection check, timing. Pass: T1's checks;
# solve times of the box LP and the tree (100 cars, 8 hours) and of LP (24) (50 cars, 14 hours)
# logged with the machine they ran on. Run time is reported, never judged" has one line per
# part, in its order. A line is changed only together with tests/test_passmark.py, which names
# all ten. Flexitroid was installed on 11 Nov 2026 (no record of the approval is kept); it is
# pinned to one commit in uv.lock. The flexitroid item reads the newest recorded run
# labelled wp4-flexitroid (run `python -m fleetreserve.run_record --name wp4-flexitroid ... --
# uv run --frozen python -m fleetreserve.flexitroid_days` first); without a verified run covering
# 50 development days, recorded at the checked-out commit or read from an earlier commit by the
# rule of fleetreserve.run_evidence, the item could not run. Each timing item reads the newest
# recorded run labelled wp4-timing (run `python -m fleetreserve.run_record --name wp4-timing ...
# -- uv run --frozen python -m fleetreserve.timing_marks` first) and passes when that run,
# verified, and recorded at the checked-out commit or read from an earlier commit by that rule,
# records every solve time with the machine it ran on; no time and no machine load is judged.
# Without such a run, a timing item could not run, never a pass. If a later run is too slow to
# finish, 2-hour slots or a day sample, as the project owner chooses (plan section 13, WP4).
check "T1 one-way box check against vertex LPs, 400 of 400 (plan 4)" uv run --frozen pytest -c pyproject.toml tests/wp1/test_t1_one_way.py
check "T1 two-way box check against vertex LPs, 300 of 300 (plan 4)" uv run --frozen pytest -c pyproject.toml tests/wp1/test_t1_two_way.py
check "T1 floors and free recovery hours, 600 of 600 (plan 4)" uv run --frozen pytest -c pyproject.toml tests/wp1/test_t1_floors_recovery.py
check "T1 band scale (c), 400 of 400 (plan 4)" uv run --frozen pytest -c pyproject.toml tests/wp1/test_t1_band.py
check "T1 on the day model: box check, scale and ceilings from a day's set functions against linear programmes (plan 2, 4)" uv run --frozen pytest -c pyproject.toml tests/wp4/test_t1_exact.py tests/wp4/test_ceilings.py
check "flexitroid matches T1(b) on 50 development days (plan 4)" uv run --frozen python tools/flexitroid_check.py
check "Frank 2011 projection check: the fleet set on S has only the faces p(A) <= u(A) <= b(A) (plan 4)" uv run --frozen pytest -c pyproject.toml tests/wp4/test_projection.py
check "box LP solve times (100 cars, 8 hours) logged with the machine they ran on (plan 13 WP4)" uv run --frozen python tools/timing_marks_check.py --box
check "tree solve times (100 cars, 8 hours) logged with the machine they ran on (plan 13 WP4)" uv run --frozen python tools/timing_marks_check.py --tree
check "LP (24) solve times (50 cars, 14 hours) logged with the machine they ran on (plan 13 WP4)" uv run --frozen python tools/timing_marks_check.py --lp24
