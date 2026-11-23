# The pass mark of work package 1 (plan section 13, WP1), sourced by passmark.sh.
# Every item of the plan's pass mark has exactly one line here, in the plan's order. A line is
# changed only together with the test it points at (tests/test_passmark.py names all twelve).
# A missing test file is "could not run", never a pass.
check "hashes committed (plan 13 WP1)" uv run --frozen python tools/data_hashes.py --check
check "T1 one-way box check against vertex LPs, 400 of 400 (plan 4)" uv run --frozen pytest -c pyproject.toml tests/wp1/test_t1_one_way.py
check "T1 two-way box check against vertex LPs, 300 of 300 (plan 4)" uv run --frozen pytest -c pyproject.toml tests/wp1/test_t1_two_way.py
check "T1 floors and free recovery hours, 600 of 600 (plan 4)" uv run --frozen pytest -c pyproject.toml tests/wp1/test_t1_floors_recovery.py
check "T1 band scale (c), 400 of 400 (plan 4)" uv run --frozen pytest -c pyproject.toml tests/wp1/test_t1_band.py
check "T3 one-car two-hour example: hindsight 0.10, upper-corner path 0.15 (plan 4)" uv run --frozen pytest -c pyproject.toml tests/wp1/test_t3_example.py
check "T4 proposition on toy fleets, 300 of 300 (plan 4)" uv run --frozen pytest -c pyproject.toml tests/wp1/test_t4_proposition.py
check "T4 conjecture: no gap on 22400 seeded toy fleets (plan 4)" uv run --frozen pytest -c pyproject.toml tests/wp1/test_t4_conjecture.py
check "T4 gap example: hindsight 1.0, tree and affine 0.556 (plan 4)" uv run --frozen pytest -c pyproject.toml tests/wp1/test_t4_gap_example.py
check "T5 discharge-after-failure case: 0.5 kWh two-way, 1.0 under the plan (plan 5)" uv run --frozen pytest -c pyproject.toml tests/wp1/test_t5_case.py
check "T10 drift-guard cases: clipped 0.500, unclipped 0.103 at t=116 (plan 6)" uv run --frozen pytest -c pyproject.toml tests/wp1/test_t10_cases.py
check "full per-car matrices confirmed in the earlier analysis's offer code (plan 13 WP1)" uv run --frozen pytest -c pyproject.toml tests/wp1/test_earlier_offer_matrices.py
