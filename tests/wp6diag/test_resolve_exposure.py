"""`resolve.draft` exposes the re-solve programme without changing it.

`resolve_before_exposure.py` in this folder is the re-solve module as it was before `draft` and
the row kinds were added (its SHA-256 is checked, so it is that file and no other). It is kept
only for these tests; a later deliberate change of the re-solve programme ends their purpose.

1. The same programme. Every re-solve met on seeded made-up days (every rung, scales 0.5 and
   1.5, the days' audited paths) is given to both modules. What each hands to the solver door
   (`lp.solve_lp`: the cost, both row matrices, both right sides, the bounds, the method and the
   time limit) is equal entry for entry, and so are the outcomes and, for optimal re-solves, the
   plans. The sample holds `N_RESOLVES` re-solves.
2. The gate's results. The gate run on made-up gate units (`gate_fake` cars, an early leaver
   every fifth day, the main configuration at kappa 0.1 with departures seen from the
   rounded-down hour) is made twice in this process: with the re-solve as it is now, and with the
   module from before the exposure as the rungs' re-solver. The results document, the day rows
   and the T5 rows are byte-identical, and both runs made the same re-solves, more than
   `MIN_GATE_RESOLVES` of them.
3. The draft's rows carry one kind each, and `resolve` solves exactly the draft's rows.
"""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import resolve_before_exposure as before
from gate_fake import ROUNDED, gate_cars, matrix_document, write_tables
from made_up_offer_days import departure_rule, offer_day

from fleetreserve import gate_config, gate_run, resolve, rungs, sparse_programme
from fleetreserve.audit_paths import audited_paths
from fleetreserve.days_from_truth import load_unit_days

BEFORE_SHA256 = "81cc9d217911ec76bc6d0269aca8315fa65960e963d3d7692fb950f6587b7de5"
SEED = 31
N_DAYS = 12
BETAS = (0.5, 1.5)
N_RESOLVES = 483
SEEN = "rounded-hour"
MIN_GATE_RESOLVES = 1000
GATE_DAYS = (10, 52, 10)  # counting days per made-up gate unit; only the second is calibrated


def test_the_kept_module_is_the_one_from_before_the_exposure():
    assert hashlib.sha256(Path(before.__file__).read_bytes()).hexdigest() == BEFORE_SHA256
    assert not hasattr(before, "draft") and hasattr(resolve, "draft")


def _recorded():
    """Every (state, guard, two-way, budget) the rungs give the re-solver on the sample."""
    rng = random.Random(SEED)
    log = []

    def resolver(state, guard, *, two_way, budget, time_limit):
        log.append((state, guard, two_way, budget))
        return resolve.resolve(state, guard, two_way=two_way, budget=budget, time_limit=300.0)

    for k in range(N_DAYS):
        day, certificate, _ = offer_day(rng, k)
        if certificate.outcome != "optimal":
            continue
        for beta in BETAS:
            for path in audited_paths(day.unit, day.date, day.n_service):
                rungs.run_rungs(
                    day,
                    certificate.policy,
                    beta,
                    path,
                    departure_bound=departure_rule(day, SEEN),
                    departure_seen=SEEN,
                    resolver=resolver,
                )
    return log


def _dense(matrix):
    return None if matrix is None else matrix.toarray()


def _solver_inputs(monkeypatch, module, state, guard, two_way, budget):
    """(what `module.resolve` hands to the solver door, its result)."""
    seen = []
    real = sparse_programme.solve_lp

    def door(c, *, A_ub, b_ub, A_eq, b_eq, bounds, time_limit, method):
        seen.append((c, _dense(A_ub), b_ub, _dense(A_eq), b_eq, bounds, time_limit, method))
        return real(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            time_limit=time_limit,
            method=method,
        )

    monkeypatch.setattr(sparse_programme, "solve_lp", door)
    result = module.resolve(state, guard, two_way=two_way, budget=budget, time_limit=300.0)
    monkeypatch.setattr(sparse_programme, "solve_lp", real)
    (given,) = seen
    return given, result


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, np.ndarray):
        return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
    return a == b


def test_the_programme_handed_to_the_solver_is_the_same_as_before(monkeypatch):
    log = _recorded()
    outcomes = set()
    for state, guard, two_way, budget in log:
        old, old_result = _solver_inputs(monkeypatch, before, state, guard, two_way, budget)
        new, new_result = _solver_inputs(monkeypatch, resolve, state, guard, two_way, budget)
        assert all(_same(a, b) for a, b in zip(old, new, strict=True))
        assert old_result.outcome == new_result.outcome
        assert old_result.sizes == new_result.sizes
        outcomes.add(new_result.outcome)
        if new_result.outcome == "optimal":
            assert np.array_equal(old_result.plan.constant, new_result.plan.constant)
            assert np.array_equal(old_result.plan.gains, new_result.plan.gains)
            assert old_result.plan.two_way == new_result.plan.two_way
    print(f"{len(log)} re-solves compared; outcomes {sorted(outcomes)}")
    assert len(log) == N_RESOLVES
    assert outcomes == {"infeasible", "optimal"}
    assert any(two_way for _, _, two_way, _ in log)


def _gate(monkeypatch, folder, resolver):
    calls = []

    def counted(state, guard, **kwargs):
        calls.append(state.hour)
        return resolver(state, guard, **kwargs)

    monkeypatch.setitem(rungs.run_rungs.__kwdefaults__, "resolver", counted)
    matrix = gate_config.parse(matrix_document(ROUNDED))
    units = {}
    for name, days in zip(matrix.gate_units, GATE_DAYS, strict=True):
        write_tables(folder, name, gate_cars(days, early_every=5))
        units[name] = load_unit_days(
            folder, name, rho=matrix.rho, no_fit=matrix.no_fit, small_fleet=matrix.small_fleet
        )
    (document, day_rows, break_rows), _ = gate_run.run_gate(
        matrix, units, workers=1, only=ROUNDED[0], trial=False
    )
    text = json.dumps(document, indent=1, sort_keys=True, allow_nan=False)
    days = gate_run._csv(day_rows, gate_run.DAY_COLUMNS, None)
    breaks = gate_run._csv(break_rows, gate_run.BREAK_COLUMNS, None)
    return (text, days, breaks), len(calls)


def test_the_gate_s_results_are_byte_identical_before_and_after(monkeypatch, tmp_path):
    now, calls_now = _gate(monkeypatch, tmp_path, resolve.resolve)
    then, calls_then = _gate(monkeypatch, tmp_path, before.resolve)
    print(f"{calls_now} re-solves in each gate run")
    assert calls_now == calls_then and calls_now > MIN_GATE_RESOLVES
    for a, b in zip(now, then, strict=True):
        assert a.encode() == b.encode()
    assert '"failed_configurations": []' in now[0]


def test_resolve_solves_exactly_the_rows_of_the_draft(monkeypatch):
    state, guard, two_way, budget = next(
        entry for entry in _recorded() if entry[0].n_service - entry[0].hour >= 1
    )
    given, _ = _solver_inputs(monkeypatch, resolve, state, guard, two_way, budget)
    built = resolve.draft(state, guard, two_way=two_way, budget=budget)
    for rows, matrix, rhs in ((built.le, given[1], given[2]), (built.eq, given[3], given[4])):
        assert len(rows.kind) == len(rows.rhs) == matrix.shape[0]
        assert set(rows.kind) <= set(resolve.ROW_KINDS)
        np.testing.assert_array_equal(np.array(rows.rhs), rhs)
    assert set(built.eq.kind) == {"request", "centre", "width"}
    assert {"power", "floor", "cap", "need", "guard", "spread"} <= set(built.le.kind)
    assert built.le.kind.count("need") == built.present.size
