"""A made-up solver answer for the O3 tests that need solves to run out of time.

A real time limit of a microsecond does not make a timeout certain. HiGHS checks its clock only at
some points (`lp` module text), so a programme small enough can finish before the first check and
come back optimal, late; `lp.solve_lp` then reports "optimal", as its module text says it must.
A test that forces a timeout with a tiny real limit therefore races the clock.

`force_timeouts(monkeypatch, limit)` replaces `lp.linprog`, the code's one call of the solver
(repository rule solver-door), for the rest of the test. A call given the time limit `limit`
returns, without solving, what scipy returns when HiGHS stops at its time limit: status 1 and
the message "Time limit reached.", here with a partial answer of zeros, so that code reading a
timed-out solve as a number would read 0. Every other call goes to the solver. `lp.solve_lp`'s
reading of the answer and all the code above it run unchanged. The list returned gets one entry
(the number of variables) per solve so answered.
"""

import numpy as np
from scipy.optimize import OptimizeResult

from fleetreserve import lp

TIME_LIMIT_STATUS = 1  # scipy's status for HiGHS's "time limit reached" (`lp` module text)
MESSAGE = "Time limit reached. (HiGHS Status 13: made-up answer, no solve)"


def force_timeouts(monkeypatch, limit: float) -> list[int]:
    """Every solve given `limit` seconds ends as timed out from now on (module text)."""
    real = lp.linprog
    forced = []

    def stand_in(c, *args, options=None, **kwargs):
        if options is not None and options.get("time_limit") == limit:
            forced.append(len(c))
            return OptimizeResult(
                x=np.zeros(len(c)),
                fun=0.0,
                status=TIME_LIMIT_STATUS,
                success=False,
                message=MESSAGE,
                nit=0,
            )
        return real(c, *args, options=options, **kwargs)

    monkeypatch.setattr(lp, "linprog", stand_in)
    return forced
