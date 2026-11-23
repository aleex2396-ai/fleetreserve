"""The sizing what-if's `scalar` definition (`size_whatif_summary.definitions`) names the driver
ramps' slopes as what a size multiplies exactly when multiplying the ramps alone reproduces the
driver loss the sizing scores (the other definitions: `tests/wp11/test_results_definitions.py`).

The days are the made-up days of `size_fake` with early departures, each scored at size 0.5 and
scalars 1, 2 and 4 by `size_scale.score`. The ramps alone, with their slopes halved
(`gate_calibrate.ramp_loss` on `(c, s / 2, total)` at the scalar), give the earlier bound's value;
the size also divides the power rows' thresholds and multiplies the gains the bound reads
(`size_scale._sized`), so on a day where a power row breaks the two differ.
"""

import numpy as np
import pytest
from size_fake import prepared_days

from fleetreserve import gate_calibrate, size_scale, size_whatif_summary

SIZE = 0.5
SCALARS = (1.0, 2.0, 4.0)


def test_the_sizing_definition_names_the_ramps_alone_exactly_when_they_alone_give_the_score():
    days = prepared_days(21, 90, early=0.04)
    terms = [size_scale.day_terms(p) for p in days]
    q = np.where([t is None for t in terms], np.nan, SIZE)
    offering = [d for d, t in enumerate(terms) if t is not None]
    assert offering
    differs = 0
    for gamma in SCALARS:
        scored = size_scale.score(terms, q, gamma, offering).driver_losses
        for d, loss in zip(offering, scored, strict=True):
            c, s, total = terms[d].driver
            ramps_alone = gate_calibrate.ramp_loss((c, s * SIZE, total), gamma)
            differs += ramps_alone != pytest.approx(loss)
    assert differs > 0  # a power row breaks on some day: the ramps alone are not the score
    says_ramps_alone = "driver ramps" in size_whatif_summary.definitions()["scalar"]
    assert says_ramps_alone == (differs == 0)
