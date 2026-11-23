"""The committed settings file `learn/drift-eta.json` meets the sealed drift step's unit names.

The dress rehearsal never reaches the sealed route's settings lookup (`drift_run.route_settings`,
then `drift_run.route_steps`): its drift entry passes no `--held-out-truth-run`. This test takes
the committed file, the committed run list's drift entry for the sealed run and the sealed matrix
(names only, no table of any unit) through the same two functions the route calls, with a
stand-in for a unit's days that holds one site named as the unit is, and asserts:

1. the file is of the plan's rule and holds a step size for each tier and site type
   (`drift_run.settings_record`, the function `route_settings` ends in);
2. each unit of the sealed run's drift group is a unit of the sealed matrix and a site with a site
   type, and gets, in each tier, the file's step size of its site type, here the type `other`
   (OSL_T is a Norwegian site), not the type `workplace`'s, made unequal in a copy of the record;
3. a unit whose site is absent from the site-type table is refused (`drift_run.NotAllowed`), and
   a file with no step size for the type `other` could not run (`CouldNotRun`).
"""

import copy
import json
import types
from pathlib import Path

import pytest

from fleetreserve import drift_run, sealed_route
from fleetreserve.learn_features import SITE_TYPES
from fleetreserve.truth_verify import CouldNotRun

ROOT = Path(__file__).resolve().parents[2]
SEALED = "sealed-run"


def _days_of(unit: str, site: str):
    """The part of a unit's days that `drift_run.site_type` reads: its sites."""
    record = types.SimpleNamespace(site=site)
    return types.SimpleNamespace(unit=unit, days=[types.SimpleNamespace(record=record)])


def test_the_committed_settings_file_gives_the_sealed_drift_units_their_site_types_step_sizes():
    # The committed bytes, as `route_settings` reads them (`sealed_route.committed_text`), not the
    # working copy's file: an edit not committed cannot make this test pass or fail.
    text, _ = sealed_route.committed_text(ROOT, "HEAD", drift_run.SETTINGS)
    record = drift_run.settings_record(text, "c" * 40)
    run_list = json.loads((ROOT / "run-lists" / "run-list.json").read_text())
    drifts = [
        e
        for e in run_list["entries"]
        if "fleetreserve.drift_run" in e["command"] and e["only"] in (None, SEALED)
    ]
    assert len(drifts) == 1, [e["label"] for e in drifts]
    units = run_list["units"][SEALED][drifts[0]["units"]]
    matrix = drifts[0]["extra"][SEALED][drifts[0]["extra"][SEALED].index("--matrix") + 1]
    sealed_units = json.loads((ROOT / matrix).read_text())["gate_units"]
    assert units and "OSL_T" in units and set(units) <= set(sealed_units), (units, sealed_units)

    # Made unequal in a copy, so a unit given the other type's step size is seen.
    apart = copy.deepcopy(record)
    for tier, sizes in apart["eta"].items():
        assert set(sizes) == {"other", "workplace"}, (tier, sizes)
        sizes["workplace"] = 12345.0 + len(tier)
    loaded = {unit: _days_of(unit, unit) for unit in units}
    steps = drift_run.route_steps(apart, loaded, units)
    expected = {"site_type": "other", "eta": {t: record["eta"][t]["other"] for t in record["eta"]}}
    assert steps == {unit: expected for unit in units}
    assert all(value < 12345 for unit in units for value in steps[unit]["eta"].values())

    # A unit whose site the site-type table lacks is refused, whatever the file holds.
    nothing = "NO_SUCH_SITE"
    assert nothing not in SITE_TYPES
    with pytest.raises(drift_run.NotAllowed, match="holds sites with no site type"):
        drift_run.route_steps(record, {nothing: _days_of(nothing, nothing)}, (nothing,))

    # The file without the type `other`'s key could not run: the break of the brief, in a copy.
    broken = copy.deepcopy(record)
    for sizes in broken["eta"].values():
        sizes["others"] = sizes.pop("other")
    with pytest.raises(CouldNotRun, match="holds no step size of tier A for the site type other"):
        drift_run.route_steps(broken, loaded, units)
