"""`tools/wp15_check.py`: a held-out unit the held-out truth build abstained is "reported, with the
reason" (reading V6 (3), decision 99 (6) of 22 November 2026); no text says "None" or "no
configuration scored"; a configuration recorded as failed is never accepted (reading V6, decision
99 (8)): the part fails, naming the unit, the configuration and the error.

The drives are those of `test_wp15_check`. The sealed truth build's stand-in copies the file
`data/made-up/sealed-truth/truth-summary.json`, in the real summary's form (`units`, `abstained`).
On the held-out route `learn_run` writes a failed configuration as `failed: <error>` under the
configuration and unit (its rungs hold that one reason) and lists it in `failed_configurations`
as `<configuration> | <unit>: <error>`.
"""

import pytest
import test_wp15_check as base
from test_wp15_check import isolated_git  # noqa: F401  (autouse fixture)

check = base.check
UNIT = "korea-work"
REASON = "no day to build a window from"
ABSTAINED = f"{UNIT}: no figure: the held-out truth build gave it no window ({REASON})"
ERROR = "RuntimeError: made to fail"
CONFIG = base.CONFIGURATIONS[1]


def abstain(root, unit: str, reason: str = REASON) -> None:
    base.put(
        root,
        "sealed-truth",
        "truth-summary.json",
        {"units": {}, "abstained": {unit: {"reason": reason}}},
    )


def learn_without(root, rho, unit: str) -> None:
    """The learn run at `rho` (None: the main matrix) as `learn_run` writes it when the held-out
    build abstained `unit`: the unit is left out of `units`."""
    document = base.learn_json(
        [u for u in base.nine() if u != unit], (root / base.MATRICES[rho]).read_bytes()
    )
    base.put(root, check.learn_label(rho), "learn.json", document)


def no_none(lines: list[str]) -> None:
    assert not [x for x in lines if "None" in x or "no configuration scored" in x], lines


# --- reported, with the reason: an abstained unit -------------------------------------------------


def test_an_abstained_unit_is_reported_with_the_reason_in_part_a(tmp_path, monkeypatch, capsys):
    root = base.made(tmp_path, monkeypatch)
    abstain(root, UNIT)
    learn_without(root, 0.1, UNIT)
    base.put(
        root,
        check.intervals_label(0.1),
        "day-intervals.json",
        base.intervals_json([u for u in base.nine() if u != UNIT]),
    )
    base.drive(0)
    status, lines = base.run(root, capsys)
    assert status == 0, lines
    (rho,) = [x for x in lines if x.startswith("  rho 0.1:")]
    assert "16 configuration-unit cells of 9 units" in rho and rho.endswith(ABSTAINED), rho
    assert f"(a) rho 0.1 and 0.2: reported, with the reason: rho 0.1: {ABSTAINED}" in lines, lines
    no_none(lines)


def test_an_abstained_osl_t_is_reported_with_the_reason_in_part_b(tmp_path, monkeypatch, capsys):
    """The drift run holds no unit (OSL_T was abstained by the build): the part is reported, with
    the reason, and says that the only chronological check is lost."""
    root = base.made(tmp_path, monkeypatch)
    abstain(root, check.DRIFT_UNIT)
    document, table = base.drift_offering_nothing(record=False)
    document["units"] = {}
    document["measure_not_made"] = {check.DRIFT_UNIT: f"abstained ({REASON})"}
    base.put(root, check.DRIFT_LABEL, base.drift_run.RESULTS_NAME, document)
    base.put(root, check.DRIFT_LABEL, base.drift_run.DAYS_NAME, table)
    base.drive(0)
    status, lines = base.run(root, capsys)
    assert status == 0, lines
    reason = (
        f"{check.DRIFT_UNIT}: no figure: the held-out truth build gave it no window ({REASON}); "
        "the only chronological check is lost"
    )
    assert f"(b) chronological calibration at OSL_T: reported, with the reason: {reason}" in lines
    no_none(lines)


def test_an_abstained_asr_unit_is_reported_with_the_reason_in_part_c(tmp_path, monkeypatch, capsys):
    root = base.made(tmp_path, monkeypatch)
    unit = check.ASR_UNITS[1]
    abstain(root, unit)
    learn_without(root, None, unit)
    base.drive(0)
    status, lines = base.run(root, capsys)
    assert status == 0, lines
    shown = f"{unit}: no figure: the held-out truth build gave it no window ({REASON})"
    assert f"(c) regimes: reported, with the reason: {shown}" in lines, lines
    no_none(lines)


# --- not abstained: the real cause, never "None" -----------------------------------------------


def test_a_unit_missing_without_an_abstention_fails_with_its_cause(tmp_path, monkeypatch, capsys):
    root = base.made(tmp_path, monkeypatch)
    learn_without(root, 0.1, UNIT)  # the build's summary records no abstention
    base.drive(0)
    status, lines = base.run(root, capsys)
    assert status == 1, lines
    assert any(
        x.startswith("FAILED: (a) rho 0.1 and 0.2: learn-lgbm_tail-rho0_1 lacks 1 figure: ")
        and f"{UNIT}: no figure: the learn run has no entry for it, and the held-out truth build "
        "recorded no abstention for it"
        in x
        for x in lines
    ), lines
    no_none(lines)


def test_a_unit_with_no_configuration_fails_with_its_cause(tmp_path, monkeypatch, capsys):
    root = base.made(tmp_path, monkeypatch)
    document = base.learn_json(base.nine(), (root / base.MATRICES[0.1]).read_bytes())
    document["units"][UNIT]["configurations"] = {}
    base.put(root, check.learn_label(0.1), "learn.json", document)
    base.drive(0)
    status, lines = base.run(root, capsys)
    assert status == 1, lines
    assert any(
        f"{UNIT}: no figure: its entry in the learn run holds no configuration, and the held-out "
        "truth build recorded no abstention for it" in x
        for x in lines
    ), lines
    no_none(lines)


def test_a_unit_the_build_wrote_tables_for_is_not_abstained(tmp_path, monkeypatch, capsys):
    """The summary lists the unit under `abstained` but also holds its tables: the build did not
    leave it out, so it is not the case of reading V6 (3)."""
    root = base.made(tmp_path, monkeypatch)
    base.put(
        root,
        "sealed-truth",
        "truth-summary.json",
        {"units": {UNIT: {}}, "abstained": {UNIT: {"reason": REASON}}},
    )
    learn_without(root, 0.1, UNIT)
    base.drive(0)
    status, lines = base.run(root, capsys)
    assert status == 1, lines
    assert any("recorded no abstention for it" in x for x in lines), lines


@pytest.mark.parametrize(
    "units, entry, said",
    [
        (
            {},
            None,
            "OSL_T is not in the drift run's results, and the held-out truth build recorded no "
            "abstention for it",
        ),
        (
            {},
            ["OSL_T: ValueError: made to fail"],
            "OSL_T did not run, the drift run lists it as failed: "
            "['OSL_T: ValueError: made to fail']",
        ),
        (
            {check.DRIFT_UNIT: {"status": "failed"}},
            [],
            "OSL_T has the status 'failed', not 'run'",
        ),
    ],
    ids=["not-in-results", "listed-as-failed", "other-status"],
)
def test_a_drift_run_without_osl_t_says_why_never_none(
    tmp_path, monkeypatch, capsys, units, entry, said
):
    root = base.made(tmp_path, monkeypatch)
    document, table = base.drift_files()
    document["units"] = units
    document["failed_units"] = entry if entry is not None else document["failed_units"]
    base.put(root, check.DRIFT_LABEL, base.drift_run.RESULTS_NAME, document)
    base.put(root, check.DRIFT_LABEL, base.drift_run.DAYS_NAME, table)
    base.drive(0)
    status, lines = base.run(root, capsys)
    assert status == 1, lines
    assert any(
        x.startswith("FAILED: (b) chronological calibration at OSL_T: drift-lgbm_tail: ")
        and x.endswith(said)
        for x in lines
    ), lines
    no_none(lines)


# --- a configuration recorded as failed --------------------------------------------------------


def failed_in(document: dict, unit: str, *, cell=True, listed=True, split=False) -> None:
    if cell:
        document["units"][unit]["configurations"][CONFIG]["rungs"] = {"reason": f"failed: {ERROR}"}
    if listed:
        document["failed_configurations"] = [f"{CONFIG} | {unit}: {ERROR}"]
    if split:
        document["units"][unit]["status"] = base.NOT_SPLIT
        document["units"][unit]["test_days"] = 0


@pytest.mark.parametrize(
    "shape",
    [
        {"cell": True, "listed": True},
        {"cell": True, "listed": False},
        {"cell": False, "listed": True},
        {"cell": True, "listed": False, "split": True},
    ],
    ids=["cell-and-list", "cell-only", "list-only", "unit-not-split"],
)
def test_a_failed_configuration_fails_part_a_naming_unit_configuration_and_error(
    tmp_path, monkeypatch, capsys, shape
):
    root = base.made(tmp_path, monkeypatch)
    document = base.learn_json(base.nine(), (root / base.MATRICES[0.1]).read_bytes())
    failed_in(document, "TRO_R", **shape)
    base.put(root, check.learn_label(0.1), "learn.json", document)
    base.drive(0)
    status, lines = base.run(root, capsys)
    assert status == 1, lines
    assert (
        "FAILED: (a) rho 0.1 and 0.2: learn-lgbm_tail-rho0_1 records 1 configuration as failed, "
        f"so no unit of it is reported: configuration {CONFIG} on unit TRO_R: {ERROR}"
    ) in lines, lines
    assert not any("with the reason" in x for x in lines), lines
    assert lines[-1].startswith("FAILED: WP15, 2 of 3 parts reported"), lines
    no_none(lines)


@pytest.mark.parametrize("cell", [True, False], ids=["cell", "list"])
def test_a_failed_configuration_fails_part_c_naming_unit_configuration_and_error(
    tmp_path, monkeypatch, capsys, cell
):
    root = base.made(tmp_path, monkeypatch)
    document = base.learn_json(base.nine(), (root / base.MAIN).read_bytes())
    failed_in(document, check.ASR_UNITS[0], cell=cell)
    base.put(root, check.learn_label(), "learn.json", document)
    base.drive(0)
    status, lines = base.run(root, capsys)
    assert status == 1, lines
    assert (
        "FAILED: (c) regimes: learn-lgbm_tail records 1 configuration as failed, so no unit of it "
        f"is reported: configuration {CONFIG} on unit {check.ASR_UNITS[0]}: {ERROR}"
    ) in lines, lines


def test_a_failed_configuration_of_a_unit_part_c_does_not_read_is_not_part_cs(
    tmp_path, monkeypatch, capsys
):
    """Part (c) reads the two a.s.r. units of the main model's learn run, no other: the part is
    reported, but WP14's mark 1 names the failed configuration of any unit, so WP15 exits 1."""
    root = base.made(tmp_path, monkeypatch)
    document = base.learn_json(base.nine(), (root / base.MAIN).read_bytes())
    failed_in(document, "TRO_R")
    base.put(root, check.learn_label(), "learn.json", document)
    base.drive(0)
    status, lines = base.run(root, capsys)
    assert status == 1, lines
    assert "(c) regimes: reported" in lines, lines
    assert any(
        x.startswith("FAILED: WP14: learn step learn-lgbm_tail: configuration ")
        and f"failed on unit TRO_R: {ERROR}" in x
        for x in lines
    ), lines
