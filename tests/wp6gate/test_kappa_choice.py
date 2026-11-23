"""The joint table of kappa and the dial (`kappa_choice`, `kappa_summary`) on made-up results of
six dial-choice runs, one per level of plan section 2 (0.05, 0.1, 0.2, 0.3, 0.4, 0.5).

1. Hand case (two site-type units, two dials; tier A mean offers in kWh per counting test day):

       nrel-from-may-2019 (workplaces)       norway-development-pooled (Norway and Korea)
                  none  eps 0.01  eps 0.05               none  eps 0.01  eps 0.05
       kappa 0.05  4.0     4.0       4.0                  0.5     0.5       0.5
       kappa 0.1   6.0     6.0       7.0                  0.4     0.2       0.4
       kappa 0.2   5.0     5.0       5.5                  1.5     1.0       1.5
       kappa 0.3   9.0     3.0       3.5                  2.5     2.0       2.5
       kappa 0.4   3.0     3.0       3.2                  2.0     1.5       2.0
       kappa 0.5   2.0     2.0       2.5                  1.8     1.5       1.8

   The configuration without rows (none) is listed and never chosen (9.0 at kappa 0.3 is not a
   pair of the rule). NREL before May 2019 offers 50.0 everywhere and is no site type's unit; tier
   B offers 20.0 at kappa 0.3, eps 0.01 and is not read by the rule.
   Reading R1 (the pair with the largest tier A offer per site type): workplaces kappa 0.1, eps
   0.05 (7.0), a kappa below 0.3; Norway and Korea kappa 0.3, eps 0.05 (2.5).
   Reading R2 (one kappa: each unit at its own best dial, summed over the two units): kappa 0.05:
   4.0 + 0.5 = 4.5; 0.1: 7.0 + 0.4 = 7.4; 0.2: 5.5 + 1.5 = 7.0; 0.3: 3.5 + 2.5 = 6.0; 0.4:
   3.2 + 2.0 = 5.2; 0.5: 2.5 + 1.8 = 4.3. So kappa
   0.1, then eps 0.05 at both units (7.0; and 0.4 above eps 0.01's 0.2, the 0.4 without rows not
   being a dial). The readings differ at Norway and Korea (R1 kappa 0.3, R2 kappa 0.1).
2. Ties go to the smaller kappa, then the smaller dial.
3. Each pair's figures come from the run at that pair's kappa, whatever the order the runs are
   given in.
4. A pair that is not defined (the unit is not split; the configuration failed) is printed "not
   defined", never 0, and is in no reading; a pair whose splits all abstain offers 0.00 with "-"
   for its losses.
5. Timeouts are counted per pair: certificate timeouts from the run's certificate-timeouts file,
   re-solve timeouts from the configuration's evaluations.
6. The command: exit 0 with its two outputs on six verified runs; exit 2 when a level is missing
   or repeated, when the runs differ in what they measured, when a run has no certificate-timeouts
   file, or for a rotation what-if; exit 1 for a run that did not end with exit 0.
"""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from fleetreserve import dial_choice, dial_summary, kappa_choice, kappa_summary, run_verify

WORK, NORWAY, BEFORE = "nrel-from-may-2019", "norway-development-pooled", "nrel-before-may-2019"
UNITS = (BEFORE, WORK, NORWAY)
DIALS = (0.01, 0.05)
LEVELS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
COMMIT = "c" * 40
# Tier A mean offers of the hand case: unit -> kappa -> (none, eps 0.01, eps 0.05).
HAND = {
    WORK: {
        0.05: (4.0, 4.0, 4.0),
        0.1: (6.0, 6.0, 7.0),
        0.2: (5.0, 5.0, 5.5),
        0.3: (9.0, 3.0, 3.5),
        0.4: (3.0, 3.0, 3.2),
        0.5: (2.0, 2.0, 2.5),
    },
    NORWAY: {
        0.05: (0.5, 0.5, 0.5),
        0.1: (0.4, 0.2, 0.4),
        0.2: (1.5, 1.0, 1.5),
        0.3: (2.5, 2.0, 2.5),
        0.4: (2.0, 1.5, 2.0),
        0.5: (1.8, 1.5, 1.8),
    },
    BEFORE: {k: (50.0, 50.0, 50.0) for k in LEVELS},
}


def _name(kappa, dial) -> str:
    return f"k{kappa:g}-" + ("plain" if dial is None else f"eps{dial:g}")


def _tier(offer, share: float = 1.0, slot: float = 0.02, driver: float = 0.01) -> dict:
    splits = 10
    offering = round(share * splits)
    return {
        "splits": splits,
        "mean_offer_kwh_per_test_day": offer,
        "splits_that_offer": offering,
        "share_of_splits_that_offer": share,
        "mean_test_slot_loss": slot if offering else None,
        "mean_test_driver_loss": driver if offering else None,
    }


def results(kappa: float, offers=HAND, not_split=(), failed=()) -> dict:
    """A made-up dial-choice results document at `kappa`: tier A offers from `offers`, tier B the
    same plus 0.5 (20.0 at kappa 0.3, eps 0.01 of the workplace unit); losses vary by pair so that
    a pair read at the wrong kappa shows. `failed`: (unit, dial) pairs whose configuration
    failed."""
    units = {}
    for unit in UNITS:
        if unit in not_split:
            units[unit] = {"status": "not calibratable: 10 days", "counting_days": 10}
            units[unit]["configurations"] = {}
            continue
        configurations = {}
        for dial, offer in zip((None, *DIALS), offers[unit][kappa], strict=True):
            name = _name(kappa, dial)
            if (unit, dial) in failed:
                configurations[name] = {"driver_risk": dial, "failed": "ValueError: made up"}
                continue
            tier_b = 20.0 if (unit, kappa, dial) == (WORK, 0.3, 0.01) else offer + 0.5
            slot = round(0.01 * kappa + 0.1 * (dial or 0), 6)
            configurations[name] = {
                "driver_risk": dial,
                "offering_days": 40,
                "evaluations": {"re_solves_of_the_one_way_rung": {"optimal": 9, "timeout": 1}},
                "tiers": {
                    "A": _tier(offer, slot=slot, driver=round(slot / 2, 6)),
                    "B": _tier(tier_b, slot=slot),
                },
            }
        units[unit] = {
            "status": "calibratable",
            "counting_days": 60,
            "calibration_days_used": 50,
            "test_days": 10,
            "configurations": configurations,
        }
    return {
        "label": f"dial choice: {dial_summary.LABEL}",
        "model": "lgbm_tail",
        "fit_units": ["acn-caltech"],
        "matrix": {"file": "matrix.json", "sha256": "m" * 64},
        "inputs": {"setting": "main", "truth_run": "truth", "table_sha256": {"x.csv": "t" * 64}},
        "run": {
            "splits": 10,
            "seeds": [1, 10],
            "scales": [0.05, 0.1],
            "kappa": kappa,
            "tiers": ["A", "B"],
            "rung": "one-way",
            "dials": list(DIALS),
        },
        "units": units,
        "timeouts": {"best_causal_ceilings": 0, "certificates": 3},
        "failed_configurations": [],
    }


def timeouts(document: dict) -> dict:
    """The made-up certificate-timeouts file of `document`: 1 per configuration of the workplace
    unit, 0 elsewhere."""
    kappa = document["run"]["kappa"]
    return {
        "kappa": kappa,
        "certificate_timeouts": {
            unit: {name: int(unit == WORK) for name in entry["configurations"]}
            for unit, entry in document["units"].items()
            if entry["status"] == "calibratable"
        },
    }


def runs(order=LEVELS, **made_up) -> list:
    found = []
    for kappa in order:
        document = results(kappa, **made_up)
        source = {"results": "dial-choice.json", "sha256": "s" * 64}
        source |= {"run_folder": f"run-k{kappa:g}", "commit": COMMIT}
        found.append(kappa_choice.Run(document, timeouts(document), source))
    return found


@pytest.fixture(scope="module")
def hand():
    return kappa_choice.joint(runs())


# --- 1. the hand case ----------------------------------------------------------------------------


def test_reading_r1_takes_the_pair_with_the_largest_tier_a_offer_per_site_type(hand):
    r1 = hand["readings"]["R1"]
    assert r1["workplaces"] == {"unit": WORK, "kappa": 0.1, "dial": 0.05, "mean_offer": 7.0}
    assert r1["Norway and Korea"] == {"unit": NORWAY, "kappa": 0.3, "dial": 0.05, "mean_offer": 2.5}


def test_reading_r2_takes_one_kappa_by_the_summed_best_offers(hand):
    r2 = hand["readings"]["R2"]
    sums = {"0.05": 4.5, "0.1": 7.4, "0.2": 7.0, "0.3": 6.0, "0.4": 5.2, "0.5": 4.3}
    assert r2["sums_by_kappa"] == pytest.approx(sums)
    assert r2["kappa"] == 0.1
    assert r2["dials"]["workplaces"] == {"unit": WORK, "dial": 0.05, "mean_offer": 7.0}
    assert r2["dials"]["Norway and Korea"] == {"unit": NORWAY, "dial": 0.05, "mean_offer": 0.4}


def test_the_two_readings_differ_and_neither_is_chosen(hand):
    readings = hand["readings"]
    assert readings["differ_at"] == ["Norway and Korea"]
    assert readings["label"] == kappa_choice.LABEL
    text = "\n".join(kappa_summary.summary_lines(hand))
    assert "The two readings differ at Norway and Korea" in text
    lines = [line for line in text.splitlines() if line.startswith("  workplaces")]
    lines += [line for line in text.splitlines() if line.startswith("  Norway and Korea")]
    assert len(lines) == 4 and all(kappa_choice.LABEL in line for line in lines)
    assert "R1: kappa 0.1, eps 0.05, 7.00 kWh" in lines[0]


def test_the_levels_0_4_and_0_5_are_read_like_the_four_below_them():
    offers = copy.deepcopy(HAND)
    offers[WORK][0.4] = (8.0, 7.5, 7.9)  # above every pair of the hand case
    offers[NORWAY][0.5] = (2.0, 2.0, 3.0)  # above Norway's 2.5 at kappa 0.3
    readings = kappa_choice.joint(runs(offers=offers))["readings"]
    r1, r2 = readings["R1"], readings["R2"]
    assert r1["workplaces"] == {"unit": WORK, "kappa": 0.4, "dial": 0.05, "mean_offer": 7.9}
    assert r1["Norway and Korea"] == {"unit": NORWAY, "kappa": 0.5, "dial": 0.05, "mean_offer": 3.0}
    # kappa 0.4: 7.9 + 2.0 = 9.9; kappa 0.5: 2.5 + 3.0 = 5.5; the other sums are the hand case's
    sums = {"0.05": 4.5, "0.1": 7.4, "0.2": 7.0, "0.3": 6.0, "0.4": 9.9, "0.5": 5.5}
    assert r2["sums_by_kappa"] == pytest.approx(sums)
    assert r2["kappa"] == 0.4
    assert r2["dials"]["Norway and Korea"] == {"unit": NORWAY, "dial": 0.05, "mean_offer": 2.0}
    assert readings["differ_at"] == ["Norway and Korea"]


# --- 2. ties -------------------------------------------------------------------------------------


def test_ties_go_to_the_smaller_kappa_then_the_smaller_dial():
    offers = copy.deepcopy(HAND)
    offers[WORK][0.2] = (5.0, 7.0, 7.0)  # both equal to kappa 0.1, eps 0.05
    offers[NORWAY][0.3] = (2.5, 2.5, 2.5)  # eps 0.01 now equal to eps 0.05
    r1 = kappa_choice.joint(runs(offers=offers))["readings"]["R1"]
    assert (r1["workplaces"]["kappa"], r1["workplaces"]["dial"]) == (0.1, 0.05)
    assert (r1["Norway and Korea"]["kappa"], r1["Norway and Korea"]["dial"]) == (0.3, 0.01)


def test_a_tie_between_kappas_in_r2_goes_to_the_smaller_kappa():
    offers = copy.deepcopy(HAND)
    offers[WORK][0.3] = (9.0, 3.0, 4.9)  # 4.9 + 2.5 = 7.4, equal to kappa 0.1
    found = kappa_choice.joint(runs(offers=offers))["readings"]["R2"]
    assert found["sums_by_kappa"]["0.3"] == pytest.approx(7.4)
    assert found["kappa"] == 0.1


# --- 3. each pair from the run at its kappa ------------------------------------------------------


def test_each_pair_comes_from_the_run_at_its_kappa_in_any_order(hand):
    shuffled = kappa_choice.joint(runs(order=(0.2, 0.5, 0.05, 0.3, 0.4, 0.1)))
    assert shuffled == hand
    for kappa in LEVELS:
        document = results(kappa)
        for unit in (WORK, NORWAY):
            for dial in (None, *DIALS):
                found = document["units"][unit]["configurations"][_name(kappa, dial)]["tiers"]
                for tier in ("A", "B"):
                    cell = hand["units"][unit]["pairs"][tier][f"{kappa:g}"][kappa_choice.key(dial)]
                    assert cell["defined"] is True
                    for field in kappa_choice.CELL_FIELDS:
                        assert cell[field] == found[tier][field], (unit, kappa, dial, tier)
    assert [r["kappa"] for r in hand["runs"]] == list(LEVELS)


def test_the_table_prints_each_pair_under_its_kappa_and_dial(hand):
    lines = kappa_summary.summary_lines(hand)
    start = lines.index(f"{WORK} (60 counting days):")
    block = lines[start : lines.index(" tier B (deadline contract)", start)]
    offer_rows = [line for line in block if " offer " in line]
    assert [row.split() for row in offer_rows] == [
        ["0.05", "offer", "4.00", "4.00", "4.00"],
        ["0.1", "offer", "6.00", "6.00", "7.00"],
        ["0.2", "offer", "5.00", "5.00", "5.50"],
        ["0.3", "offer", "9.00", "3.00", "3.50"],
        ["0.4", "offer", "3.00", "3.00", "3.20"],
        ["0.5", "offer", "2.00", "2.00", "2.50"],
    ]
    slot_rows = [line for line in block if " slot loss " in line]
    assert slot_rows[3].split()[2:] == ["0.003", "0.004", "0.008"]  # 0.01 kappa + 0.1 eps


# --- 4. pairs that are not defined ---------------------------------------------------------------


def test_a_pair_that_is_not_defined_is_printed_so_and_never_as_zero():
    found = kappa_choice.joint(runs(not_split=(BEFORE,), failed=((WORK, 0.05),)))
    cell = found["units"][WORK]["pairs"]["A"]["0.1"]["0.05"]
    assert cell["defined"] is False and "failed" in cell["why"]
    assert found["units"][BEFORE]["pairs"]["A"]["0.3"]["none"]["defined"] is False
    r1 = found["readings"]["R1"]["workplaces"]
    assert (r1["kappa"], r1["dial"], r1["mean_offer"]) == (0.1, 0.01, 6.0)
    lines = kappa_summary.summary_lines(found)
    start = lines.index(f"{WORK} (60 counting days):")
    offer = next(line for line in lines[start:] if line.split()[:2] == ["0.1", "offer"])
    assert offer.split()[2:4] == ["6.00", "6.00"] and offer.endswith("not defined")
    start = lines.index(f"{BEFORE} (10 counting days, not calibratable: 10 days):")
    rows = [line for line in lines[start : start + 60] if " offer " in line][:6]
    assert len(rows) == 6
    assert all(line.count("not defined") == 3 and "0.00" not in line for line in rows)


def test_a_pair_whose_splits_all_abstain_offers_zero_with_no_loss():
    offers = copy.deepcopy(HAND)
    offers[NORWAY][0.05] = (0.0, 0.0, 0.0)
    made_up = runs(offers=offers)
    for entry in made_up[0].document["units"][NORWAY]["configurations"].values():
        entry["tiers"]["A"] = _tier(0.0, share=0.0)
    found = kappa_choice.joint(made_up)
    cell = found["units"][NORWAY]["pairs"]["A"]["0.05"]["0.01"]
    assert cell["defined"] is True and cell["mean_offer_kwh_per_test_day"] == 0.0
    lines = kappa_summary.summary_lines(found)
    start = lines.index(f"{NORWAY} (60 counting days):")
    block = lines[start : start + 8]
    assert next(x for x in block if " offer " in x).split()[2:5] == ["0.00"] * 3
    assert next(x for x in block if " offering " in x).split()[1:4] == ["0%"] * 3
    assert next(x for x in block if " slot loss " in x).split()[2:5] == ["-"] * 3


def test_a_site_type_unit_that_is_not_split_leaves_both_readings_not_defined_there():
    found = kappa_choice.joint(runs(not_split=(NORWAY,)))
    readings = found["readings"]
    assert readings["R1"]["Norway and Korea"] is None
    assert readings["R1"]["workplaces"]["kappa"] == 0.1
    assert readings["R2"]["kappa"] is None
    assert set(readings["R2"]["sums_by_kappa"].values()) == {None}
    text = "\n".join(kappa_summary.summary_lines(found))
    assert "  Norway and Korea (norway-development-pooled), R1: not defined" in text
    assert "  Norway and Korea (norway-development-pooled), R2: not defined" in text


# --- 5. timeouts per pair ------------------------------------------------------------------------


def test_timeouts_are_counted_per_pair(hand):
    cell = hand["units"][WORK]["timeouts"]["0.2"]["0.01"]
    assert cell == {"certificates": 1, "re_solves": 1}
    assert hand["units"][NORWAY]["timeouts"]["0.2"]["none"] == {"certificates": 0, "re_solves": 1}
    lines = kappa_summary.summary_lines(hand)
    start = lines.index(f"{WORK} (60 counting days):")
    timed = next(line for line in lines[start:] if line.split()[:1] == ["0.2"] and "/" in line)
    assert timed.split()[1:] == ["1/1", "1/1", "1/1"]


# --- 6. the command ------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)


def record(folder: Path, document: dict, extra=None, code: int = 0, commit: str = COMMIT) -> Path:
    """`document` and its certificate-timeouts file in the run folder `folder`, with a made-up
    manifest as `run_record` writes it; returns the results file."""
    folder.mkdir(parents=True)
    (folder / dial_choice.RESULTS_NAME).write_text(json.dumps(document))
    if extra is not False:
        found = timeouts(document) if extra is None else extra
        (folder / dial_choice.CERTIFICATE_TIMEOUTS_NAME).write_text(json.dumps(found))
    outputs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}
    manifest = {"commit": commit, "command": ["python", "-m", "fleetreserve.dial_choice"]}
    manifest |= {"start": "s", "end": "e", "exit_code": code, "outputs": outputs}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder / dial_choice.RESULTS_NAME


def _command(tmp_path: Path, monkeypatch, files) -> tuple[int, Path]:
    out = tmp_path / "joint"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    return kappa_choice.main([str(f) for f in files]), out


def _files(tmp_path: Path, change=None, levels=LEVELS, **record_options) -> list[Path]:
    files = []
    for k, kappa in enumerate(levels):
        document = results(kappa)
        if change is not None and k == len(levels) - 1:
            change(document)
        options = record_options if k == len(levels) - 1 else {}
        files.append(record(tmp_path / f"runs/run{k}-k{kappa:g}", document, **options))
    return files


def test_the_command_writes_the_joint_table(tmp_path, monkeypatch, hand, capsys):
    code, out = _command(tmp_path, monkeypatch, _files(tmp_path))
    assert code == 0
    document = json.loads((out / kappa_choice.RESULTS_NAME).read_text())
    assert document["readings"] == json.loads(json.dumps(hand["readings"]))
    assert [r["run_folder"] for r in document["runs"]] == [
        f"run{k}-k{x:g}" for k, x in enumerate(LEVELS)
    ]
    assert all(r["commit"] == COMMIT for r in document["runs"])
    text = (out / kappa_choice.SUMMARY_NAME).read_text()
    assert text == capsys.readouterr().out
    assert kappa_choice.LABEL in text


@pytest.mark.parametrize(
    "levels",
    [
        (0.05, 0.1, 0.2, 0.3),
        (0.05, 0.1, 0.2, 0.3, 0.4),
        (0.05, 0.1, 0.2, 0.3, 0.4, 0.4),
        (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5),
    ],
)
def test_a_missing_or_repeated_level_could_not_run(tmp_path, monkeypatch, capsys, levels):
    code, _ = _command(tmp_path, monkeypatch, _files(tmp_path, levels=levels))
    assert code == 2
    assert "one run at each level" in capsys.readouterr().out


def _set(path, value):
    def change(document):
        place = document
        for key in path[:-1]:
            place = place[key]
        place[path[-1]] = value

    return change


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("model",), "lgbm"),
        (("matrix", "sha256"), "n" * 64),
        (("inputs", "truth_run"), "other"),
        (("run", "splits"), 9),
        (("run", "dials"), [0.01]),
        (("units", WORK, "counting_days"), 59),
    ],
)
def test_runs_that_differ_could_not_run(tmp_path, monkeypatch, capsys, path, value):
    code, _ = _command(tmp_path, monkeypatch, _files(tmp_path, change=_set(path, value)))
    assert code == 2
    assert "the runs differ in" in capsys.readouterr().out


def test_runs_at_two_commits_could_not_run(tmp_path, monkeypatch, capsys):
    code, _ = _command(tmp_path, monkeypatch, _files(tmp_path, commit="d" * 40))
    assert code == 2
    assert "the runs differ in commit" in capsys.readouterr().out


def test_a_run_without_its_certificate_timeouts_could_not_run(tmp_path, monkeypatch, capsys):
    code, _ = _command(tmp_path, monkeypatch, _files(tmp_path, extra=False))
    assert code == 2
    assert dial_choice.CERTIFICATE_TIMEOUTS_NAME in capsys.readouterr().out


def test_a_rotation_whatif_could_not_run(tmp_path, monkeypatch, capsys):
    change = _set(("label",), "a what-if. dial choice: " + dial_summary.LABEL)
    code, _ = _command(tmp_path, monkeypatch, _files(tmp_path, change=change))
    assert code == 2
    assert "not a results file of dial_choice" in capsys.readouterr().out


def test_a_run_that_did_not_end_with_exit_0_is_refused(tmp_path, monkeypatch, capsys):
    code, _ = _command(tmp_path, monkeypatch, _files(tmp_path, code=1))
    assert code == 1
    assert "exit code 1" in capsys.readouterr().out
