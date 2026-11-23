"""The settings writer of the drift guard's step size (`drift_settings`; plan section 6, T10).

1. A results file of the time-ordered run that holds a unit other than a training unit (a
   development unit beside two training units) is refused: exit 1, "REFUSED", the unit named,
   nothing written. The same file with the training units alone is written, from the same
   builder of the results (`drift_run.results`) the command uses: the file names the rule, the
   start reading, a step size per tier and the results file's hash, run folder and commit.
2. A file that is not verified against its run's manifest is refused (exit 1); a missing one
   could not run (exit 2); an unknown rule could not run (exit 2).
3. Each rule on hand-made figures (per step size: whether every unit's final mean loss is within
   alpha, the offer summed over units, the share of offering days the guard abstains):
   step sizes 0.1, 0.5 and 1 with offers 3.0, 5.0 and 4.0 kWh, within alpha at 0.1 and 1 only,
   abstaining shares 0.30, 0.10 and 0.20. "largest-offer-within-alpha" gives 1 (4.0 is the
   largest offer of the two within alpha); "largest-offer" gives 0.5 (5.0); "fewest-abstaining-
   within-alpha" gives 1 (0.20 < 0.30). With no step size within alpha the two rules that need
   one find none; equal offers go to the smaller step size.
4. The grouping is named on the command line, with no default: without `--grouping` the command
   could not run (exit 2) and writes nothing; an unknown grouping is an error. Items 1 and 2 name
   `all-units`, the one value over all units.
5. Per site type, worked by hand (`HAND`): two site types, three step sizes 0.1, 0.5 and 1,
   start zero. Workplaces: caltech offers 1.0, 2.0, 3.0 kWh per counting day and jpl 0.5, 1.5,
   2.5, every guarded loss within alpha except jpl's driver loss at step size 1 (tier A only:
   tier B does not hold the driver loss down); office001 has too few days and is left out. The
   other type: OSL_S offers 2.0, 1.0, 0.5 and ASK 1.0, 1.0, 0.5, every loss within alpha.
   Summed per type, workplaces 1.5, 3.5, 5.5 and the other type 3.0, 2.0, 1.0. By
   "largest-offer-within-alpha": tier A workplaces 0.5 (5.5 at 1 is not within alpha on jpl),
   the other type 0.1; tier B workplaces 1, the other type 0.1. One value over all units
   (summed 4.5, 5.5, 6.5): tier A 0.5, tier B 1, so the other type's 0.1 is not the one value.
   The file holds per tier one step size per site type, the units each was read from and the
   units left out for too few days, and the figures per type.
6. Per site type, a type with no unit that has a guarded step (no unit of the type in the run;
   its only unit with too few days; its only unit with no offering guarded day), a type where
   the rule finds no step size (OSL_S outside alpha at every step size in tier A), or a unit
   with no site type could not run (exit 2), naming the type or unit; nothing is written.
7. `step_size` reads both forms: the one value per tier for every site type, or the step size
   of the scored unit's site type; a file of another grouping is an error.
8. The plan's rule (`largest-below-first-failure`, reading E25) on hand-made figures, step sizes
   0.1, 0.5 and 1: the smallest fails, "offers nothing" (a value of its own, not None, not a
   step size); 0.5 fails and 1 passes again with the largest offer, 0.1 (E10 gives 1); none
   fails, 1 whatever the offers; one step size, itself when it passes and "offers nothing" when
   it fails; no figures, none.
9. The plan's rule on the hand case of item 5: workplaces tier A 0.5 (jpl fails at 1), every
   other tier and type 1, the largest tried; office001 named as left out and not judged. One
   failing unit (jpl at 0.5) fails the step size while caltech passes: 0.1. A unit absent at one
   step size (caltech with no summary, or no guarded step, at 0.5; reading E27) fails it: 0.1,
   where skipping it would give 0.5. A step size where no unit has a guarded step (0.5 in tier
   B; reading E26) is kept and fails: 0.1, not 1. A unit whose guarded days never offer (ASK;
   reading E28) is not judged: the type other takes OSL_S's 1. OSL_S failing at every step size
   in tier A: the file records "offers nothing" for tier A of the type other (and over all
   units for tier A), and `step_size` returns it. Last, what readings E26 and E27 rest on: a
   unit's offering days are the same at every step size.
"""

import hashlib
import json

import pytest
from gate_fake import matrix_document

from fleetreserve import drift_path, drift_run, drift_settings, learn_features, run_verify
from fleetreserve.gate_config import parse
from fleetreserve.learn_features import OTHER_SITE, WORKPLACE

COMMIT = "b" * 40
ETAS = (0.1, 1.0)
N_CAL = 20
# The rules of readings E10 to E12, which compare offers or abstaining shares, so a tie decides
# between step sizes; the rule of reading E25 compares neither.
BY_OFFER_OR_SHARE = (
    "largest-offer-within-alpha",
    "largest-offer",
    "fewest-abstaining-within-alpha",
)
FIRST = drift_settings.FIRST_FAILURE
NOTHING = drift_settings.OFFERS_NOTHING


def _unit(seed: int):
    n = N_CAL + 40
    keys = [f"2020-{1 + d // 28:02d}-{1 + d % 28:02d} S{seed}" for d in range(n)]
    offering = [(d + seed) % 7 != 0 for d in range(n)]
    values = [4.0 + seed if o else 0.0 for o in offering]

    def evaluate(pairs):
        return {(d, s): ((1.0, 0.5) if s > 0.4 + 0.05 * (d % 5) else (0.0, 0.0)) for d, s in pairs}

    return drift_path.run_unit(keys, values, offering, evaluate, n_calibration=N_CAL, etas=ETAS), n


def _document(units):
    names = ("main-rounded-hour-k0.3-eps0.02-empirical", "main-rounded-hour-k0.3-eps0.02-oracle")
    matrix = parse(matrix_document(names))
    configuration = matrix.configuration(names[0])
    entries = {}
    for seed, unit in enumerate(units):
        found, n = _unit(seed)
        entries[unit] = drift_run.unit_entry(unit, found, n, fit_units=("x",), rotation=None)
    return drift_run.results(
        "lgbm_tail_rotation", configuration, units, ETAS, N_CAL, entries, failed={}
    )


def _recorded(folder, document):
    """`document` as the results file of a made-up recorded run, with its manifest."""
    run = folder / "20260101T000000Z-wp9-drift-run"
    run.mkdir()
    results = run / drift_run.RESULTS_NAME
    results.write_text(json.dumps(document))
    manifest = {
        "commit": COMMIT,
        "command": ["drift"],
        "start": "s",
        "end": "e",
        "exit_code": 0,
        "outputs": {results.name: hashlib.sha256(results.read_bytes()).hexdigest()},
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    return results


@pytest.fixture(autouse=True)
def commits_exist(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)


def _write(results, out, rule="largest-offer-within-alpha", start="zero", grouping="all-units"):
    return drift_settings.main(
        [str(results), "--rule", rule, "--start", start, "--grouping", grouping, "--out", str(out)]
    )


def test_a_results_file_with_a_unit_that_is_not_training_is_refused(tmp_path, capsys):
    results = _recorded(tmp_path, _document(("caltech", "nrel-from-may-2019", "jpl")))
    out = tmp_path / "drift-eta.json"
    assert _write(results, out) == 1
    printed = capsys.readouterr().out
    assert printed.startswith("REFUSED") and "nrel-from-may-2019" in printed
    assert not out.exists()


@pytest.mark.parametrize("rule", sorted(drift_settings.RULES))
def test_a_results_file_of_training_units_is_written(tmp_path, rule):
    results = _recorded(tmp_path, _document(("caltech", "jpl")))
    out = tmp_path / "drift-eta.json"
    assert _write(results, out, rule=rule) == 0
    written = json.loads(out.read_text())
    assert written["rule"] == rule and written["start"] == "zero"
    assert set(written["eta"]) == {"A", "B"}
    assert all(written["eta"][tier] in ETAS for tier in ("A", "B"))
    assert written["results"] == {
        "file": drift_run.RESULTS_NAME,
        "sha256": hashlib.sha256(results.read_bytes()).hexdigest(),
        "run_folder": results.parent.name,
        "commit": COMMIT,
    }


def test_an_unverified_missing_or_unknown_rule_is_not_written(tmp_path):
    results = _recorded(tmp_path, _document(("caltech", "jpl")))
    results.write_text(results.read_text() + " ")  # changed after the run
    out = tmp_path / "drift-eta.json"
    assert _write(results, out) == 1
    assert _write(tmp_path / "absent.json", out) == 2
    with pytest.raises(SystemExit) as stop:
        _write(results, out, rule="most-offer")
    assert stop.value.code == 2
    assert not out.exists()


# --- 3. the rules on hand-made figures -----------------------------------------------------------


def _figures(within=(True, False, True), offers=(3.0, 5.0, 4.0), shares=(0.3, 0.1, 0.2)):
    return {
        eta: drift_settings.Figures(w, o, s)
        for eta, w, o, s in zip((0.1, 0.5, 1.0), within, offers, shares, strict=True)
    }


def test_the_rules_are_the_three_by_offer_or_share_and_the_plans():
    assert set(drift_settings.RULES) == {*BY_OFFER_OR_SHARE, FIRST}
    assert set(drift_settings.RULE_TEXTS) == set(drift_settings.RULES)


def test_each_rule_picks_its_hand_computed_step_size():
    figures = _figures()
    assert drift_settings.RULES["largest-offer-within-alpha"](figures) == 1.0
    assert drift_settings.RULES["largest-offer"](figures) == 0.5
    assert drift_settings.RULES["fewest-abstaining-within-alpha"](figures) == 1.0


def test_no_step_size_within_alpha_gives_none_and_ties_go_to_the_smaller():
    none_within = _figures(within=(False, False, False))
    assert drift_settings.RULES["largest-offer-within-alpha"](none_within) is None
    assert drift_settings.RULES["fewest-abstaining-within-alpha"](none_within) is None
    assert drift_settings.RULES["largest-offer"](none_within) == 0.5
    tied = _figures(within=(True, True, True), offers=(4.0, 4.0, 4.0), shares=(0.2, 0.2, 0.2))
    for rule in BY_OFFER_OR_SHARE:
        assert drift_settings.RULES[rule](tied) == 0.1


# --- 4. the grouping is named --------------------------------------------------------------------


def test_the_grouping_must_be_named(tmp_path):
    results = _recorded(tmp_path, _document(("caltech", "jpl")))
    out = tmp_path / "drift-eta.json"
    with pytest.raises(SystemExit) as stop:
        drift_settings.main(
            [str(results), "--rule", "largest-offer", "--start", "zero", "--out", str(out)]
        )
    assert stop.value.code == 2
    assert not out.exists()
    with pytest.raises(ValueError, match="grouping"):
        drift_settings.eta_document(results, "largest-offer", "zero", "by-unit")


# --- 5 to 7. per site type, the hand case --------------------------------------------------------

HAND_ETAS = (0.1, 0.5, 1.0)
HAND_CALIBRATION = 50
WORKPLACES = ("caltech", "jpl", "office001")
OTHERS = ("OSL_S", "ASK")
# unit -> per step size of HAND_ETAS: (offer in kWh per counting day, driver loss within alpha);
# every slot loss is within alpha. A unit not listed has too few days.
HAND = {
    "caltech": ((1.0, True), (2.0, True), (3.0, True)),
    "jpl": ((0.5, True), (1.5, True), (2.5, False)),
    "OSL_S": ((2.0, True), (1.0, True), (0.5, True)),
    "ASK": ((1.0, True), (1.0, True), (0.5, True)),
}
TOO_FEW = drift_path.too_few(30, HAND_CALIBRATION)


def _hand_summary(tier, eta, offer, driver_within, offering_days):
    """The fields of `drift_path.Path.summary` that `drift_settings` reads, by hand; with no
    offering day, no step: offer 0 and no final mean."""
    within = {"slot": True, "driver": driver_within}
    losses = {}
    for loss in drift_path.LOSSES:
        losses[loss] = {"guarded": loss in drift_path.guards(tier)}
        if losses[loss]["guarded"]:
            losses[loss]["within_alpha"] = within[loss] if offering_days else None
    return {
        "tier": tier,
        "start": "zero",
        "eta": eta,
        "offering_days": offering_days,
        "guard_abstaining_days": 1 if offering_days else 0,
        "mean_offer_kwh_per_counting_day": offer if offering_days else 0.0,
        "losses": losses,
    }


def _hand_entry(unit, hand, offering_days):
    if unit not in hand:
        keys = [f"2020-01-{d + 1:02d} {unit}" for d in range(30)]
        found = drift_path.run_unit(
            keys, [1.0] * 30, [True] * 30, None, n_calibration=HAND_CALIBRATION, etas=HAND_ETAS
        )
        return drift_run.unit_entry(unit, found, 30, fit_units=("x",), rotation=None)
    paths = {
        tier: {
            "zero": {
                drift_run.eta_key(eta): _hand_summary(tier, eta, offer, ok, offering_days)
                for eta, (offer, ok) in zip(HAND_ETAS, hand[unit], strict=True)
            }
        }
        for tier in drift_path.TIERS
    }
    return {"role": "training", "status": "run", "calibration": {}, "paths": paths}


def _hand_document(units=WORKPLACES + OTHERS, hand=HAND, no_steps=()):
    names = ("main-rounded-hour-k0.3-eps0.02-empirical", "main-rounded-hour-k0.3-eps0.02-oracle")
    configuration = parse(matrix_document(names)).configuration(names[0])
    entries = {u: _hand_entry(u, hand, 0 if u in no_steps else 10) for u in units}
    return drift_run.results(
        "lgbm_tail_rotation", configuration, units, HAND_ETAS, HAND_CALIBRATION, entries, {}
    )


def _column(table, field):
    return [table[drift_run.eta_key(eta)][field] for eta in HAND_ETAS]


def test_per_site_type_the_hand_case_gives_each_type_its_own_step_size(tmp_path):
    results = _recorded(tmp_path, _hand_document())
    out = tmp_path / "drift-eta.json"
    assert _write(results, out, grouping="site-type") == 0
    written = json.loads(out.read_text())
    assert written["grouping"] == "site-type"
    assert written["rule"] == "largest-offer-within-alpha" and written["start"] == "zero"
    assert written["eta"] == {
        "A": {"workplace": 0.5, "other": 0.1},
        "B": {"workplace": 1.0, "other": 0.1},
    }
    assert written["site_types"] == {
        "workplace": {"units": ["caltech", "jpl"], "units_left_out": {"office001": TOO_FEW}},
        "other": {"units": ["OSL_S", "ASK"], "units_left_out": {}},
    }
    offer = "offer_kwh_per_counting_day_summed_over_units"
    within = "within_alpha_on_every_unit"
    figures = written["figures"]
    assert _column(figures["A"]["workplace"], offer) == [1.5, 3.5, 5.5]
    assert _column(figures["A"]["workplace"], within) == [True, True, False]
    assert _column(figures["B"]["workplace"], within) == [True, True, True]
    for tier in drift_path.TIERS:
        assert _column(figures[tier]["other"], offer) == [3.0, 2.0, 1.0]
        assert _column(figures[tier]["other"], within) == [True, True, True]
    assert written["results"]["sha256"] == hashlib.sha256(results.read_bytes()).hexdigest()
    one = tmp_path / "one.json"
    assert _write(results, one, grouping="all-units") == 0
    assert json.loads(one.read_text())["eta"] == {"A": 0.5, "B": 1.0}


@pytest.mark.parametrize(
    ("units", "no_steps"),
    [
        (WORKPLACES, ()),  # no unit of the other type in the run
        (WORKPLACES + ("TRO",), ()),  # its only unit has too few days
        (WORKPLACES + ("OSL_S",), ("OSL_S",)),  # its only unit has no offering guarded day
    ],
)
def test_a_site_type_without_a_guarded_step_could_not_run(tmp_path, capsys, units, no_steps):
    results = _recorded(tmp_path, _hand_document(units, no_steps=no_steps))
    out = tmp_path / "drift-eta.json"
    assert _write(results, out, grouping="site-type") == 2
    printed = capsys.readouterr().out
    assert printed.startswith("COULD NOT RUN") and "site type other" in printed
    assert "no unit that has a guarded step" in printed
    assert not out.exists()


def test_a_site_type_where_the_rule_finds_no_step_size_could_not_run(tmp_path, capsys):
    hand = {**HAND, "OSL_S": ((2.0, False), (1.0, False), (0.5, False))}
    results = _recorded(tmp_path, _hand_document(hand=hand))
    out = tmp_path / "drift-eta.json"
    assert _write(results, out, grouping="site-type") == 2
    printed = capsys.readouterr().out
    assert printed.startswith("COULD NOT RUN") and "site type other in tier A" in printed
    assert not out.exists()
    # the rule without a loss condition finds one: workplaces 1 (5.5), the other type 0.1 (3.0)
    assert _write(results, out, rule="largest-offer", grouping="site-type") == 0
    assert json.loads(out.read_text())["eta"]["A"] == {"workplace": 1.0, "other": 0.1}


def test_a_unit_without_a_site_type_could_not_run(tmp_path, capsys, monkeypatch):
    known = {s: t for s, t in learn_features.SITE_TYPES.items() if s != "jpl"}
    monkeypatch.setattr(drift_settings, "SITE_TYPES", known)
    results = _recorded(tmp_path, _hand_document())
    out = tmp_path / "drift-eta.json"
    assert _write(results, out, grouping="site-type") == 2
    printed = capsys.readouterr().out
    assert printed.startswith("COULD NOT RUN") and "jpl" in printed
    assert not out.exists()


def test_step_size_reads_both_forms(tmp_path):
    results = _recorded(tmp_path, _hand_document())
    one_file, by_type_file = tmp_path / "one.json", tmp_path / "by-type.json"
    assert _write(results, one_file, grouping="all-units") == 0
    assert _write(results, by_type_file, grouping="site-type") == 0
    one, by_type = json.loads(one_file.read_text()), json.loads(by_type_file.read_text())
    for kind in (WORKPLACE, OTHER_SITE):
        assert drift_settings.step_size(one, "A", kind) == 0.5
        assert drift_settings.step_size(one, "B", kind) == 1.0
    assert drift_settings.step_size(by_type, "A", WORKPLACE) == 0.5
    assert drift_settings.step_size(by_type, "A", OTHER_SITE) == 0.1
    assert drift_settings.step_size(by_type, "B", WORKPLACE) == 1.0
    assert drift_settings.step_size(by_type, "B", OTHER_SITE) == 0.1
    with pytest.raises(ValueError, match="unknown grouping"):
        drift_settings.step_size({**by_type, "grouping": "by-unit"}, "A", WORKPLACE)


# --- 8. the plan's rule (reading E25) on hand-made figures ---------------------------------------


def _first(within, offers=(3.0, 5.0, 4.0)):
    return drift_settings.RULES[FIRST](_figures(within=within, offers=offers))


def test_the_plans_rule_when_the_smallest_fails_offers_nothing():
    for within in ((False, True, True), (False, False, False), (False, True, False)):
        found = _first(within)
        assert found == NOTHING == "offers nothing"
        assert found is not None and not isinstance(found, float)


def test_the_plans_rule_stops_at_the_first_failure_though_a_later_one_passes():
    # 0.5 fails and 1 passes again with the largest offer: the rule gives 0.1, the step before
    # the first failure; the rule of 16 November 2026 (E10) gives 1
    figures = _figures(within=(True, False, True), offers=(1.0, 2.0, 9.0))
    assert drift_settings.RULES[FIRST](figures) == 0.1
    assert drift_settings.RULES["largest-offer-within-alpha"](figures) == 1.0
    assert _first((True, True, False)) == 0.5


def test_the_plans_rule_when_none_fails_gives_the_largest_tried_whatever_the_offer():
    assert _first((True, True, True), offers=(9.0, 1.0, 0.5)) == 1.0
    assert _first((True, True, True), offers=(0.5, 1.0, 9.0)) == 1.0


def test_the_plans_rule_on_one_step_size_and_on_none():
    one = {0.5: drift_settings.Figures(True, 2.0, 0.1)}
    assert drift_settings.RULES[FIRST](one) == 0.5
    one = {0.5: drift_settings.Figures(False, 2.0, 0.1)}
    assert drift_settings.RULES[FIRST](one) == NOTHING
    assert drift_settings.RULES[FIRST]({}) is None


# --- 9. the plan's rule on the hand case (readings E25 to E28) -----------------------------------


def _summaries(document, unit, tier="A"):
    return document["units"][unit]["paths"][tier]["zero"]


def _write_first(tmp_path, document, grouping="site-type", name="drift-eta.json"):
    results_folder = tmp_path / name.replace(".json", "")
    results_folder.mkdir()
    results = _recorded(results_folder, document)
    out = tmp_path / name
    status = _write(results, out, rule=FIRST, grouping=grouping)
    return status, (json.loads(out.read_text()) if out.exists() else None)


def test_per_site_type_the_plans_rule_on_the_hand_case(tmp_path, capsys):
    """`HAND`: workplaces fail at 1 in tier A (jpl's driver loss), so 0.5; every other step size
    and type passes, so 1, the largest tried, though the type other offers least there (E10 gave
    it 0.1). office001, left out with too few days, is named and not judged."""
    status, written = _write_first(tmp_path, _hand_document())
    assert status == 0
    assert written["rule"] == FIRST and written["rule_text"] == drift_settings.RULE_TEXTS[FIRST]
    assert written["eta"] == {
        "A": {"workplace": 0.5, "other": 1.0},
        "B": {"workplace": 1.0, "other": 1.0},
    }
    assert written["site_types"]["workplace"]["units_left_out"] == {"office001": TOO_FEW}
    judged = {"workplace": ["caltech", "jpl"], "other": ["OSL_S", "ASK"]}
    assert written["units_judged"] == {"A": judged, "B": judged}
    table = written["figures"]["A"]["workplace"]
    assert list(table) == ["0.1", "0.5", "1"]
    assert _column(table, "units_failing") == [{}, {}, {"jpl": drift_settings.OUTSIDE}]
    assert _column(table, "within_alpha_on_every_unit") == [True, True, False]
    assert _column(table, "offer_kwh_per_counting_day_summed_over_units") == [1.5, 3.5, 5.5]
    assert capsys.readouterr().out.startswith(
        "Wrote drift-eta.json: tier A workplace 0.5, other 1; tier B workplace 1, other 1, by "
        f"{FIRST} per site type (start zero)"
    )


def test_one_failing_unit_fails_the_step_size_while_the_types_other_units_pass(tmp_path):
    """jpl's driver loss is outside its level at 0.5 alone, caltech passes everywhere: tier A's
    workplaces fail at 0.5, so 0.1, though 1 passes again; tier B, which does not hold the
    driver loss down, gives 1."""
    hand = {**HAND, "jpl": ((0.5, True), (1.5, False), (2.5, True))}
    status, written = _write_first(tmp_path, _hand_document(hand=hand))
    assert status == 0
    assert written["eta"]["A"]["workplace"] == 0.1
    assert written["eta"]["B"]["workplace"] == 1.0
    table = written["figures"]["A"]["workplace"]
    assert _column(table, "units_failing") == [{}, {"jpl": drift_settings.OUTSIDE}, {}]


@pytest.mark.parametrize("absent", ["no summary", "no guarded step"])
def test_a_unit_absent_at_one_step_size_fails_it(tmp_path, absent):
    """Reading E27: caltech has no summary, or no guarded step, at 0.5 in tier A; jpl passes
    there. The step size fails on caltech, so tier A's workplaces get 0.1; skipping caltech at
    0.5, as `figures` does, would pass 0.5 on jpl alone and give 0.5."""
    document = _hand_document()
    summaries = _summaries(document, "caltech")
    if absent == "no summary":
        del summaries["0.5"]
    else:
        summaries["0.5"] = _hand_summary("A", 0.5, 2.0, True, 0)
    found = drift_settings.grid_figures(document, "A", "zero", ["caltech", "jpl"])
    assert found[0.5].failing == (("caltech", drift_settings.NO_STEP),)
    assert drift_settings.figures(document, "A", "zero", ["caltech", "jpl"])[0.5].within_alpha
    status, written = _write_first(tmp_path, document)
    assert status == 0
    assert written["eta"]["A"]["workplace"] == 0.1
    assert written["eta"]["B"]["workplace"] == 1.0


def test_a_step_size_with_no_guarded_step_is_kept_and_fails(tmp_path):
    """Reading E26: no workplace unit has a summary at 0.5 in tier B; `figures` drops 0.5, the
    plan's rule keeps it as a candidate that fails, so tier B's workplaces get 0.1, not 1."""
    document = _hand_document()
    for unit in ("caltech", "jpl"):
        del _summaries(document, unit, "B")["0.5"]
    assert 0.5 not in drift_settings.figures(document, "B", "zero", ["caltech", "jpl"])
    found = drift_settings.grid_figures(document, "B", "zero", ["caltech", "jpl"])
    assert sorted(found) == [0.1, 0.5, 1.0] and not found[0.5].within_alpha
    status, written = _write_first(tmp_path, document)
    assert status == 0
    assert written["eta"]["B"] == {"workplace": 0.1, "other": 1.0}
    table = written["figures"]["B"]["workplace"]
    assert table["0.5"]["units_failing"] == {
        "caltech": drift_settings.NO_STEP,
        "jpl": drift_settings.NO_STEP,
    }
    assert table["0.5"]["share_guard_abstaining_of_offering_days"] is None


def test_a_unit_with_no_guarded_step_at_any_step_size_is_not_judged(tmp_path):
    """Reading E28: ASK ran but none of its guarded days offers; like office001 (too few days)
    it has a mean loss at no step size and is not judged, so the type other takes OSL_S's 1;
    judged, it would fail every step size and the type would offer nothing."""
    status, written = _write_first(tmp_path, _hand_document(no_steps=("ASK",)))
    assert status == 0
    assert written["site_types"]["other"] == {"units": ["OSL_S", "ASK"], "units_left_out": {}}
    assert written["units_judged"]["A"]["other"] == ["OSL_S"]
    assert written["eta"]["A"]["other"] == 1.0 and written["eta"]["B"]["other"] == 1.0


def test_offers_nothing_is_written_per_tier_and_site_type_and_read_by_step_size(tmp_path, capsys):
    """OSL_S's driver loss is outside its level at every step size: tier A of the type other
    offers nothing; tier B, which does not hold the driver loss down, gives 1. Over all units,
    tier A offers nothing too (OSL_S fails at 0.1)."""
    hand = {**HAND, "OSL_S": ((2.0, False), (1.0, False), (0.5, False))}
    status, written = _write_first(tmp_path, _hand_document(hand=hand))
    assert status == 0
    assert written["eta"] == {
        "A": {"workplace": 0.5, "other": NOTHING},
        "B": {"workplace": 1.0, "other": 1.0},
    }
    assert "tier A workplace 0.5, other offers nothing; tier B" in capsys.readouterr().out
    assert drift_settings.step_size(written, "A", OTHER_SITE) == NOTHING
    assert drift_settings.step_size(written, "A", WORKPLACE) == 0.5
    assert drift_settings.step_size(written, "B", OTHER_SITE) == 1.0
    status, one = _write_first(tmp_path, _hand_document(hand=hand), "all-units", "one.json")
    assert status == 0
    assert one["eta"] == {"A": NOTHING, "B": 1.0}
    assert one["units_judged"]["A"] == ["caltech", "jpl", "OSL_S", "ASK"]
    for kind in (WORKPLACE, OTHER_SITE):
        assert drift_settings.step_size(one, "A", kind) == NOTHING
        assert drift_settings.step_size(one, "B", kind) == 1.0


def test_a_units_guarded_days_are_steps_at_every_step_size_or_at_none():
    """What readings E26 and E27 rest on: in `drift_path`, a guarded day is a step when it
    offers, whatever eta, so a unit's offering days are the same at every step size."""
    found, _ = _unit(3)
    for tier in drift_path.TIERS:
        for start in drift_path.STARTS:
            days = {
                p.summary()["offering_days"]
                for p in found.paths
                if (p.tier, p.start) == (tier, start)
            }
            assert len(days) == 1 and days.pop() > 0
