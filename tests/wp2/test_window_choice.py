"""The window-choice command on a small made-up ACN session table: the pair it chooses, the tie
rule, what it writes, and its exit statuses."""

import json

import pytest
from fake_raw import acn_record, raw_root, write_acn

from fleetreserve import window_choice


def _cars(site: str, day: str, count: int, energy: float) -> list[dict]:
    """`count` cars that plug in at 09:00 local (16:00 UTC, May: UTC-7) on `day` and leave at
    21:00 local the same day (04:00 UTC the next), with recorded charging that ends four hours
    after plug-in (13:00 local), one user each."""
    return [
        acn_record(
            f"{site}-{day}-{n}",
            f"{day} 16:00:00",
            f"{_next(day)} 04:00:00",
            energy,
            user=f"{site}-u{n}",
            done=f"{day} 20:00:00",
            station=f"1-1-1-{n}",
        )
        for n in range(count)
    ]


def _next(day: str) -> str:
    year, month, number = day.split("-")
    return f"{year}-{month}-{int(number) + 1:02d}"


@pytest.fixture
def raw(tmp_path):
    """Only ACN files: the command must not need NREL or Norway files.

    Caltech: three cars on one day; JPL: four cars on one day (another day). Each car: 4 kWh over
    4 hours of charging, so its rate is 1 kW; plugged in from 09:00 to 21:00, 12 hours.
    """
    root = raw_root(tmp_path)
    write_acn(root, "caltech", "2019-05", _cars("caltech", "2019-05-06", 3, 4.0))
    write_acn(root, "jpl", "2019-05", _cars("jpl", "2019-05-07", 4, 4.0))
    write_acn(root, "office001", "2019-05", _cars("office001", "2019-05-08", 3, 4.0))
    return root


def test_the_chosen_pair_is_the_hand_known_best_and_ties_go_to_the_shorter_window(
    raw, tmp_path, capsys
):
    out = tmp_path / "out"
    assert window_choice.main(["--raw-root", str(raw), "--out", str(out)]) == 0
    document = json.loads((out / window_choice.RESULTS_NAME).read_text())
    by_pair = {(row["tau_hour"], row["service_hours"]): row for row in document["pairs"]}
    assert len(by_pair) == 24 and set(by_pair) == set(window_choice.candidates())

    # By hand. A car's term is max over y in [e - min(r, e), e - x_low] of
    # min(r s, y) - max(y - r c, 0), with s = min(d - 1, n) and c = max(d - 1 - n, 0).
    # τ = 9:00: every car (rate 1 kW, 4 kWh, plugged in at 09:00, so nothing is drawn before τ)
    # has room 4 and departure hour 12, so s = n, c = 11 - n and x_low = 0, y in [3, 4]. The
    # recovery hours (c at least 5) take all the energy, so the term is min(n, y) at y = 4:
    # 2 for n = 2, 3 for n = 3, 4 for n = 4 and 4 for n = 6. A day of 3 cars is worth 12 at
    # n = 4 and n = 6; a day of 4 cars is worth 16.
    assert by_pair[(9, 4)]["sites"]["caltech"]["ceiling_kwh_sum"] == pytest.approx(12.0)
    assert by_pair[(9, 4)]["sites"]["jpl"]["ceiling_kwh_sum"] == pytest.approx(16.0)
    assert by_pair[(9, 4)]["total_ceiling_kwh"] == pytest.approx(28.0)
    assert by_pair[(9, 6)]["total_ceiling_kwh"] == pytest.approx(28.0)
    assert by_pair[(9, 3)]["total_ceiling_kwh"] == pytest.approx(21.0)
    assert by_pair[(9, 2)]["total_ceiling_kwh"] == pytest.approx(14.0)
    assert by_pair[(9, 4)]["sites"]["caltech"]["counting_days"] == 1
    assert by_pair[(9, 4)]["sites"]["jpl"]["cars_in_fleet"] == 4
    # τ = 7:00 and 8:00: no car is plugged in yet, so there is no counting day and no ceiling.
    for hours in window_choice.SERVICE_HOURS:
        for tau in (7, 8):
            assert by_pair[(tau, hours)]["total_ceiling_kwh"] == 0.0
            assert by_pair[(tau, hours)]["sites"]["caltech"]["counting_days"] == 0
    # τ = 10:00 and later: the nominal cap of 6.66 kW has already delivered the 4 kWh: room 0.
    for tau in (10, 11, 12):
        assert by_pair[(tau, 4)]["total_ceiling_kwh"] == 0.0
    # The third ACN site is not used.
    assert set(by_pair[(9, 4)]["sites"]) == {"caltech", "jpl"}

    chosen = document["chosen"]
    assert (chosen["tau_hour"], chosen["service_hours"]) == (9, 4)
    assert chosen["total_ceiling_kwh"] == pytest.approx(28.0)
    assert chosen["decided_by_tie_rule"] is True
    assert [(t["tau_hour"], t["service_hours"]) for t in chosen["tied_pairs"]] == [(9, 4), (9, 6)]
    text = (out / window_choice.TEXT_NAME).read_text()
    assert "CHOSEN: tau 09:00, S 10:00 to 14:00 (4 hours)" in text
    assert "the tie rule decided the choice" in text
    assert capsys.readouterr().out == text


def test_same_inputs_give_the_same_bytes(raw, tmp_path):
    for name in ("one", "two"):
        assert window_choice.main(["--raw-root", str(raw), "--out", str(tmp_path / name)]) == 0
    for name in (window_choice.RESULTS_NAME, window_choice.TEXT_NAME):
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()


def test_default_output_is_the_run_folder(raw, tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(run_dir))
    assert window_choice.main(["--raw-root", str(raw)]) == 0
    assert (run_dir / window_choice.RESULTS_NAME).is_file()


def test_tie_rule_earlier_tau_then_shorter_service():
    totals = {(tau, hours): 1.0 for tau, hours in window_choice.candidates()}
    assert window_choice.choose(totals).pair == (7, 2)
    totals[(9, 3)] = totals[(10, 2)] = totals[(9, 6)] = 5.0
    choice = window_choice.choose(totals)
    assert choice.pair == (9, 3) and choice.tied == ((9, 3), (9, 6), (10, 2))
    assert choice.decided_by_tie_rule
    totals[(11, 2)] = 5.5
    choice = window_choice.choose(totals)
    assert choice.pair == (11, 2) and not choice.decided_by_tie_rule
    # Sums that differ only by rounding are tied.
    close = {(8, 6): 3.0, (7, 4): 3.0 - 1e-12, (9, 2): 3.0 - 1e-6}
    assert window_choice.choose(close).pair == (7, 4)


def test_missing_raw_files_cannot_run(tmp_path, capsys):
    root = raw_root(tmp_path)
    assert window_choice.main(["--raw-root", str(root), "--out", str(tmp_path / "o")]) == 2
    assert "COULD NOT RUN" in capsys.readouterr().out
    assert not (tmp_path / "o").exists()


def test_an_unwritable_output_folder_cannot_run(raw, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a folder")
    assert window_choice.main(["--raw-root", str(raw), "--out", str(blocker / "sub")]) == 2


def test_a_file_without_its_published_layout_fails_with_exit_1(tmp_path, capsys):
    root = raw_root(tmp_path)
    wrong_zone = [
        acn_record("c-0", "2019-05-06 16:00:00", "2019-05-07 04:00:00", 4.0, zone="Europe/Oslo")
    ]
    write_acn(root, "caltech", "2019-05", wrong_zone)
    write_acn(root, "jpl", "2019-05", _cars("jpl", "2019-05-07", 3, 4.0))
    out = tmp_path / "o"
    assert window_choice.main(["--raw-root", str(root), "--out", str(out)]) == 1
    assert capsys.readouterr().out.startswith("FAILED")
    assert not out.exists()


def test_the_output_states_its_readings_and_the_sizes_of_the_readings(raw, tmp_path):
    out = tmp_path / "out"
    assert window_choice.main(["--raw-root", str(raw), "--out", str(out)]) == 0
    document = json.loads((out / window_choice.RESULTS_NAME).read_text())
    text = (out / window_choice.TEXT_NAME).read_text()
    readings = document["readings"]
    assert set(readings) == {"ceiling", "decision_hour_departures", "room_cap", "small_fleets"}
    for statement, phrases in (
        ("ceiling", ("known-car causal ceiling", "fixed at the decision time", "best-causal")),
        ("ceiling", ("not b(S) - p(S) with a free decision hour",)),
        ("decision_hour_departures", ("left out of the fleet", "the day still counts")),
        ("room_cap", ("capped at rate times departure hour", "energy removed")),
        ("small_fleets", ("below 3 cars", "is kept")),
    ):
        for phrase in phrases:
            assert phrase in readings[statement], (statement, phrase)
            assert phrase in text, (statement, phrase)
    entry = document["pairs"][0]["sites"]["caltech"]
    assert {
        "kwh_removed_by_room_cap",
        "cars_with_room_capped",
        "cars_left_out_departing_in_decision_hour",
        "days_with_fleet_below_3_kept",
    } <= set(entry)
    assert "room cap removed" in text


def test_the_sizes_of_the_cap_and_of_cars_left_out_are_reported(tmp_path):
    # Three cars a day plug in at 09:00 local and leave at 10:54, 1.9 hours later, with 3.8 kWh
    # and no recorded charging time, so the rate is 3.8 / 1.9 = 2 kW. At τ 9:00 the room is 3.8
    # and the departure hour is 1: the cap takes it to 2 kW x 1 h = 2 kWh, removing 1.8 kWh a
    # car, 5.4 kWh a site-day; their ceiling is 0. At τ 10:00 the cars have drawn their 3.8 kWh
    # from plug-in and leave within the decision hour (departure hour 0): all three are left
    # out, the day still counts, and its fleet is empty. At 11:00 and later nobody is plugged in.
    root = raw_root(tmp_path)
    for site in ("caltech", "jpl"):
        records = [
            acn_record(
                f"{site}-{n}",
                "2019-05-06 16:00:00",
                "2019-05-06 17:54:00",
                3.8,
                user=f"{site}-u{n}",
                station=f"1-1-1-{n}",
            )
            for n in range(3)
        ]
        write_acn(root, site, "2019-05", records)
    out = tmp_path / "out"
    assert window_choice.main(["--raw-root", str(root), "--out", str(out)]) == 0
    document = json.loads((out / window_choice.RESULTS_NAME).read_text())
    by_pair = {(row["tau_hour"], row["service_hours"]): row for row in document["pairs"]}
    nine = by_pair[(9, 4)]["sites"]["caltech"]
    assert nine["counting_days"] == 1 and nine["cars_in_fleet"] == 3
    assert nine["cars_with_room_capped"] == 3
    assert nine["kwh_removed_by_room_cap"] == pytest.approx(5.4)
    assert nine["ceiling_kwh_sum"] == 0.0
    ten = by_pair[(10, 4)]["sites"]["jpl"]
    assert ten["counting_days"] == 1 and ten["cars_at_tau"] == 3 and ten["cars_in_fleet"] == 0
    assert ten["cars_left_out_departing_in_decision_hour"] == 3
    assert ten["days_with_fleet_below_3_kept"] == 1
    assert by_pair[(11, 4)]["sites"]["caltech"]["counting_days"] == 0
    assert "room cap removed 5.400 kWh" in (out / window_choice.TEXT_NAME).read_text()
