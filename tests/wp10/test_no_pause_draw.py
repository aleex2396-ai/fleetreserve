"""The draw of the no-pause sensitivity (`no_pause`), on made-up cars.

1. A car is drawn exactly when the first 8 bytes of the SHA-256 of `no-pause|<seed>|<site>|
   <session id>`, read big-endian, are below a third of 2^64 (recomputed here with its own
   arithmetic, on 3,000 made-up cars); at share 0 no car is drawn.
2. Over 30,000 made-up cars the share drawn is within 0.01 of 1/3 (the binomial standard deviation
   is about 0.0027).
3. Small days: no count per day is fixed. Among 2,000 made-up days of three cars each count 0, 1,
   2 and 3 occurs, and the days with every car drawn are near 1/27 of them (between 1/54 and
   2/27).
4. The draw reads only the share, the seed, the site and the session: a car keeps its draw when the
   day's other cars, its position in the day and the day's date change, and a session
   plugged in on two days is drawn alike on both.
5. A different seed gives a different draw (on 3,000 made-up cars the two draws differ on more
   than a quarter of them); the same site and session at another site is another car.
6. Only the shares "0" and "1/3" are read (`read_share`); "0.333", "1/2", "0.0", "", "1" and "0.5"
   are refused with a message naming the two values. A seed that is not a whole number (1.5,
   True) and a share not of the two are refused too.
7. `counts` equals a recount; a day without cars is not a day with every car drawn; `without`
   removes exactly the cars named and refuses a position that is not a car of the day.
8. A site or session identifier holding `|` is refused (with it, two cars' hashed texts could be
   equal: site `A|b` with session `c`, and site `A` with session `b|c`).
9. The written values: the two shares and their texts, the default seed 1, the keys of the
   counts and the counts of three hand-drawn days (the first with every car drawn); a fleet of
   another length than the day's cars, a number of draws other than the number of days and a
   position not of the day are refused; `without` removes the named cars' forecasts with them.
"""

import datetime
import hashlib
from fractions import Fraction

import pytest

from fleetreserve import no_pause
from fleetreserve.day import Day, ForecastCar, TrueCar
from fleetreserve.days_from_truth import DayRecord, FleetCar, TruthDay

THIRD = Fraction(1, 3)
SEED = no_pause.DEFAULT_SEED
DATE = datetime.date(2020, 1, 1)


def _own_draw(site, session, seed) -> bool:
    digest = hashlib.sha256(f"no-pause|{seed}|{site}|{session}".encode()).digest()
    return 3 * int.from_bytes(digest[:8], "big") < 2**64


def truth_of(sessions, site="A", date=DATE) -> TruthDay:
    cars = tuple(TrueCar(4, 2.0, 1.0 + 0.1 * k) for k in range(len(sessions)))
    day = Day("u", date, 2, 3, 0.0, cars)
    record = DayRecord(
        "u",
        site,
        date.isoformat(),
        "t",
        len(cars),
        len(cars),
        0,
        0.0,
        0,
        0.0,
        0,
        0.0,
        False,
        False,
        0,
        0,
    )
    fleet = tuple(FleetCar(k, s, site, 8, False, 4.5) for k, s in enumerate(sessions))
    return TruthDay(day, record, fleet)


def test_the_draw_is_the_hash_rule_and_share_zero_draws_nothing():
    drawn = 0
    for k in range(3000):
        expected = _own_draw("A", f"s{k}", SEED)
        assert no_pause.is_drawn("A", f"s{k}", THIRD, SEED) is expected
        assert no_pause.is_drawn("A", f"s{k}", Fraction(0), SEED) is False
        drawn += expected
    assert 0 < drawn < 3000


def test_the_share_drawn_is_near_one_third():
    found = sum(no_pause.is_drawn("A", f"car-{k}", THIRD, SEED) for k in range(30000))
    assert abs(found / 30000 - 1 / 3) < 0.01


def test_no_count_per_day_is_fixed_on_three_car_days():
    sizes = [
        len(no_pause.drawn(truth_of([f"d{k}-{j}" for j in range(3)]), THIRD, SEED))
        for k in range(2000)
    ]
    assert set(sizes) == {0, 1, 2, 3}
    every = sizes.count(3) / len(sizes)
    assert 1 / 54 < every < 2 / 27


def test_the_draw_reads_only_share_seed_site_and_session():
    sessions = [f"x{k}" for k in range(40)]
    alone = {s: no_pause.drawn(truth_of([s]), THIRD, SEED) == (0,) for s in sessions}
    together = no_pause.drawn(truth_of(sessions), THIRD, SEED)
    assert {sessions[i] for i in together} == {s for s, d in alone.items() if d}
    reordered = list(reversed(sessions))
    later = no_pause.drawn(truth_of(reordered, date=datetime.date(2021, 6, 3)), THIRD, SEED)
    assert {reordered[i] for i in later} == {sessions[i] for i in together}
    # a session plugged in on two days: the same car, the same draw on both
    drawn_session = next(s for s, d in alone.items() if d)
    kept_session = next(s for s, d in alone.items() if not d)
    first = truth_of([drawn_session, kept_session, "other-1"], date=DATE)
    second = truth_of(["other-2", kept_session, drawn_session], date=DATE + datetime.timedelta(1))
    assert 0 in no_pause.drawn(first, THIRD, SEED) and 1 not in no_pause.drawn(first, THIRD, SEED)
    assert 2 in no_pause.drawn(second, THIRD, SEED)
    assert 1 not in no_pause.drawn(second, THIRD, SEED)


def test_another_seed_or_site_gives_another_draw():
    one = [no_pause.is_drawn("A", f"s{k}", THIRD, SEED) for k in range(3000)]
    two = [no_pause.is_drawn("A", f"s{k}", THIRD, SEED + 1) for k in range(3000)]
    site = [no_pause.is_drawn("B", f"s{k}", THIRD, SEED) for k in range(3000)]
    assert sum(a != b for a, b in zip(one, two, strict=True)) > 750
    assert sum(a != b for a, b in zip(one, site, strict=True)) > 750


@pytest.mark.parametrize("text", ["0.333", "1/2", "0.0", "", "1", "0.5"])
def test_only_the_two_shares_of_the_plan_are_read(text):
    with pytest.raises(no_pause.NoPauseError, match=r"\['0', '1/3'\]"):
        no_pause.read_share(text)


def test_the_two_shares_and_a_bad_seed():
    assert no_pause.read_share("0") == 0 and no_pause.read_share("1/3") == THIRD
    with pytest.raises(no_pause.NoPauseError):
        no_pause.share_text(Fraction(1, 2))
    with pytest.raises(no_pause.NoPauseError):
        no_pause.is_drawn("A", "s", THIRD, 1.5)
    with pytest.raises(no_pause.NoPauseError):
        no_pause.is_drawn("A", "s", THIRD, True)


def test_counts_equal_a_recount_and_without_removes_the_named_cars():
    truths = [truth_of([f"c{k}-{j}" for j in range(1 + k % 4)]) for k in range(300)]
    draws = [no_pause.drawn(t, THIRD, SEED) for t in truths]
    found = no_pause.counts(truths, draws)
    assert found.cars == sum(len(t.fleet) for t in truths)
    assert found.drawn == sum(
        _own_draw(c.site, c.session_id, SEED) for t in truths for c in t.fleet
    )
    every = 0
    for t in truths:
        if all(_own_draw(c.site, c.session_id, SEED) for c in t.fleet):
            every += 1
    assert found.days_every_car_drawn == every > 0
    day = truths[3].day
    assert no_pause.without(day, (0, 2)).cars == (day.cars[1], day.cars[3])
    assert no_pause.without(day, ()).cars == day.cars
    with pytest.raises(ValueError):
        no_pause.without(day, (4,))


def test_a_day_without_cars_is_not_a_day_with_every_car_drawn():
    empty = truth_of([])
    assert no_pause.drawn(empty, THIRD, SEED) == ()
    found = no_pause.counts([empty, truth_of(["e1"])], [(), ()])
    assert (found.cars, found.drawn, found.days_every_car_drawn) == (1, 0, 0)


def test_the_written_values_and_the_refused_inputs():
    assert {"0": Fraction(0), "1/3": Fraction(1, 3)} == no_pause.SHARES
    assert no_pause.DEFAULT_SEED == 1
    assert no_pause.share_text(Fraction(0)) == "0" and no_pause.share_text(THIRD) == "1/3"
    assert no_pause.Counts(5, 2, 1).as_dict() == {
        "cars": 5,
        "cars_drawn": 2,
        "days_every_car_drawn": 1,
    }
    two = truth_of(["f1", "f2"])
    found = no_pause.counts([two, two, truth_of(["f3"])], [(0, 1), (1,), ()])
    assert (found.cars, found.drawn, found.days_every_car_drawn) == (5, 3, 1)
    with pytest.raises(ValueError, match="one entry per car"):
        no_pause.drawn(TruthDay(two.day, two.record, two.fleet[:1]), THIRD, SEED)
    with pytest.raises(ValueError, match="one draw per day"):
        no_pause.counts([two], [(), ()])
    with pytest.raises(ValueError, match="one draw per day"):
        no_pause.counts([two, two], [()])
    with pytest.raises(ValueError, match="not all cars"):
        no_pause.without(two.day, (2,))
    forecasts = (ForecastCar(4.0, 2.0, 1.0, 1.0), ForecastCar(4.0, 2.0, 1.1, 1.1))
    day = Day("u", DATE, 2, 3, 0.0, two.day.cars, forecasts, 0.1)
    kept = no_pause.without(day, (0,))
    assert kept.cars == (day.cars[1],) and kept.forecasts == (forecasts[1],)


@pytest.mark.parametrize(("site", "session"), [("A|b", "c"), ("A", "b|c")])
def test_a_separator_in_an_identifier_is_refused(site, session):
    with pytest.raises(no_pause.NoPauseError, match="'\\|'"):
        no_pause.is_drawn(site, session, THIRD, SEED)
    with pytest.raises(no_pause.NoPauseError):
        no_pause.drawn(truth_of([session], site=site), THIRD, SEED)
