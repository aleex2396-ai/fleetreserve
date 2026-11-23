"""Made-up sealed files of the three held-out sets, small enough to check by hand, for the tests of
`heldout_truth_build`.

Each day of a unit repeats one pattern of sessions, written in the layout of the set's loader:
- Norway (`norway_bytes`): OSL_T over `OSL_T_DAYS` days from 6 January 2020 and TRO_R over
  `TRO_R_DAYS` days, each day four cars 15:30 to 22:30 (10 kWh) and two cars 16:30 to 21:30
  (8 kWh): six cars plugged in from 17:00 to 21:00, four at 16:00 and 22:00, so the window is τ
  16:00 with S 17:00 to 23:00 (six hours), and every day counts (four cars at τ).
- a.s.r. (`asr_bytes`): 27 and 28 February and 1 and 2 March 2023, each day three cars 07:30 to
  17:30 (20 kWh: 2 kW on average), one car 08:30 to 12:00 (5 kWh) and one row of 0 kWh,
  removed; on 2 March the car EV1 leaves at 12:30 (4 kW on average), so its highest average
  over both units is 4 kW and over the first unit alone 2 kW.
- Korea (`korea_bytes`): `korea-home` (Apartment) over `HOME_DAYS` days from 1 March 2022, each
  evening four cars 21:30 to 07:30 the next morning (30 kWh) and one car 22:30 to 06:00 (20 kWh):
  the plugged-in count peaks at 23:00 and runs past midnight, so τ is 22:00 and S is 23:00 to
  07:00 (eight hours); two fast-charger rows. `korea-work` (Company), six days: three cars 07:30
  to 17:30 and two 08:30 to 17:30, so τ 08:00 (three cars) and S 09:00 to 17:00 (eight hours);
  one fast-charger row. `korea-public-parking` (Public parking lot), four days: three cars 08:30
  to 14:30 and one 09:30 to 12:30, so τ 09:00 and S 10:00 to 15:00 (five hours).
  `korea-lodging` (Hotel), five days: three cars 17:30 to 09:30 the next morning. `korea-visits`
  (Restaurant): one day only, so it has no calibration day and abstains.
"""

from datetime import date, datetime, timedelta
from unittest import mock

from fake_heldout import ASR_FILE, NORWAY_FILE, asr_csv, asr_row
from fake_korea import KOREA_FILE, korea_csv, korea_row
from fake_raw import NORWAY_HEADER, norway_row, raw_root, write_norway

OSL_T_DAYS = 30
TRO_R_DAYS = 6
HOME_DAYS = 10
NORWAY_START = date(2020, 1, 6)
KOREA_START = date(2022, 3, 1)
FORM = "%Y-%m-%d %H:%M:%S"


def at(day: date, hours: float, form: str = FORM) -> str:
    """The local time `hours` after the start of `day` (above 24: a later date)."""
    return (datetime(day.year, day.month, day.day) + timedelta(hours=hours)).strftime(form)


# (user, plug-in hours, plug-out hours, kWh) per day
NORWAY_DAY = [(f"u{k}", 15.5, 22.5, 10.0) for k in range(1, 5)] + [
    (f"u{k}", 16.5, 21.5, 8.0) for k in (5, 6)
]
NORWAY_ODD_DAY = [(f"v{k}", 15.5, 16.5, 4.0) for k in range(1, 4)] + [
    (f"v{k}", 9.5, 11.5, 6.0) for k in range(4, 14)
]


def norway_rows(site: str, days, pattern=None, patterns: dict | None = None) -> list[dict]:
    """Rows of `site` on each of `days`; `patterns` maps a date to another day pattern."""
    rows = []
    for day in days:
        chosen = (patterns or {}).get(day, pattern or NORWAY_DAY)
        for k, (user, start, end, energy) in enumerate(chosen):
            session = f"{site}-{day}-{k}"
            rows.append(norway_row(site, user, session, at(day, start), at(day, end), energy))
    return rows


def norway_days(start: date, n: int) -> list[date]:
    return [start + timedelta(days=k) for k in range(n)]


# A site whose window search cycles (worked out by hand in the window tests): `CYCLE_DAYS` days
# from `NORWAY_START`, the days `CYCLE_B_INDEX` of type B, the rest of type A.
# A: four cars 09:30 to 10:30 and ten 14:30 to 16:30: 4 plugged in at 10:00, 10 at 15:00 and
#    16:00, none at any other whole hour.
# B: ten cars 10:30 to 13:30 and six 13:30 to 17:30: 10 at 11:00 to 13:00, 6 at 14:00 to 17:00.
# B short: as B, with four cars 13:30 to 14:30 in place of the six: 4 at 14:00, none at 15:00.
CYCLE_DAYS = 34
CYCLE_B_INDEX = (3, 11, 19, 27)
CYCLE_A_DAY = [(f"a{k}", 9.5, 10.5, 3.0) for k in range(1, 5)] + [
    (f"a{k}", 14.5, 16.5, 8.0) for k in range(5, 15)
]
CYCLE_B_DAY = [(f"b{k}", 10.5, 13.5, 9.0) for k in range(1, 11)] + [
    (f"b{k}", 13.5, 17.5, 12.0) for k in range(11, 17)
]
CYCLE_B_SHORT_DAY = CYCLE_B_DAY[:10] + [(f"b{k}", 13.5, 14.5, 3.0) for k in range(11, 15)]


def cycle_rows(site: str = "OSL_T", b_day=None) -> tuple[list[dict], list[date], list[date]]:
    """The rows of the cycling site, its A days and its B days."""
    days = norway_days(NORWAY_START, CYCLE_DAYS)
    b_days = [days[k] for k in CYCLE_B_INDEX]
    a_days = [day for day in days if day not in b_days]
    pattern = CYCLE_B_DAY if b_day is None else b_day
    rows = norway_rows(site, days, pattern=CYCLE_A_DAY, patterns=dict.fromkeys(b_days, pattern))
    return rows, a_days, b_days


# A second cycling site, whose start draw is mostly of a third type: of `CYCLE_DAYS` days, the days
# `CYCLE_B_INDEX` of type B, `CYCLE_A_INDEX` of type A and the other 26 of type C.
# C: ten cars 14:30 to 19:30: 10 plugged in at 15:00 to 19:00, none at any other whole hour (so
#    none at 10:00 or 14:00).
CYCLE_A_INDEX = (5, 13, 21, 29)
CYCLE_C_DAY = [(f"c{k}", 14.5, 19.5, 15.0) for k in range(1, 11)]


def three_type_cycle_rows(site: str = "OSL_T") -> tuple[list[dict], list, list, list]:
    """The rows of the second cycling site, its A days, its B days and its C days."""
    days = norway_days(NORWAY_START, CYCLE_DAYS)
    a_days = [days[k] for k in CYCLE_A_INDEX]
    b_days = [days[k] for k in CYCLE_B_INDEX]
    c_days = [day for day in days if day not in a_days and day not in b_days]
    patterns = {**dict.fromkeys(a_days, CYCLE_A_DAY), **dict.fromkeys(b_days, CYCLE_B_DAY)}
    rows = norway_rows(site, days, pattern=CYCLE_C_DAY, patterns=patterns)
    return rows, a_days, b_days, c_days


def norway_bytes(folder, rows: list[dict] | None = None, header=NORWAY_HEADER) -> bytes:
    if rows is None:
        rows = norway_rows("OSL_T", norway_days(NORWAY_START, OSL_T_DAYS))
        rows += norway_rows("TRO_R", norway_days(NORWAY_START, TRO_R_DAYS))
    path = write_norway(raw_root(folder), "held-out", rows, header=header)
    return path.read_bytes()


ASR_DAYS = [date(2023, 2, 27), date(2023, 2, 28), date(2023, 3, 1), date(2023, 3, 2)]
ASR_FORM = "%Y-%m-%dT%H:%M:%S"


def asr_rows() -> list[dict]:
    rows = []
    for day in ASR_DAYS:
        for k in range(1, 4):
            end = 12.5 if (k, day) == (1, ASR_DAYS[-1]) else 17.5  # EV1 at 4 kW on 2 March
            rows.append(asr_row(f"EV{k}", at(day, 7.5, ASR_FORM), at(day, end, ASR_FORM), 20.0))
        rows.append(asr_row("EV4", at(day, 8.5, ASR_FORM), at(day, 12.0, ASR_FORM), 5.0))
        rows.append(asr_row("EV5", at(day, 9.0, ASR_FORM), at(day, 10.0, ASR_FORM), 0.0))
    return rows


def asr_bytes(**options) -> bytes:
    return asr_csv(asr_rows(), **options)


# portfolio: (location type, number of days, day pattern of (user, start, end, kWh))
KOREA_PATTERNS = {
    "korea-home": (
        "Apartment",
        HOME_DAYS,
        [(f"1{k}", 21.5, 31.5, 30.0) for k in range(1, 5)] + [("15", 22.5, 30.0, 20.0)],
    ),
    "korea-work": (
        "Company",
        6,
        [(f"2{k}", 7.5, 17.5, 20.0) for k in range(1, 4)]
        + [(f"2{k}", 8.5, 17.5, 15.0) for k in (4, 5)],
    ),
    "korea-public-parking": (
        "Public parking lot",
        4,
        [(f"3{k}", 8.5, 14.5, 12.0) for k in range(1, 4)] + [("34", 9.5, 12.5, 6.0)],
    ),
    "korea-lodging": ("Hotel", 5, [(f"4{k}", 17.5, 33.5, 25.0) for k in range(1, 4)]),
    "korea-visits": ("Restaurant", 1, [(f"5{k}", 12.0, 14.0, 5.0) for k in range(1, 4)]),
}
FAST_ROWS = {"korea-home": 2, "korea-work": 1}


def korea_rows(patterns: dict | None = None) -> list[dict]:
    rows = []
    for unit, (location, n_days, pattern) in (patterns or KOREA_PATTERNS).items():
        for day in norway_days(KOREA_START, n_days):
            for k, (user, start, end, energy) in enumerate(pattern):
                session = f"{unit}-{day}-{k}"
                rows.append(
                    korea_row(
                        session, user, f"C-{unit}", location, at(day, start), at(day, end), energy
                    )
                )
        for k in range(FAST_ROWS.get(unit, 0)):
            day = KOREA_START
            rows.append(
                korea_row(
                    f"{unit}-fast-{k}",
                    "99",
                    f"F-{unit}",
                    location,
                    at(day, 12.0),
                    at(day, 12.5),
                    20.0,
                    charger_type="1",
                )
            )
    return rows


def korea_bytes(rows: list[dict] | None = None, **options) -> bytes:
    return korea_csv(korea_rows() if rows is None else rows, **options)


def sealed_files(folder) -> dict[tuple[str, str], bytes]:
    """The three made-up sealed files under the opener's set and file names."""
    return {
        NORWAY_FILE: norway_bytes(folder),
        ASR_FILE: asr_bytes(),
        KOREA_FILE: korea_bytes(),
    }


UNIT_CHECKS = ("require_sites", "require_periods", "require_portfolios")


def unchecked(loader, content: bytes):
    """`loader.from_bytes(content)` without the loader's check that each named unit keeps a row.
    For the tests of what the truth build does with a table in which a named unit has no row or
    no row kept, which no loader gives any more: each refuses such a file (the loader tests and
    `test_heldout_unit_refusal.py`)."""
    name = next(name for name in UNIT_CHECKS if hasattr(loader, name))
    with mock.patch.object(loader, name, lambda sessions, dropped: None):
        return loader.from_bytes(content)
