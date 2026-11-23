"""Made-up Korean charging files, for the tests of the Korean loader.

The layout is built only from the data paper's Table 7: its 16 columns in the order it lists
them, times written YYYY-MM-DD HH:MM:SS, `Duration` in minutes, `Demand` in kWh, `ChargerType`
1 fast and 0 slow, `Location` one of the 14 names of Table 6. The field separator and the form
of a missing value are not stated there; the writer uses "," and an empty field by default, and
can write ";" with a decimal comma. No real data is read or copied.
"""

from datetime import datetime

KOREA_FILE = ("korea", "ChargingRecords.csv")
KOREA_HEADER = (
    "ChargingsessionID",
    "UserID",
    "ChargerID",
    "ChargerCompany",
    "Location",
    "ChargerType",
    "ChargerCapacity",
    "ChargerACDC",
    "StartDay",
    "StartTime",
    "EndDay",
    "EndTime",
    "StartDatetime",
    "EndDatetime",
    "Duration",
    "Demand",
)
# A header name, trimmed and folded, to the key of a row dict (the paper's misspelling included).
_KEYS = {name.casefold(): name for name in KOREA_HEADER}
_KEYS["srartdatetime"] = "StartDatetime"


def _minutes(start: str | None, end: str | None) -> float | None:
    if start is None or end is None:
        return None
    form = "%Y-%m-%d %H:%M:%S"
    return (datetime.strptime(end, form) - datetime.strptime(start, form)).total_seconds() / 60


def korea_row(
    session,
    user,
    charger,
    location,
    start,
    end,
    demand,
    *,
    charger_type="0",
    duration="from times",
    **extra,
) -> dict:
    """One session; `duration` defaults to plug-out minus plug-in in minutes, and the day and
    time columns are split from the two times."""
    row = {
        "ChargingsessionID": session,
        "UserID": user,
        "ChargerID": charger,
        "ChargerCompany": "1",
        "Location": location,
        "ChargerType": charger_type,
        "ChargerCapacity": "2",
        "ChargerACDC": "1",
        "StartDay": None if start is None else start.split(" ")[0],
        "StartTime": None if start is None else start.split(" ")[-1],
        "EndDay": None if end is None else end.split(" ")[0],
        "EndTime": None if end is None else end.split(" ")[-1],
        "StartDatetime": start,
        "EndDatetime": end,
        "Duration": _minutes(start, end) if duration == "from times" else duration,
        "Demand": demand,
    }
    row.update(extra)
    return row


def korea_csv(rows: list[dict], header=KOREA_HEADER, sep: str = ",") -> bytes:
    """The rows under `header`; None is written as an empty field, a float under ";" with a
    decimal comma. A header name is matched to a row's key after trimming and case folding."""
    lines = [sep.join(header)]
    for row in rows:
        cells = []
        for name in header:
            value = row.get(_KEYS.get(name.strip().casefold(), name))
            text = "" if value is None else str(value)
            if sep == ";" and isinstance(value, float):
                text = text.replace(".", ",")
            cells.append(text)
        lines.append(sep.join(cells))
    return ("\n".join(lines) + "\n").encode("utf-8")
