"""Where each site lies and which public context files describe it (plan sections 3 and 7).

The departure model's weather and OxCGRT inputs (`weather_inputs`, `oxcgrt_inputs`) are read
per site from public files downloaded by `tools/fetch_data.py --context`. This module holds the
one table that maps a site (the `site` column of a cars table) to those files: `SITE_PLACES`, a
site to its weather places (a site's weather is the plain mean over its places), and
`SITE_OXCGRT`, a site to its OxCGRT country code. `PLACES` gives each place's point and the dates
its weather file covers; `weather_url` the one query that downloads it.

Every point is a public fact about a place, never read from a charging data file. Sources and
readings, each with its alternative in brackets:

- ACN (Lee et al. 2019; the ACN-Data web page, read 15 Nov 2026): Caltech in Pasadena, JPL in La
  Canada Flintridge, office001 "an office building located in the Silicon Valley area". Points:
  the Caltech campus and the JPL campus (general knowledge of where they lie).
  P1. office001: a point in the middle of the Santa Clara valley (Sunnyvale), since the site's
      own place is not published; about 30 km from either end of the area the page names.
      [Alternative: no weather for office001.]
- NREL workplace (data.nlr.gov 182; plan section 3: two sites, pooled, Mountain time):
  P2. both sites at NREL's main campus in Golden, Colorado. [Alternative: one of them on the
      Flatirons campus, about 25 km north.]
- Norway (the data paper, Sorensen et al. 2024, Data in Brief 57:110883, its specifications
  table): "Asker (59.83 N, 10.43E), Baerum (2) (59.56 N, 10.30E), Bergen (60.23 N, 5.19E), Bodo
  (67.16 N, 14.24E), Krokkleiva (59.86 N, 10.78E), Oslo (4) (59.54 N,10.45E), and Trondheim (2)
  (63.44 N, 10.42E)", and per site its place (ASK Asker; BAR and BAR_2 Baerum; BER Bergen
  south-west; BOD Bodo; KRO Krokkleiva; OSL_1, OSL_2, OSL_S, OSL_T Oslo; TRO, TRO_R Trondheim).
  P3. The printed pairs mix two forms. Baerum (59 deg 56', 10 deg 30'), Bergen (60 deg 23',
      5 deg 19'), Bodo (67 deg 16', 14 deg 24') and Oslo (59 deg 54', 10 deg 45') are read as
      degrees and minutes: so read they are the usual centre points of those places (general
      knowledge), while as decimals Baerum and Oslo would lie some 40 km south in the Oslofjord
      area and Bodo about 13 km south of the town. Asker, Krokkleiva and Trondheim are read as
      printed decimals (Asker's and Krokkleiva's fractional parts, .83 and .86, cannot be
      minutes). [Alternatives: every pair as printed decimals; Bergen as printed decimals, about
      17 km south-west of the centre, which the site's description "Bergen south-west" may
      mean.] Every site of one place shares its point.
- a.s.r. (its README, `data/raw/asr/README.txt`: "an office parking lot located in Utrecht"):
  P4. the centre of Utrecht. [Alternative: the a.s.r. office in Utrecht's east, about 3 km away,
      inside the same 9 km model cell or the next.]
- Korea (plan section 7: "national for Korea"; the data set has no site identifier, so a
  portfolio pools car parks across the country):
  P5. the national weather of a Korean portfolio is the plain mean over seven points: Seoul and
      the six metropolitan cities (Busan, Incheon, Daegu, Daejeon, Gwangju, Ulsan). [Alternatives:
      a mean weighted by population; Seoul alone.]

Dates (`Place.first`, `Place.last`): the union of the published spans of the sites of a place,
from the same sources (the ACN month files' names; NREL from the archive's first day, 1 Jan 2017,
to its file's month, October 2021; the Norway paper's per-site spans; the a.s.r. README's 25 Aug
2020 to 2 Oct 2024; for Korea the Korean loader's accepted range, 1 Sep 2021 to 31 Oct 2022),
widened by `MARGIN_DAYS` on each side so that a 24-hour window ending before the first decision
time is covered. The archive holds nothing before 1 Jan 2017 (`ARCHIVE_FIRST`).

OxCGRT (`SITE_OXCGRT`): the jurisdiction of each site as the OxCGRT file keys it: the state
rows `US_CA` for the ACN sites (California) and `US_CO` for NREL (Colorado), and the national rows
`NOR`, `NLD` and `KOR` (ISO 3166-1 alpha-3 codes). P6: the state rows for the United States, since
the measures that closed workplaces there were set by the states. [Alternative: the national
row `USA`.]

A site listed in neither table raises `KeyError` where it is looked up: no site gets a default.
The table holds the sites of plan section 3 under the names the loaders give them, the same
sites as `learn_features.SITE_COUNTRIES`: the Korean portfolios under the five unit names of the
Korean loader (`load_korea.UNITS`: `korea-home`, `korea-lodging`, `korea-work`,
`korea-public-parking`, `korea-visits`). The data-set name `korea` is no site.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from urllib.parse import urlencode

from fleetreserve.load_korea import UNITS as KOREA_UNITS

ARCHIVE_FIRST = date(2017, 1, 1)
MARGIN_DAYS = 7
WEATHER_HOST = "https://historical-forecast-api.open-meteo.com/v1/forecast"
WEATHER_MODEL = "ecmwf_ifs"  # ECMWF IFS HRES 9 km, archived from 1 Jan 2017
WEATHER_VARIABLES = ("temperature_2m", "precipitation", "snowfall")
WEATHER_SET = "weather"


@dataclass(frozen=True)
class Place:
    latitude: float
    longitude: float
    first: date
    last: date


def _span(first: date, last: date) -> tuple[date, date]:
    margin = timedelta(days=MARGIN_DAYS)
    return max(first - margin, ARCHIVE_FIRST), last + margin


def _place(latitude: float, longitude: float, first: date, last: date) -> Place:
    return Place(latitude, longitude, *_span(first, last))


def _minutes(degrees: int, minutes: int) -> float:
    return round(degrees + minutes / 60, 4)


ACN_LAST = date(2021, 9, 30)
PLACES: dict[str, Place] = {
    "pasadena-caltech": _place(34.1377, -118.1253, date(2018, 4, 1), ACN_LAST),
    "la-canada-jpl": _place(34.2013, -118.1714, date(2018, 9, 1), ACN_LAST),
    "silicon-valley": _place(37.3688, -122.0363, date(2019, 3, 1), ACN_LAST),
    "golden-nrel": _place(39.7405, -105.1686, ARCHIVE_FIRST, date(2021, 10, 31)),
    "asker": _place(59.83, 10.43, date(2018, 11, 15), date(2020, 2, 3)),
    "baerum": _place(_minutes(59, 56), _minutes(10, 30), date(2018, 9, 7), date(2021, 8, 6)),
    "bergen": _place(_minutes(60, 23), _minutes(5, 19), date(2019, 10, 31), date(2020, 2, 3)),
    "bodo": _place(_minutes(67, 16), _minutes(14, 24), date(2018, 10, 24), date(2020, 2, 2)),
    "krokkleiva": _place(59.86, 10.78, date(2021, 1, 15), date(2021, 5, 6)),
    "oslo": _place(_minutes(59, 54), _minutes(10, 45), date(2018, 2, 6), date(2021, 3, 29)),
    "trondheim": _place(63.44, 10.42, date(2018, 12, 21), date(2020, 2, 3)),
    "utrecht": _place(52.0907, 5.1214, date(2020, 8, 25), date(2024, 10, 2)),
    **{
        name: _place(lat, lon, date(2021, 9, 1), date(2022, 10, 31))
        for name, lat, lon in (
            ("seoul", 37.5665, 126.9780),
            ("busan", 35.1796, 129.0756),
            ("incheon", 37.4563, 126.7052),
            ("daegu", 35.8714, 128.6014),
            ("daejeon", 36.3504, 127.3845),
            ("gwangju", 35.1595, 126.8526),
            ("ulsan", 35.5384, 129.3114),
        )
    },
}
KOREA_PLACES = ("seoul", "busan", "incheon", "daegu", "daejeon", "gwangju", "ulsan")
KOREA_SITES = KOREA_UNITS
SITE_PLACES: dict[str, tuple[str, ...]] = {
    "caltech": ("pasadena-caltech",),
    "jpl": ("la-canada-jpl",),
    "office001": ("silicon-valley",),
    "nrel": ("golden-nrel",),
    "ASK": ("asker",),
    "BAR": ("baerum",),
    "BAR_2": ("baerum",),
    "BER": ("bergen",),
    "BOD": ("bodo",),
    "KRO": ("krokkleiva",),
    "OSL_1": ("oslo",),
    "OSL_2": ("oslo",),
    "OSL_S": ("oslo",),
    "OSL_T": ("oslo",),
    "TRO": ("trondheim",),
    "TRO_R": ("trondheim",),
    "asr": ("utrecht",),
    **dict.fromkeys(KOREA_SITES, KOREA_PLACES),
}
SITE_OXCGRT: dict[str, str] = {
    **dict.fromkeys(("caltech", "jpl", "office001"), "US_CA"),
    "nrel": "US_CO",
    **dict.fromkeys(
        ("ASK", "BAR", "BAR_2", "BER", "BOD", "KRO", "OSL_1", "OSL_2", "OSL_S", "OSL_T", "TRO"),
        "NOR",
    ),
    "TRO_R": "NOR",
    "asr": "NLD",
    **dict.fromkeys(KOREA_SITES, "KOR"),
}


def weather_file(place: str) -> str:
    """The file name of a place's weather file under `data/raw/weather/`."""
    return f"{place}.csv"


def weather_url(place: str) -> str:
    """The one query that downloads a place's hourly weather file (module text): the forecast
    archive of `WEATHER_MODEL`, times in UTC, as CSV."""
    p = PLACES[place]
    query = {
        "latitude": f"{p.latitude:.4f}",
        "longitude": f"{p.longitude:.4f}",
        "start_date": p.first.isoformat(),
        "end_date": p.last.isoformat(),
        "hourly": ",".join(WEATHER_VARIABLES),
        "models": WEATHER_MODEL,
        "timezone": "GMT",
        "format": "csv",
    }
    return f"{WEATHER_HOST}?{urlencode(query, safe=',')}"
