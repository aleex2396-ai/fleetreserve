"""Download the ACN-Data sessions of the office001 site, one calendar month at a time.

Run as `python tools/fetch_acn.py [--root <folder>]` from the project's top folder.

The files have the layout of the Caltech and JPL month files already in `data/raw/acn/`: each
month is canonical JSON (sessions sorted by `_id`, keys sorted, no whitespace, `_links` removed),
gzip-compressed with a zero time stamp, in `data/raw/acn/office001/<YYYY-MM>.json.gz`. Its
fingerprint is the SHA-256 of the canonical JSON; the table
`data/raw/acn/office001-fingerprints.csv` has one row `month,sessions,sha256` per stored
month. Months are queried from FIRST_MONTH to LAST_MONTH by plug-in time (`connectionTime`) in
UTC, start included, end excluded, 100 sessions a page. A month is accepted only if the
sessions received equal the total the server reports and no session appears twice. Months
without sessions are reported and not stored.

The access token is read by this script from TOKEN_FILE and sent only in the request's
Authorization header. It is never printed, logged, written or passed on a command line; an
HTTP error is reported by its status code only.

What it never does: print any session field (it prints months and session counts only);
overwrite a stored month whose fingerprint differs from the download (that month fails).

Exit status: 0 all months fetched and stored; 1 a stored month differs from the download, or a
month's count does not match the server's total; 2 the token is missing or refused, or the API
could not be reached.
"""

import argparse
import base64
import csv
import gzip
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import format_datetime
from pathlib import Path

TOP = Path(__file__).resolve().parents[1]
API = "https://ev.caltech.edu/api/v1/sessions/"
SITE = "office001"
FIRST_MONTH = "2018-01"
LAST_MONTH = "2026-10"
PAGE_SIZE = 100
TOKEN_FILE = "~/.config/acndata/api_key"
USER_AGENT = "data-fetch (python urllib)"
TIMEOUT_SECONDS = 90.0
RETRIES = 4

PASSED, FAILED, COULD_NOT_RUN = 0, 1, 2

Getter = Callable[[str], dict]


class FetchError(Exception):
    """The API could not be used; the message holds a status code at most, never the token."""


class CountMismatch(Exception):
    pass


def read_token(path: str | None = None) -> str:
    path = path or TOKEN_FILE  # looked up at call time
    try:
        token = Path(path).expanduser().read_text().strip()
    except OSError:
        raise FetchError(f"no readable ACN-Data token file at {path}") from None
    if not token:
        raise FetchError(f"the ACN-Data token file at {path} is empty")
    return token


def http_getter(token: str, urlopen=None, pause: float = 2.0) -> Getter:
    """A function that GETs an API URL with the token and returns the parsed JSON reply."""
    urlopen = urlopen or urllib.request.urlopen
    auth = "Basic " + base64.b64encode(f"{token}:".encode()).decode()

    def get(url: str) -> dict:
        delay = pause
        for attempt in range(RETRIES):
            request = urllib.request.Request(
                url, headers={"Authorization": auth, "User-Agent": USER_AGENT}
            )
            try:
                with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as error:
                code = error.code
                if code not in (429, 500, 502, 503, 504) or attempt == RETRIES - 1:
                    raise FetchError(f"HTTP {code} from the ACN-Data API") from None
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
                if attempt == RETRIES - 1:
                    kind = type(error).__name__
                    raise FetchError(f"network or reply error: {kind}") from None
            time.sleep(delay)
            delay *= 2
        raise FetchError("request failed")

    return get


def months_between(first: str, last: str) -> list[str]:
    year, month = map(int, first.split("-"))
    end = tuple(map(int, last.split("-")))
    months = []
    while (year, month) <= end:
        months.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def month_window(month: str) -> tuple[datetime, datetime]:
    year, number = map(int, month.split("-"))
    start = datetime(year, number, 1, tzinfo=UTC)
    end = (
        datetime(year + 1, 1, 1, tzinfo=UTC)
        if number == 12
        else datetime(year, number + 1, 1, tzinfo=UTC)
    )
    return start, end


def month_query(site: str, month: str, page: int) -> str:
    start, end = month_window(month)
    first, after = format_datetime(start, usegmt=True), format_datetime(end, usegmt=True)
    where = f'connectionTime>="{first}" and connectionTime<"{after}"'
    params = {"where": where, "page": page, "max_results": PAGE_SIZE}
    return f"{API}{site}?{urllib.parse.urlencode(params)}"


def fetch_month(site: str, month: str, get: Getter, attempts: int = 3) -> list[dict]:
    """Every session of one site that plugged in during one UTC month."""
    received, distinct, total = 0, 0, None
    for _ in range(attempts):
        items: list[dict] = []
        page = 1
        while True:
            reply = get(month_query(site, month, page))
            batch = reply.get("_items") or []
            total = (reply.get("_meta") or {}).get("total")
            for item in batch:
                item.pop("_links", None)
            items.extend(batch)
            if len(batch) < PAGE_SIZE or (total is not None and len(items) >= total):
                break
            page += 1
        ids = [item["_id"] for item in items]
        received, distinct = len(items), len(set(ids))
        if total is not None and received == total and distinct == received:
            return items
    raise CountMismatch(
        f"{site} {month}: received {received} sessions ({distinct} distinct), "
        f"the server reports {total}"
    )


def canonical_bytes(items: list[dict]) -> bytes:
    ordered = sorted(items, key=lambda item: item["_id"])
    return json.dumps(ordered, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def month_path(root: Path, site: str, month: str) -> Path:
    return Path(root) / "data" / "raw" / "acn" / site / f"{month}.json.gz"


def stored_fingerprint(path: Path) -> str:
    with gzip.open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def store_month(path: Path, payload: bytes) -> bool:
    """Write a month unless a different one is stored; True if the stored month matches."""
    fingerprint = hashlib.sha256(payload).hexdigest()
    if path.exists():
        return stored_fingerprint(path) == fingerprint
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    with gzip.GzipFile(temp, "wb", mtime=0) as handle:
        handle.write(payload)
    temp.replace(path)
    return True


def fetch_site(
    root: Path,
    get: Getter,
    site: str = SITE,
    months: list[str] | None = None,
    say: Callable[[str], None] = print,
) -> int:
    months = months or months_between(FIRST_MONTH, LAST_MONTH)
    rows, failed, empty = [], False, []
    for month in months:
        try:
            items = fetch_month(site, month, get)
        except CountMismatch as error:
            say(f"FAILED: {error}")
            failed = True
            continue
        if not items:
            empty.append(month)
            continue
        payload = canonical_bytes(items)
        if not store_month(month_path(root, site, month), payload):
            say(f"FAILED: {site} {month}: a different month file is already stored; kept")
            failed = True
            continue
        rows.append((month, len(items), hashlib.sha256(payload).hexdigest()))
        say(f"{site} {month}: {len(items)} sessions")
    say(f"{site}: months without sessions: {', '.join(empty) if empty else 'none'}")
    if failed:
        say(f"{site}: fingerprints table not written because a month failed")
        return FAILED
    table = Path(root) / "data" / "raw" / "acn" / f"{site}-fingerprints.csv"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["month", "sessions", "sha256"])
        writer.writerows(rows)
    total = sum(count for _, count, _ in rows)
    say(f"{site}: {total} sessions in {len(rows)} stored months, {months[0]} to {months[-1]}")
    return PASSED


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Download ACN-Data office001 sessions.")
    parser.add_argument("--root", type=Path, default=TOP, help="project top folder")
    args = parser.parse_args(argv)
    try:
        get = http_getter(read_token())
        return fetch_site(args.root, get)
    except FetchError as error:
        print(f"COULD NOT RUN: {error}")
        return COULD_NOT_RUN


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
