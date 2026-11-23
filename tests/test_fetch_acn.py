"""The ACN-Data tools `tools/fetch_acn.py` and `tools/acn_fingerprints.py`, on made-up sessions.

No network is used and no real token is read: a fake API reply stands in for the server.
"""

import base64
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import urllib.error
from email.utils import format_datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch_acn = load("fetch_acn")
acn_fingerprints = load("acn_fingerprints")

FAKE_TOKEN = "fake-token-QX93"


@pytest.fixture(autouse=True)
def no_network_and_no_real_token(monkeypatch, tmp_path_factory):
    """Any attempt to reach the network fails the test; the token file is never the real one."""

    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to use the network")

    monkeypatch.setattr(fetch_acn.urllib.request, "urlopen", refuse)
    missing = tmp_path_factory.mktemp("no-token") / "absent"
    monkeypatch.setattr(fetch_acn, "TOKEN_FILE", str(missing))


def sessions(month: str, count: int) -> list[dict]:
    return [
        {
            "_id": f"{month}-{i:04d}",
            "kWhDelivered": 4.25 + i,
            "userID": f"USER-VALUE-{i}",
            "_links": {"self": "x"},
        }
        for i in range(count)
    ]


class FakeApi:
    """Answers month queries from a dict month -> sessions, in pages, like the real API."""

    def __init__(self, by_month: dict[str, list[dict]], report_extra: int = 0):
        self.by_month, self.report_extra, self.urls = by_month, report_extra, []

    def __call__(self, url: str) -> dict:
        self.urls.append(url)
        query = parse_qs(urlparse(url).query)
        where, page = query["where"][0], int(query["page"][0])
        month = next(
            m for m in fetch_acn.months_between("2000-01", "2030-12") if self._has(where, m)
        )
        items = [dict(item) for item in self.by_month.get(month, [])]
        size = fetch_acn.PAGE_SIZE
        batch = items[(page - 1) * size : page * size]
        for item in batch:
            item["_links"] = {"self": "x"}
        return {"_items": batch, "_meta": {"total": len(items) + self.report_extra}}

    @staticmethod
    def _has(where: str, month: str) -> bool:
        start, _ = fetch_acn.month_window(month)
        return f'connectionTime>="{format_datetime(start, usegmt=True)}"' in where


def run_site(root, api, months):
    lines: list[str] = []
    status = fetch_acn.fetch_site(root, api, months=months, say=lines.append)
    return status, lines


def test_month_query_uses_a_utc_window_with_english_names():
    url = fetch_acn.month_query("office001", "2019-12", 2)
    where = parse_qs(urlparse(url).query)["where"][0]
    assert where == (
        'connectionTime>="Sun, 01 Dec 2019 00:00:00 GMT" and '
        'connectionTime<"Wed, 01 Jan 2020 00:00:00 GMT"'
    )
    assert url.startswith("https://ev.caltech.edu/api/v1/sessions/office001?")


def test_months_between_includes_both_ends():
    assert fetch_acn.months_between("2018-11", "2019-02") == [
        "2018-11",
        "2018-12",
        "2019-01",
        "2019-02",
    ]


def test_pages_are_followed_and_month_is_stored_in_the_acn_layout(tmp_path):
    api = FakeApi({"2019-03": sessions("2019-03", 230)})
    status, lines = run_site(tmp_path, api, ["2019-03"])
    assert status == 0
    path = tmp_path / "data/raw/acn/office001/2019-03.json.gz"
    raw = gzip.decompress(path.read_bytes())
    items = json.loads(raw)
    assert len(items) == 230
    assert all("_links" not in item for item in items)
    assert raw == json.dumps(items, sort_keys=True, separators=(",", ":")).encode()
    assert len(api.urls) == 3
    assert "office001 2019-03: 230 sessions" in lines


def test_fingerprint_table_has_the_caltech_layout(tmp_path):
    api = FakeApi({"2019-03": sessions("2019-03", 3), "2019-05": sessions("2019-05", 2)})
    status, lines = run_site(tmp_path, api, ["2019-03", "2019-04", "2019-05"])
    assert status == 0
    table = tmp_path / "data/raw/acn/office001-fingerprints.csv"
    rows = list(csv.reader(table.read_text().splitlines()))
    assert rows[0] == ["month", "sessions", "sha256"]
    assert [row[:2] for row in rows[1:]] == [["2019-03", "3"], ["2019-05", "2"]]
    raw = gzip.decompress((tmp_path / "data/raw/acn/office001/2019-03.json.gz").read_bytes())
    assert rows[1][2] == hashlib.sha256(raw).hexdigest()
    assert "office001: months without sessions: 2019-04" in lines
    assert lines[-1] == "office001: 5 sessions in 2 stored months, 2019-03 to 2019-05"


def test_same_download_gives_the_same_bytes(tmp_path):
    api = FakeApi({"2019-03": sessions("2019-03", 7)})
    run_site(tmp_path / "a", api, ["2019-03"])
    run_site(tmp_path / "b", api, ["2019-03"])
    part = "data/raw/acn/office001/2019-03.json.gz"
    assert (tmp_path / "a" / part).read_bytes() == (tmp_path / "b" / part).read_bytes()


def test_no_session_field_is_printed(tmp_path):
    _, lines = run_site(tmp_path, FakeApi({"2019-03": sessions("2019-03", 5)}), ["2019-03"])
    printed = "\n".join(lines)
    assert "USER-VALUE" not in printed
    assert "4.25" not in printed
    assert "2019-03-0001" not in printed


def test_count_short_of_the_server_total_fails_and_writes_no_table(tmp_path):
    api = FakeApi({"2019-03": sessions("2019-03", 5)}, report_extra=1)
    status, lines = run_site(tmp_path, api, ["2019-03"])
    assert status == 1
    assert (
        lines[0]
        == "FAILED: office001 2019-03: received 5 sessions (5 distinct), the server reports 6"
    )
    assert not (tmp_path / "data/raw/acn/office001-fingerprints.csv").exists()


def test_stored_month_that_differs_is_kept_and_fails(tmp_path):
    run_site(tmp_path, FakeApi({"2019-03": sessions("2019-03", 5)}), ["2019-03"])
    path = tmp_path / "data/raw/acn/office001/2019-03.json.gz"
    before = path.read_bytes()
    status, lines = run_site(tmp_path, FakeApi({"2019-03": sessions("2019-03", 6)}), ["2019-03"])
    assert status == 1
    assert path.read_bytes() == before
    assert "already stored" in lines[0]


# --- the token -----------------------------------------------------------------------------------


def http_error(code):
    def urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, code, "refused", {}, io.BytesIO(b""))

    return urlopen


def test_refused_token_reports_the_status_code_only():
    get = fetch_acn.http_getter(FAKE_TOKEN, urlopen=http_error(401), pause=0.0)
    with pytest.raises(fetch_acn.FetchError) as caught:
        get(fetch_acn.month_query("office001", "2019-03", 1))
    message = str(caught.value)
    assert message == "HTTP 401 from the ACN-Data API"
    assert caught.value.__cause__ is None and caught.value.__suppress_context__


_REAL_GETTER = fetch_acn.http_getter


def _refusing(token):
    return _REAL_GETTER(token, urlopen=http_error(403), pause=0.0)


def test_main_prints_nothing_of_the_token(tmp_path, capsys, monkeypatch):
    token_file = tmp_path / "token"
    token_file.write_text(FAKE_TOKEN + "\n")
    monkeypatch.setattr(fetch_acn, "TOKEN_FILE", str(token_file))
    monkeypatch.setattr(fetch_acn, "http_getter", _refusing)
    status = fetch_acn.main(["--root", str(tmp_path)])
    captured = capsys.readouterr()
    assert status == 2
    encoded = base64.b64encode(f"{FAKE_TOKEN}:".encode()).decode()
    for secret in (FAKE_TOKEN, encoded, encoded.rstrip("=")):
        assert secret not in captured.out + captured.err
    assert "COULD NOT RUN: HTTP 403 from the ACN-Data API" in captured.out


def test_the_token_goes_only_into_the_authorization_header():
    seen = []

    def urlopen(request, timeout):
        seen.append(request)
        return io.BytesIO(b'{"_items": [], "_meta": {"total": 0}}')

    get = fetch_acn.http_getter(FAKE_TOKEN, urlopen=urlopen, pause=0.0)
    get(fetch_acn.month_query("office001", "2019-03", 1))
    (request,) = seen
    assert FAKE_TOKEN not in request.full_url
    headers = {name.lower(): value for name, value in request.header_items()}
    assert set(headers) == {"authorization", "user-agent"}
    assert headers["user-agent"] == "data-fetch (python urllib)"


def test_missing_token_file_could_not_run(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(fetch_acn, "TOKEN_FILE", str(tmp_path / "none"))
    assert fetch_acn.main(["--root", str(tmp_path)]) == 2
    assert "no readable ACN-Data token file" in capsys.readouterr().out


# --- fingerprints check --------------------------------------------------------------------------


def stored_site(tmp_path):
    api = FakeApi({"2019-03": sessions("2019-03", 3), "2019-04": sessions("2019-04", 2)})
    run_site(tmp_path, api, ["2019-03", "2019-04"])
    return tmp_path


def check(root, *sites):
    lines: list[str] = []
    results = [acn_fingerprints.check_site(root, site, say=lines.append) for site in sites]
    return results, lines


def test_fingerprints_match(tmp_path):
    results, lines = check(stored_site(tmp_path), "office001")
    assert results == [0]
    assert lines == [
        "office001: 2 months match their fingerprints (2019-03 to 2019-04, 5 sessions in the table)"
    ]


def test_fingerprint_differs_fails(tmp_path):
    root = stored_site(tmp_path)
    path = root / "data/raw/acn/office001/2019-04.json.gz"
    path.write_bytes(gzip.compress(b"[]", mtime=0))
    results, lines = check(root, "office001")
    assert results == [1]
    assert lines == ["FAILED: office001 2019-04: fingerprint differs from the table"]


def test_unlisted_month_fails(tmp_path):
    root = stored_site(tmp_path)
    (root / "data/raw/acn/office001/2019-09.json.gz").write_bytes(gzip.compress(b"[]"))
    assert check(root, "office001")[0] == [1]


def test_missing_month_or_table_could_not_run(tmp_path):
    root = stored_site(tmp_path)
    (root / "data/raw/acn/office001/2019-03.json.gz").unlink()
    assert check(root, "office001")[0] == [2]
    assert check(root, "caltech")[0] == [2]
