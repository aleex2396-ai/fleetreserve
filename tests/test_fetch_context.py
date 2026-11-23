"""The context table of `tools/fetch_data.py --context` and the context role of
`tools/data_hashes.py`, on made-up bytes; no network is used.

The committed table is checked against `fleetreserve.context_places`: one weather file per place,
each URL the one `weather_url` gives, and every row with a committed size and SHA-256.
"""

import hashlib
import importlib.util
import io
from pathlib import Path

import pytest

from fleetreserve import context_places

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch_data = _load("fetch_data")
data_hashes = _load("data_hashes")

CONTENT = b"time,temperature_2m\n2019-01-01T00:00,1.5\n"
SHA = hashlib.sha256(CONTENT).hexdigest()
SIZE = len(CONTENT)
HEADER = "set,name,role,bytes,sha256,url\n"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to use the network")

    monkeypatch.setattr(fetch_data.urllib.request, "urlopen", refuse)


def served(url):
    return io.BytesIO(CONTENT)


def context_source(sha256=SHA, size=SIZE):
    return fetch_data.Source("weather", "x.csv", "context", size, None, "https://x/y", sha256)


def run(root, sources):
    lines: list[str] = []
    status = fetch_data.fetch_all(root, sources, opener=served, say=lines.append, pause=0.0)
    return status, lines


def test_context_file_goes_to_the_raw_folder_with_its_sha256_checked(tmp_path):
    status, lines = run(tmp_path, [context_source()])
    assert status == 0
    assert (tmp_path / "data/raw/weather/x.csv").read_bytes() == CONTENT
    assert "SHA-256 as committed" in lines[0]


def test_wrong_sha256_is_failed_and_nothing_is_kept(tmp_path):
    status, lines = run(tmp_path, [context_source(sha256="0" * 64)])
    assert status == 1
    assert "SHA-256 differs" in lines[0]
    assert not (tmp_path / "data/raw/weather/x.csv").exists()


def test_existing_file_with_another_sha256_is_failed_and_kept(tmp_path):
    target = tmp_path / "data/raw/weather/x.csv"
    target.parent.mkdir(parents=True)
    other = CONTENT.replace(b"1.5", b"2.5")
    target.write_bytes(other)
    status, lines = run(tmp_path, [context_source()])
    assert status == 1
    assert target.read_bytes() == other


def test_context_table_reads_sha256_and_refuses_other_roles(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text(HEADER + f"weather,a.csv,context,12,{SHA.upper()},https://x\n")
    (only,) = fetch_data.read_context_table(path)
    assert (only.size, only.sha256, only.md5) == (12, SHA, None)
    for role in ("held-out", "train", "document"):
        path.write_text(HEADER + f"weather,a.csv,{role},12,{SHA},https://x\n")
        with pytest.raises(fetch_data.TableError):
            fetch_data.read_context_table(path)


def test_context_table_refuses_a_malformed_sha256(tmp_path):
    path = tmp_path / "t.csv"
    for bad in ("abc", "g" * 64):
        path.write_text(HEADER + f"weather,a.csv,context,12,{bad},https://x\n")
        with pytest.raises(fetch_data.TableError):
            fetch_data.read_context_table(path)


@pytest.fixture(scope="module")
def table():
    return fetch_data.read_context_table(fetch_data.CONTEXT_TABLE)


def test_committed_table_has_one_weather_file_per_place_with_its_query(table):
    weather = {s.name: s for s in table if s.set_name == context_places.WEATHER_SET}
    assert set(weather) == {context_places.weather_file(p) for p in context_places.PLACES}
    for place in context_places.PLACES:
        assert weather[context_places.weather_file(place)].url == context_places.weather_url(place)


def test_committed_table_pins_the_oxcgrt_file_to_one_commit(table):
    (oxcgrt,) = [s for s in table if s.set_name == "oxcgrt"]
    assert oxcgrt.url.startswith(
        "https://raw.githubusercontent.com/OxCGRT/covid-policy-dataset/"
        "e7f66ee39654293b5c068efd2f195bd591dc27f6/"
    )
    assert oxcgrt.size == 2_527_943


def test_committed_table_has_a_size_and_sha256_for_every_row(table):
    school = [s for s in table if s.set_name == "school"]
    assert len(table) == len(context_places.PLACES) + 1 + len(school)
    for s in table:
        assert s.role == "context" and s.size and s.sha256 and len(s.sha256) == 64


def test_context_files_have_the_context_role_in_the_hash_list():
    assert data_hashes.role_of("data/raw/weather/oslo.csv") == "context"
    assert data_hashes.role_of("data/raw/oxcgrt/a.csv") == "context"
    assert data_hashes.role_of("data/raw/school/oslo-2019-20.html") == "context"
    assert "context" in data_hashes.ROLE_NAMES
    assert data_hashes.role_of("data/raw/weathers/oslo.csv") is None
