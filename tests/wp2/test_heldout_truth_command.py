"""The command `heldout_truth_build` on made-up sealed repositories (`fake_heldout.sealed_repo`).

1. At the tag and clean, it opens each of the three sealed files exactly once, in the order
   Norway, a.s.r., Korea (three lines of the openings log), writes the tables of every unit with a
   window into its default folder under the ignored `data/` and the summary and report into the
   run folder; each table equals `truth.build` on the loaders' sessions; the summary holds the
   Korean fast-charger count. The tree stays clean, so a second run opens again, and gives the
   same bytes; the held setting writes its own folder with the same windows.
2. Run as `python -m fleetreserve.heldout_truth_build` from the made-up repository's own copy of
   the package, it gives the same bytes as the imported command.
3. Every check before the first opening refuses with exit 2, opening nothing (no log line) and
   writing nothing: an output folder that is tracked, untracked and not ignored, the repository
   itself, the training and development build's folders, inside the sealed folder, a file,
   holding the other setting's tables, or not writable; a run folder inside the repository and
   not ignored, or missing; a refused matrix.
4. A refusal of the opener (no tag; an untracked file) exits 2 with nothing written and no line
   logged.
5. A layout failure of one set exits 1, names the set, writes nothing for any set, leaves the
   openings made logged and opens no later set.
6. A window search that cycles exits 0, writes the unit's tables at the first τ met twice and
   lists the cycle among the findings; any other error is raised.
7. The command's own source names no path of the sealed folder and calls nothing that opens,
   lists, reads, stats or hashes a file; of the loader it uses only the three gated routes, the
   refusal it raises and the way it runs git; of the seal module only the folder's name.
"""

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from fake_heldout import ASR_FILE, commit, git, sealed_repo
from fake_korea import KOREA_FILE
from heldout_truth_fake import (
    NORWAY_START,
    TRO_R_DAYS,
    asr_bytes,
    cycle_rows,
    korea_bytes,
    norway_bytes,
    norway_days,
    norway_rows,
    sealed_files,
)

import fleetreserve
from fleetreserve import (
    heldout_loader,
    load_asr,
    load_heldout_norway,
    load_korea,
    truth,
    truth_build,
)
from fleetreserve import heldout_truth_build as h
from fleetreserve.sessions import LayoutError
from fleetreserve.windows import WORKPLACE, Window, site_days

MATRIX = Path(fleetreserve.__file__).resolve().parents[2] / "gate" / "matrix.json"
WRITTEN = [
    "asr-before-march-2023",
    "asr-from-march-2023",
    "OSL_T",
    "TRO_R",
    "korea-home",
    "korea-work",
    "korea-public-parking",
]


def _hashes(folder: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}


def _run(root, monkeypatch, run_dir, *args):
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    return h.main(["--matrix", str(MATRIX), *args], root=root)


def _status(root) -> str:
    return git(root, "status", "--porcelain", "--untracked-files=normal")


@pytest.fixture
def repo(tmp_path):
    return sealed_repo(tmp_path, sealed_files(tmp_path / "made"))


@pytest.fixture(scope="module")
def unopened(tmp_path_factory):
    """One made-up repository on which no test may open anything."""
    folder = tmp_path_factory.mktemp("unopened")
    root = sealed_repo(folder, sealed_files(folder / "made"))
    (root / "data" / "derived" / "a-file").parent.mkdir(parents=True)
    (root / "data" / "derived" / "a-file").write_text("x")
    other = root / "data" / "derived" / "truth-heldout"
    other.mkdir()
    (other / truth_build.SETTING_FILE).write_text("held\n")
    return root


NOT_IGNORED = "{out} lies inside the repository and git does not ignore it"
TRAINING = "{out} is the training and development build's folder"
SEALED = "the output folder {out} lies inside the sealed folder"


@pytest.mark.parametrize(
    "out, why",
    [
        ("src/fleetreserve", NOT_IGNORED),
        ("src/new-folder", NOT_IGNORED),
        ("results", NOT_IGNORED),
        ("src/../results", NOT_IGNORED),  # printed as written
        (".", NOT_IGNORED),
        ("data/derived/truth", TRAINING),
        ("data/derived/truth-held", TRAINING),
        ("data/../data/derived/truth", TRAINING),  # printed as written
        ("data/heldout/derived", SEALED),
        ("data/heldout", SEALED),
        ("data/derived/a-file", "{out} is not a folder"),
        ("data/../data/derived/a-file", "{out} is not a folder"),  # printed as written
        ("data/derived/a-file/below", "{out} cannot be made or written"),
        ("data/derived/truth-heldout", "{resolved} already holds tables of the held setting"),
    ],
)
def test_an_output_folder_that_cannot_finish_is_refused_before_any_opening(
    unopened, tmp_path, monkeypatch, capsys, out, why
):
    before = _status(unopened)
    given = unopened / out
    code = _run(unopened, monkeypatch, tmp_path / "run", "--out", str(given))
    message = why.format(out=given, resolved=given.resolve())
    assert (code, capsys.readouterr().out) == (2, f"COULD NOT RUN: {message}; nothing opened\n")
    assert heldout_loader.logged_openings(unopened) == []
    assert not (unopened / "results").exists() and not (unopened / "src" / "new-folder").exists()
    assert not (unopened / "data" / "heldout" / "derived").exists()
    assert list((tmp_path / "run").iterdir()) == []
    assert _status(unopened) == before == ""


def test_other_refusals_before_any_opening(unopened, tmp_path, monkeypatch, capsys):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    out = tmp_path / "o"
    in_sealed = unopened / "data" / "heldout" / "run-in-sealed"
    runs = unopened / "runs"
    runs_written = unopened / "src" / ".." / "runs"  # a run folder is printed as written
    missing = tmp_path / "gone" / ".." / "missing"
    dangling = tmp_path / "dangling"  # a link to nothing: no folder can be made below it
    dangling.symlink_to(tmp_path / "nowhere")
    try:
        cases = [
            (locked / "out", tmp_path / "run", f"{locked / 'out'} cannot be made or written"),
            (dangling / "out", tmp_path / "run", f"{dangling / 'out'} cannot be made or written"),
            (
                out,
                runs_written,
                f"the run folder {runs_written} lies inside the repository, not ignored",
            ),
            (out, missing, f"the run folder {missing} is not a writable folder"),
            (out, in_sealed, f"the run folder {in_sealed} lies inside the sealed folder"),
        ]
        for given, run_dir, message in cases:
            monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(run_dir))
            if run_dir not in (missing, in_sealed):
                run_dir.mkdir(exist_ok=True)
            code = h.main(["--matrix", str(MATRIX), "--out", str(given)], root=unopened)
            printed = capsys.readouterr().out
            assert (code, printed) == (2, f"COULD NOT RUN: {message}; nothing opened\n")
            assert not in_sealed.exists()
            if runs.exists():
                runs.rmdir()
    finally:
        locked.chmod(0o700)
    bad = tmp_path / "matrix.json"
    bad.write_text("{}")
    monkeypatch.delenv("FLEETRESERVE_RUN_DIR")
    assert h.main(["--matrix", str(bad), "--out", str(out)], root=unopened) == 2
    printed = capsys.readouterr().out
    assert printed.startswith("COULD NOT RUN: the run matrix is refused: ")
    assert printed.endswith("; nothing opened\n")
    assert h.main(["--out", str(out)], root=unopened) == 2  # no gate/matrix.json at the root
    printed = capsys.readouterr().out
    assert printed.startswith("COULD NOT RUN: the run matrix is refused: the matrix file cannot")
    assert str(unopened / "gate" / "matrix.json") in printed
    assert heldout_loader.logged_openings(unopened) == []
    assert not out.exists()


@pytest.mark.parametrize("mode, kind", [(0o500, "not writable"), (0o600, "not searchable")])
def test_a_run_folder_that_cannot_be_written_into_is_refused(unopened, tmp_path, mode, kind):
    out = unopened / "data" / "derived" / "fine"
    folder = tmp_path / kind.replace(" ", "-")
    folder.mkdir()
    a_file = tmp_path / "a-file"
    a_file.write_text("x")
    a_file.chmod(0o700)  # a file the user may write and run is still no folder
    folder.chmod(mode)
    try:
        for run_dir in (folder, a_file):
            with pytest.raises(h.CouldNotRun) as refused:
                h.preflight(unopened, out, truth.MAIN, str(run_dir))
            assert str(refused.value) == f"the run folder {run_dir} is not a writable folder"
    finally:
        folder.chmod(0o700)
    assert not out.exists()


def test_the_command_line_and_its_help(unopened, monkeypatch, capsys):
    monkeypatch.setenv("COLUMNS", "200")
    with pytest.raises(SystemExit) as stopped:
        h.main(["--help"], root=unopened)
    assert stopped.value.code == 0
    printed = capsys.readouterr().out
    assert printed.startswith(
        "usage: "
    ) and "[-h] [--setting {main,held}] [--out OUT] [--matrix MATRIX]" in (printed)
    assert (
        "\n\nBuild the truth tables of the held-out units from the sealed files (WP2; first step "
        "of WP14).\n\n"
    ) in printed
    assert re.search(r"\n  --out OUT +default: by setting \(OUT_NAMES\)\n", printed)
    assert re.search(r"\n  --matrix MATRIX +default: gate/matrix.json\n", printed)
    with pytest.raises(SystemExit) as stopped:
        h.main(["--setting", "other"], root=unopened)
    assert stopped.value.code == 2
    assert "invalid choice: 'other'" in capsys.readouterr().err
    assert heldout_loader.logged_openings(unopened) == []


def test_both_folders_inside_the_sealed_folder_name_the_output_folder(unopened):
    out = unopened / "data" / "heldout" / "out"
    run_dir = unopened / "data" / "heldout" / "run"
    with pytest.raises(h.CouldNotRun) as refused:
        h.preflight(unopened, out, truth.MAIN, str(run_dir))
    assert str(refused.value) == f"the output folder {out} lies inside the sealed folder"


def test_an_output_folder_below_a_file_or_an_unsearchable_folder_is_refused(unopened, tmp_path):
    program = tmp_path / "program"  # a file the user may write and run is still no folder
    program.write_text("x")
    program.chmod(0o700)
    unsearchable = tmp_path / "unsearchable"
    unsearchable.mkdir()
    unsearchable.chmod(0o600)
    try:
        with pytest.raises(h.CouldNotRun) as refused:
            h.preflight(unopened, program / "out", truth.MAIN, None)
        assert str(refused.value) == f"{program / 'out'} cannot be made or written"
        # in or below it nothing can be looked at: refused by the setting check, in its own words
        for out in (unsearchable, unsearchable / "out"):
            with pytest.raises(h.CouldNotRun, match=r"^\[Errno 13\] Permission denied"):
                h.preflight(unopened, out, truth.MAIN, None)
    finally:
        unsearchable.chmod(0o700)


def test_a_folder_holding_tables_of_the_same_setting_is_accepted(unopened):
    held = unopened / "data" / "derived" / "truth-heldout"  # marked as the held setting
    assert h.preflight(unopened, held, truth.HELD, None) is None
    with pytest.raises(h.CouldNotRun):
        h.preflight(unopened, held, truth.MAIN, None)


def test_folders_that_git_ignores_pass_whatever_git_variables_the_caller_has(
    unopened, tmp_path, monkeypatch
):
    """The ignore check runs git as the opener does, without the caller's `GIT_` variables."""
    out = unopened / "data" / "derived" / "fine"
    run_dir = unopened / "data" / "derived" / "run-fine"
    run_dir.mkdir()
    try:
        monkeypatch.setenv("GIT_DIR", str(tmp_path / "no-repository"))
        monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path))
        assert h.preflight(unopened, out, truth.MAIN, str(run_dir)) is None
    finally:
        run_dir.rmdir()
    assert not out.exists()


def test_the_findings_follow_the_order_of_the_units(tmp_path, monkeypatch):
    """a.s.r., Norway, Korea, as in `UNITS` and in the report's CHECKS section."""
    result = h.build_heldout(_in_memory(tmp_path), truth.MAIN, h.SplitRule(1, 100, 200, "m"))
    monkeypatch.setattr(h.data_checks, "findings", lambda source, _checks: [source])
    assert h.findings(result)[:3] == ["asr", "norway", "korea"]


@pytest.mark.parametrize("problem", ["no tag", "an untracked file"])
def test_a_refusal_of_the_opener_writes_nothing_and_logs_nothing(
    tmp_path, monkeypatch, capsys, problem
):
    root = sealed_repo(tmp_path, sealed_files(tmp_path / "made"), tag=problem != "no tag")
    if problem == "an untracked file":
        (root / "notes.txt").write_text("x")
    run_dir = tmp_path / "run"
    assert _run(root, monkeypatch, run_dir) == 2
    printed = capsys.readouterr().out
    with pytest.raises(heldout_loader.SealRefused) as refused:  # the opener's own words
        heldout_loader.norway_sites(root)
    assert printed == (
        f"COULD NOT RUN: the norway file was refused by the opener: {refused.value}; openings "
        "logged by this run: 0; nothing written\n"
    )
    assert heldout_loader.logged_openings(root) == []
    assert not (root / "data" / "derived").exists()
    assert list(run_dir.iterdir()) == []


@pytest.mark.parametrize(
    "broken, opened", [("asr", ["norway", "asr"]), ("korea", ["norway", "asr", "korea"])]
)
def test_a_layout_failure_of_one_set_writes_nothing_for_any_set(
    tmp_path, monkeypatch, capsys, broken, opened
):
    files = sealed_files(tmp_path / "made")
    if broken == "asr":
        files[ASR_FILE] = asr_bytes(header=load_asr_header_without("total_energy"))
    else:
        files[KOREA_FILE] = korea_bytes().replace(b"Apartment", b"Somewhere")
    root = sealed_repo(tmp_path, files)
    run_dir = tmp_path / "run"
    assert _run(root, monkeypatch, run_dir) == 1
    printed = capsys.readouterr().out
    parse, name = (
        (load_asr.from_bytes, ASR_FILE) if broken == "asr" else (load_korea.from_bytes, KOREA_FILE)
    )
    with pytest.raises(LayoutError) as layout:  # the parser's own words
        parse(files[name])
    assert printed == (
        f"FAILED: the {broken} file does not have its published layout: {layout.value}; nothing "
        f"written for any set; openings logged by this run: {len(opened)}\n"
    )
    assert [line["set"] for line in heldout_loader.logged_openings(root)] == opened
    assert not (root / "data" / "derived").exists()
    assert list(run_dir.iterdir()) == []
    assert _status(root) == ""


def load_asr_header_without(name):
    from fake_heldout import ASR_HEADER

    return tuple(column for column in ASR_HEADER if column != name)


def _in_memory(tmp_path):
    files = sealed_files(tmp_path / "made")
    return {
        "norway": load_heldout_norway.from_bytes(files[heldout_loader.NORWAY]),
        "asr": load_asr.from_bytes(files[ASR_FILE]),
        "korea": load_korea.from_bytes(files[KOREA_FILE]),
    }


def test_a_cycle_exits_0_and_any_other_error_is_raised(tmp_path, monkeypatch, capsys):
    root = tmp_path / "root"  # not a repository: the output lies outside it, no git is asked
    root.mkdir()
    loaded = _in_memory(tmp_path)
    monkeypatch.setattr(h, "load_sets", lambda _root: loaded)
    monkeypatch.delenv("FLEETRESERVE_RUN_DIR", raising=False)
    out = tmp_path / "out"
    args = ["--matrix", str(MATRIX), "--out", str(out)]
    assert h.main(args, root=root) == 0  # the made-up sets build without the opener
    assert len(list(out.iterdir())) == 2 * len(WRITTEN)
    assert capsys.readouterr().out.startswith("FINDINGS")
    rows = cycle_rows()[0] + norway_rows("TRO_R", norway_days(NORWAY_START, TRO_R_DAYS))
    cycling = {**loaded, "norway": load_heldout_norway.from_bytes(norway_bytes(tmp_path, rows))}
    monkeypatch.setattr(h, "load_sets", lambda _root: cycling)
    cycled = tmp_path / "cycled"
    assert h.main(["--matrix", str(MATRIX), "--out", str(cycled)], root=root) == 0
    printed = capsys.readouterr().out
    assert "\n- OSL_T: the window search cycles through τ [14, 10, 14]: τ 14:00, the first " in (
        printed
    )
    assert "\n- OSL_T: τ 14:00, S 15:00 to 17:00 (2 hours); median peak 10.0 cars " in printed
    assert (cycled / "OSL_T-days.csv").exists() and len(list(cycled.iterdir())) == 2 * len(WRITTEN)
    monkeypatch.setattr(h, "load_sets", lambda _root: loaded)
    fresh = tmp_path / "fresh"

    def full(*_args):
        raise OSError("made-up full disk")

    monkeypatch.setattr(h.truth_build, "write", full)
    assert h.main(["--matrix", str(MATRIX), "--out", str(fresh)], root=root) == 2
    assert capsys.readouterr().out == (
        "COULD NOT RUN: cannot write the tables: made-up full disk\n"
    )

    def broken(*_args):
        raise ValueError("made-up fault")

    monkeypatch.setattr(h, "build_heldout", broken)
    with pytest.raises(ValueError, match="made-up fault"):
        h.main(["--matrix", str(MATRIX), "--out", str(fresh)], root=root)
    assert not fresh.exists()


EXPECTED = Path(__file__).parent / "heldout_truth_expected"


def written_texts(tmp_path, setting: str) -> dict[str, str]:
    """The summary and report of one run on the made-up sets, the matrix at its default place.

    The caller replaces `h.load_sets` with the in-memory sets (no opener, no repository)."""
    root = tmp_path / f"root-{setting}"
    (root / "gate").mkdir(parents=True)
    (root / "gate" / "matrix.json").write_bytes(MATRIX.read_bytes())
    run_dir = tmp_path / f"run-{setting}"
    run_dir.mkdir()
    os.environ["FLEETRESERVE_RUN_DIR"] = str(run_dir)
    args = ["--setting", setting, "--out", str(tmp_path / f"out-{setting}")]
    assert h.main(args, root=str(root)) == 0  # the top folder may be given as text
    assert h.main(args, root=root) == 0  # again into the same folder: the same setting passes
    return {
        f"{setting}-{name}": (run_dir / name).read_text()
        for name in (truth_build.SUMMARY_NAME, truth_build.REPORT_NAME)
    }


@pytest.mark.parametrize("setting", truth.SETTINGS)
def test_the_summary_and_the_report_are_the_expected_texts(tmp_path, monkeypatch, capsys, setting):
    """Every line of both, against texts read once by eye (the numbers are checked above)."""
    loaded = _in_memory(tmp_path)
    monkeypatch.setattr(h, "load_sets", lambda _root: loaded)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", "")
    for name, text in written_texts(tmp_path, setting).items():
        assert text == (EXPECTED / name).read_text(), name
    report = (EXPECTED / f"{setting}-{truth_build.REPORT_NAME}").read_text()
    assert capsys.readouterr().out == report * 2  # both runs print the same report


def test_a_later_commit_on_top_of_the_tag_is_refused(repo, tmp_path, monkeypatch, capsys):
    (repo / "extra.txt").write_text("x")
    commit(repo, "a later commit")
    assert _run(repo, monkeypatch, tmp_path / "run") == 2
    assert "commit after the tag" in capsys.readouterr().out
    assert heldout_loader.logged_openings(repo) == []


FILE_CALLS = {
    "open",
    "read_bytes",
    "read_text",
    "listdir",
    "scandir",
    "glob",
    "rglob",
    "iterdir",
    "walk",
    "stat",
    "lstat",
    "hashlib",
    "sha256",
    "md5",
    "shutil",
    "copyfile",
    "open_sealed",
    "sealed_path",
    "sealed_files",
    "file_digest",
    "logged_openings",
}


def test_the_command_reaches_sealed_data_only_through_the_gated_routes():
    source = Path(h.__file__).read_text(encoding="utf-8")
    assert "data/heldout" not in source and "heldout/" not in source
    tree = ast.parse(source)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not (names | imported) & FILE_CALLS

    def used(module):
        return {
            n.attr
            for n in ast.walk(tree)
            if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name)
            and n.value.id == module
        }

    assert used("heldout_loader") == {
        "norway_sites",
        "asr",
        "korea",
        "SealRefused",
        "GIT_SETTINGS",
        "git_environment",
    }
    assert used("heldout_seal") == {"FOLDER"}
    routes = (
        ("norway", heldout_loader.norway_sites),
        ("asr", heldout_loader.asr),
        ("korea", heldout_loader.korea),
    )
    assert routes == h.ROUTES
    assert h.SETS == ("norway", "asr", "korea")


# The two slowest tests come last, so that a run stopped at its first failure (as a sweep of
# one-line breaks runs it) meets the fast tests first.


def test_one_run_opens_each_set_once_and_writes_tables_equal_to_truth_build(
    repo, tmp_path, monkeypatch, capsys
):
    run_dir = tmp_path / "run"
    assert _run(repo, monkeypatch, run_dir) == 0
    openings = heldout_loader.logged_openings(repo)
    assert [line["set"] for line in openings] == ["norway", "asr", "korea"]
    out = repo / "data" / "derived" / "truth-heldout"
    names = sorted(f"{u}-{kind}.csv" for u in WRITTEN for kind in ("cars", "days"))
    assert sorted(p.name for p in out.iterdir()) == names
    summary = json.loads((run_dir / truth_build.SUMMARY_NAME).read_text())
    assert summary["korea_fast_charger_rows_removed"] == 3
    assert summary["units"]["korea-home"]["rows_removed"] == {"fast charger (ChargerType 1)": 2}
    assert set(summary["abstained"]) == {"korea-lodging", "korea-visits"}
    assert summary["calibration_rule"] == {
        "seed": 1,
        "n_calibration": 100,
        "half_below": 200,
        "matrix": str(MATRIX),  # as given
    }
    assert "setting" not in summary
    report = (run_dir / truth_build.REPORT_NAME).read_text()
    assert report == capsys.readouterr().out
    assert "KOREA FAST CHARGERS: 3 rows removed" in report
    for unit in WRITTEN:
        assert summary["units"][unit]["sha256"] == {
            kind: hashlib.sha256((out / f"{unit}-{kind}.csv").read_bytes()).hexdigest()
            for kind in ("cars", "days")
        }
    # the tables are truth.build's on the loaders' own sessions of the same bytes
    files = sealed_files(tmp_path / "again")
    norway = load_heldout_norway.from_bytes(files[heldout_loader.NORWAY])[0]
    korea = load_korea.from_bytes(files[KOREA_FILE])[0]
    asr = load_asr.from_bytes(files[ASR_FILE])[0]
    average, kind = truth.session_averages(asr, load_asr.LIMIT_KW)
    rates = truth.user_rates(asr, average, kind)
    for unit in WRITTEN:
        if unit in load_asr.UNITS:
            days = [d for d in site_days(asr["plug_in"]) if load_asr.unit_of(d) == unit]
            tables = truth.build(
                asr,
                WORKPLACE,
                load_asr.ZONE,
                unit=unit,
                days=days,
                limit_kw=load_asr.LIMIT_KW,
                bound_by_charge_fields=False,
                rates=rates,
            )
        else:
            sessions, zone, limit = (
                (norway, h.load_norway.ZONE, h.PHYSICAL_LIMIT_KW["norway"])
                if unit in load_heldout_norway.SITES
                else (korea, load_korea.ZONE, load_korea.LIMIT_KW)
            )
            part = sessions[sessions["site"] == unit]
            entry = summary["windows"][unit]
            window = Window(int(entry["tau"][:2]), entry["service_hours"], "from the summary")
            tables = truth.build(
                part,
                window,
                zone,
                unit=unit,
                days=site_days(part["plug_in"]),
                limit_kw=limit,
                bound_by_charge_fields=False,
            )
        for kind, table in zip(("cars", "days"), tables, strict=True):
            assert (out / f"{unit}-{kind}.csv").read_bytes() == truth_build.csv_bytes(table)
    assert summary["windows"]["korea-home"]["tau"] == "22:00"
    assert summary["windows"]["OSL_T"]["service"] == "17:00 to 23:00"
    # the tree is still clean: a second run opens again and gives the same bytes
    assert _status(repo) == ""
    other = repo / "data" / "derived" / "second"
    assert _run(repo, monkeypatch, tmp_path / "run2", "--out", str(other)) == 0
    assert len(heldout_loader.logged_openings(repo)) == 6
    assert _hashes(other) == _hashes(out)
    assert (tmp_path / "run2" / truth_build.SUMMARY_NAME).read_bytes() == (
        run_dir / truth_build.SUMMARY_NAME
    ).read_bytes()
    # the held setting: its own default folder, the same windows
    assert _run(repo, monkeypatch, tmp_path / "run3", "--setting", "held") == 0
    held = repo / "data" / "derived" / "truth-heldout-held"
    assert (held / truth_build.SETTING_FILE).read_text() == "held\n"
    held_summary = json.loads((tmp_path / "run3" / truth_build.SUMMARY_NAME).read_text())
    assert held_summary["setting"] == "held"
    assert held_summary["windows"] == summary["windows"]
    assert _status(repo) == ""


def test_run_with_python_m_from_the_repository_copy_gives_the_same_bytes(
    repo, tmp_path, monkeypatch
):
    copy = repo / "src" / "fleetreserve" / "heldout_truth_build.py"
    assert copy.read_bytes() == Path(h.__file__).read_bytes()
    environment = {k: v for k, v in os.environ.items() if not k.startswith("FLEETRESERVE_")}
    environment["PYTHONPATH"] = str(repo / "src")
    # as fake_heldout.run_python: no __pycache__ folder in the made-up tree, whose .gitignore
    # holds only data/, whatever the caller's environment (the real repository ignores it)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    outs = {}
    for how in ("module", "imported"):
        out = repo / "data" / "derived" / how
        run_dir = tmp_path / f"run-{how}"
        run_dir.mkdir()
        if how == "module":
            done = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "fleetreserve.heldout_truth_build",
                    "--matrix",
                    str(MATRIX),
                    "--out",
                    str(out),
                ],
                cwd=repo,
                env={**environment, "FLEETRESERVE_RUN_DIR": str(run_dir)},
                capture_output=True,
                text=True,
                check=False,
            )
            assert done.returncode == 0, done.stdout + done.stderr
        else:
            monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(run_dir))
            assert h.main(["--matrix", str(MATRIX), "--out", str(out)], root=repo) == 0
        outs[how] = (_hashes(out), (run_dir / truth_build.SUMMARY_NAME).read_bytes())
    assert outs["module"] == outs["imported"]
    commands = [line["command"] for line in heldout_loader.logged_openings(repo)]
    assert len(commands) == 6
    assert commands[0][0].endswith("heldout_truth_build.py")
