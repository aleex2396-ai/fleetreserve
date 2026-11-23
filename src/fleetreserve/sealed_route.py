"""The route by which commands score held-out units (`learn_run`, `compare_run`, `corner_run`,
which searches the corners of each method's calibrated box on them (plan section 11, O7),
`o3_run`, `ceiling_report`, which measures outcome O2 on them with its every-day option (plan
section 11), and `drift_run`, which runs the drift guard on them (plan section 6, T10);
`gate_run` has it and refuses itself on it), and by which `ledger_run` reads a `learn_run` run
that scored them (plan sections 3 and 13: WP12's dress rehearsal on development units, WP14's
held-out run "from the tag").

What the route is, and what it is not. Eight commands have it today. Five take it through
`matrix_route` or its checks: `gate_run`, which on the route could not run at all (reading S3);
`learn_run`, which scores a held-out unit; `compare_run`, which scores the comparison methods of
plan section 10 on the one split the held-out build drew; `corner_run`, which searches the
corners of each method's calibrated box on that split (plan section 11, O7) and checks the fixed
plan's closed form on held-out test days at its calibrated scale alone (its reading H5); and
`ledger_run`, which scores nothing: it reads a `learn_run` run made through the route and the
scored units' tables from the held-out build (`matrix_route`, `check_split_rule`,
`read_two_builds`, `check_split_days`). The other three,
`o3_run`, `ceiling_report` and `drift_run`, whose units are named on their command lines, take it
through `units_route` (below). The cold start (`cold_start`) makes the same checks, by the names
of `cold_start_gate`. The route starts only at the tag `frozen-plan`, or at a logged parsing fix
after it (the held-out loader's reading H6), with a clean tree and that commit's code running
(`require_gate`): anywhere else, or with a change not committed, a run that names a held-out unit
could not run. Its checks stop ordinary mistakes by an honest operator: a held-out
unit named before the tag, a folder or file named by a slip inside a working copy's `data/`
outside `data/derived/`, a path written with `~` or as a URL, tables of another build or of
another setting, a model or option that would look at a scored unit's test days. They are not a
defence against deliberate evasion (links or copies made on purpose, a second working copy): some
such tricks are refused and others are not, so the checks are no proof that no sealed file was
opened. The proof that the seal held is the plan's own (section 3 and WP13): the committed hashes
of the sealed files, the held-out loader, which opens a sealed file only at the tag `frozen-plan`,
or at a logged parsing fix of it (what that is: the loader's module text), with a clean tree, and
its openings log; nothing here adds to that proof. No other command has the
route. Those with no route yet, the other commands that read truth tables (through
`gate_run.check_inputs`, `gate_run.read_inputs` or `days_from_truth.load_unit_days`) and
`truth_verify`: `asa_diagnosis_run`, `centre_scale_whatif`, `dev_day_checks`,
`dial_choice`, `fleet_size_whatif`, `flexitroid_days`, `forecast_report`, `hazard_tune`,
`learn_tune`, `nine_eighty_run`, `resolve_diagnosis`, `size_resolve_whatif`, `size_whatif`,
`soft_scale_check`, `t7_test` and `truth_verify`.

The held-out units' truth tables are the outputs of `heldout_truth_build`, a build separate from
the training and development build (`truth_build`), written to its own folder and recorded by its
own run. A command that scores held-out units therefore reads two builds: the tables of the units
its models are fitted on (training units) from the development build, and the tables of the units
it scores from the held-out build. This module holds the checks that route makes, each written
once for every command that makes it. Not every command that has the route makes every check:
`gate_run` refuses itself once `matrix_route` has passed (reading S3); `ledger_run`, `o3_run` and
`ceiling_report` do not pass the gate again (`gate_again`); and `o3_run`, `ceiling_report` and
`drift_run`, whose units are named on their command lines, make neither `check_split_rule` nor
`check_split_days` (`o3_run` reading R5, `ceiling_report` reading O2-1, `drift_run` reading
E22). Each command's module text says which checks it makes. The checks:

`check_units(names)`: the units of one run, each named once, all development units
(`truth_build.DEVELOPMENT_UNITS`) or all held-out units (`heldout_truth_build.UNITS`: the two
a.s.r. regimes, OSL_T, TRO_R and the five Korean portfolios), never a mix, since their tables
come from two different truth builds. A training unit (`truth_build.TRAIN_UNITS`) is refused:
the plan fits on training units and never scores them. Any other name is refused. Returns the
names and their kind, `DEVELOPMENT` or `HELD_OUT`. A refusal raises `UnitsRefused`.

`require_gate(root)`: a run that names a held-out unit passes, before any file is read, the
checks the held-out loader makes before it opens a sealed file, the loader's own functions in its
order (checks 2 to 8 of `heldout_loader`), with the loader's look for links
(`heldout_loader.check_no_link`, made by the loader's own opener too) before the first file is
read. That look is made when HEAD is a later commit that descends from the tag, the one case in
which the loader's check 5 reads the openings log and the refusal log (to find a parsing fix's
logged refusal), and before that check: the data folder `data`, the sealed folder in it, the
openings log `data/heldout-openings.log` and the refusal log `data/heldout-refusals.log` under
`root` (their paths by the loader's `log_path`, `refusals_path` and its seal module's `folder`;
this module imports no seal module), in that order, are each refused when the path holds a ".."
part or when it is a symbolic link, dangling or not, looked at with `lstat` only after the
folder above it was seen not to be a link, so the look itself follows no link; a place that is
not there passes. The refusal names the path and says nothing was opened. Without the look,
check 5 would read both logs through a linked `data` before any reader refused them (the
loader opens a log with `O_NOFOLLOW`, which guards the log's own name only). At the tag's
commit, or at HEAD not descending from it, no log is read and no look is made, so the gate's
refusals there are as before. The checks: `root` is the top folder of the repository's main
working copy; the tag `frozen-plan` names a commit and the checked-out commit is that commit or a
logged parsing fix after it (the loader's check 5 and reading H6, with its message when HEAD is
neither); the tree is clean and no tracked file is hidden from `git status`; every loaded module
of the package and the entry module are the checked-out commit's files. Any failure raises
`heldout_loader.SealRefused` with the loader's message; any other error is raised as
`SealRefused` with that error as its cause, as the loader's opener does. Once its checks have
passed, it marks the recorded run going in `root`, if there is one (`gate_mark.mark_run`, as the
opener does after its check 8), so that `run_evidence` reads no file of that run from an earlier
commit; a mark that cannot be written refuses (`gate_mark`, reading G3). Returns the tag's commit
as a `GateCommit`: a string equal to the tag's commit, so a caller records the same string as
before, whose `fix_commit` is HEAD at a parsing fix and None at the tag. Nothing here opens,
lists, reads or hashes a file of the sealed folder: the held-out units' tables are the outputs of
`heldout_truth_build`, which alone reaches the sealed files, through the loader's gated routes;
the command then checks those tables against that build's recorded run at the checked-out commit
(`read_two_builds`), so on the route a held-out unit is scored only from tables of a build
recorded at the commit the gate passed at, the tag's or the fix's.

`fix_record(found)`: what a route's record adds for a parsing fix, from a `Route` or the
gate's `GateCommit`: `{"fix_commit": HEAD}` at a fix, nothing at the tag, so a record made at the
tag is as before, byte for byte. `fix_text(record)`: what a summary line naming the tag's commit
adds from such a record, "; at the logged parsing fix <HEAD>" at a fix and nothing at the tag. A
run at a fix thus states the fix in its results: the tag's commit and the fix commit in its
route's record and in its summary (plan section 13, WP14).

`held_out_tables(root, setting)`: the folder `heldout_truth_build` writes the held-out tables of a
setting to by default under the top folder `root` (`heldout_truth_build.default_out`),
`data/derived/truth-heldout` (main) or `data/derived/truth-heldout-held`.

`check_folders(root, tables, others)`: before anything is read, every folder a run on the route
reads is out of the held-out places under `root`. Under the top folder's `data/` a run reads only
`data/derived/`, where the truth builds write their tables: any other folder at or under `data/`
is refused, whatever its name, so a folder named there by a slip (the sealed folder, kept under
`data/`, plan section 3, or the raw files) is refused before it is read, and this check never
takes the sealed folder's name (only the loader and the held-out truth build may import
`heldout_seal`; the look for links is the loader's, above). `tables` (the
tables of the fit units and of development units) is refused there and also at or under either
folder of `held_out_tables`; each of `others` (a truth run, the held-out tables, the held-out
truth run; None is skipped) is refused there. Each folder is compared three times. First by
its names as written, made absolute from the current folder with
no call to the file system (`heldout_truth_build._written`, as that build's own check of its
output folder), so a folder written inside a refused place is refused before the file system is
asked anything; then by its names with links followed (`Path.resolve`, which asks the file
system only whether each part of the path is a link and where it points, so a link into a
refused place is caught). In both the names are compared in any spelling a file system that
ignores letter case and Unicode normalisation takes for them (`gate_run.folded`), so
`DATA/heldout` is `data/heldout`. Last by identity on disk: the folder, or a folder it lies
under, is `root`'s `data` folder or a held-out build's folder by device and inode
(`gate_run.identity`), whatever it is called on the way in. A folder in `data/` is read only when
the part of its path after `data` is named `derived` exactly. Only `stat` and `readlink` are
asked of the file system; no file is opened, listed, read or hashed. A refusal raises
`FoldersRefused`. The files inside the folders are checked by `gate_run.check_inputs` before it
opens any of them.

`read_two_builds(truth_run, tables, fit_units, held_out_truth_run, held_out_tables, units)`: the
two builds of a run that scores held-out units, each checked as `gate_run.check_inputs` checks a
build (a recorded truth build that passes `run_verify` at the commit checked out now, of the
setting its tables folder holds, listing each unit's two tables with the SHA-256 they have now):
the fit units' tables against the development build (`tables`, `truth_run`), then the scored
units' tables against the held-out build (`held_out_tables`, `held_out_truth_run`). The two
builds must be of one setting (`main` or `held`, plan section 3: the settings are never pooled),
or the run could not run: between the two checks, the held-out tables folder's setting file
(checked first as the tables are, `gate_run.check_files`) must name the development build's
setting, so no held-out table is opened for a run of two settings (the held-out build's own check
then requires its summary to be of that folder's setting). Returns the two checks' records,
(development, held-out), and the bytes of every table the two checks hashed. A failure raises
`truth_verify.CouldNotRun`. A model with no fit unit passes none: the development build is still
checked (verified at the commit, of the setting of its tables folder), so a run on the route
always reads two builds of one setting. The checks are `gate_run.read_inputs`'s, which returns
the bytes of every table it hashed, by file name, so the caller parses the bytes whose SHA-256
the records hold (`learn_run`, `compare_run`, `corner_run`, `drift_run`, `ledger_run`, `o3_run`,
`ceiling_report` and the cold start, every command that reads tables on the route). The bytes come
as a `days_from_truth.ReadFiles`, which knows the folder each table was read from and what the
file system said of it just before the read, so `days_from_truth.parse_unit_days` asks again
after parsing it (`stat` only, no second open) and the run could not run when a table changed
between its check and its parse. Each build's summary, which holds the
tables' SHA-256, is parsed in the same way from the bytes its manifest's check hashed
(`truth_verify.load_summary`).

The route of a command whose scored units are a run matrix's gate units (`gate_run`, `learn_run`,
`compare_run`, `corner_run`; `ledger_run`, which reads a `learn_run` run on such a matrix),
`matrix_route(root, units, tables, truth_run, scored_tables, scored_truth_run)`, in this order,
each before anything is read:
1. The kind of the units (`held_out_kind`): all held-out units, or none; a mix is refused. When
   none is held out, the route is off: nothing below is checked and the command runs exactly as
   before (a development or rotation what-if matrix, `gate_config`), and the two held-out
   arguments (`scored_tables`, `scored_truth_run`, the options `--held-out-tables` and
   `--held-out-truth-run`) are refused if given, since they name tables no unit of the matrix is
   scored from.
2. The folders (`check_folders`): `tables` and the three other folders.
3. The gate (`require_gate`).
4. The held-out build's run is required; its tables folder defaults to `held_out_tables(root)`
   (the main setting's), as the cold start's does.
A refusal raises `RouteRefused`, whose text the command prints after "COULD NOT RUN: " (exit 2),
in the cold start's words. The returned `Route` holds the kind, the tag's commit, the two
held-out folders and the fix commit (`fix_commit`, None at the tag). A command that loads code
after the route (a model's module) passes the gate again before it reads a table (`gate_again`),
as the cold start does, and the gate must then pass at the commits the route passed at, the
tag's commit and the fix commit (`same_gate`): if the tag or HEAD moved in between, from the tag
to a fix, from one fix to a later one or back, the run could not run, so a run never records one
fix and reads tables under another. The text `LABEL` names where the gate passes, "at the tag
frozen-plan, or at a logged parsing fix of it" (`AT_THE_TAG`), one text true at both (what a
logged parsing fix is: the held-out loader's module text). `mark` records the route in a results
document under `sealed_route` (`gate_summary.ROUTE_KEY`: the kind, the label `LABEL`, the tag's
commit, at a parsing fix the fix commit (`fix_record`), the held-out build's check, the fit
units, under `kappa_selection` the
text `NO_SELECTION`, and under `checks` the text `CHECKS`, what the route's checks stop and what
they do not, as in the first paragraph above); `summary_head` gives the summary's first two lines
from it, the route (naming the fix commit at a fix, `fix_text`) and then `CHECKS`; both add
nothing when the route is off.

The route of a command whose scored units are named on its command line (`--units`) rather than
by a matrix, `units_route(root, units, tables, truth_run, scored_tables, scored_truth_run)`
(`o3_run`, `ceiling_report`, `drift_run`): `matrix_route`'s steps 1 to 4, in its order and with
its checks, its texts naming the units named where `matrix_route`'s name the matrix's gate units.
It returns a `Route` as `matrix_route` does. It is a function of its own because `matrix_route`'s
texts name a matrix, which such a command has not. Such a command records its own route: `mark` and
`summary_head` write a kappa selection and fit units, which belong to the commands of a matrix.

`committed_text(root, tag_commit, relative)`: a setting a command reads on the route from a
committed file (`drift_run`'s step sizes, `learn/drift-eta.json`) is read as the tag's commit
holds it, from git's objects (`git ls-tree` of that commit, then `git cat-file blob`), never from
the working copy, so the setting is the one frozen with the code. Returns the text and its git
object id. A path that the tag's commit holds as no file (absent, or a folder) could not run
(`CouldNotRun`), whatever lies in the working copy at that path; the gate (`require_gate`) has
already required a clean tree, so the two agree whenever the route runs (at a parsing fix too,
since a fix changes only loaders, their test files and the plan: the loader's reading H6(b)).

`openings_log(root)`, not part of the route: the openings log's path under the top folder `root`
and every opening it holds, by the held-out loader's own `log_path` and `logged_openings` (an
absent or empty log gives none; a line that is not JSON with the log's fields raises
`ValueError`, and bytes that are not UTF-8 its kind `UnicodeDecodeError`; a link or a folder at
the log's path raises `OSError`), for WP13's check of its pass mark (`tools/freeze_check.py
openings`), which the repository rule `heldout-seal` keeps from naming the loader. That check
also calls it on a temporary top folder, first with no log there for the log's place, then with
one line of an unreadable log there at a time (its reading F7). It reads the log alone: nothing
in the sealed folder is opened, listed or read, and no tag is needed. `refusals_log(root)`, the
same for the refusal log (the loader's `refusals_path` and `logged_refusals`, its reading H6),
and `sealed_logs(root)`, the two logs' paths alone, reading nothing, are for WP14's check of its
pass mark (`tools/sealed_run_check.py`), which the same rule keeps from naming the loader; they
too open, list or read nothing in the sealed folder.

No look at a scored unit's test days beyond scoring them. On the route a command refuses (exit 2,
could not run), before any table is read, with the texts here: a forecaster built from a scored
unit's own other days, test days included (`OUTCOME_READER`: `gate_run`'s empirical forecaster,
under either reference, and every model of `forecast_model.OUTCOME_READERS` in `learn_run`,
`compare_run` and `corner_run`), as the cold start refuses one (its reading C12; `drift_run`
refuses such a model on and off the route in its own words, its reading E7); and the trial options
`--limit-days` and `--only` of `gate_run` (`trial_refusal`), so a scored run is every counting
day under every configuration. `compare_run` has no trial option; on the route it scores the one
split the build drew and could not run with `--splits` other than 1 (its reading H1). The kappa
selection per unit, which picks kappa on the test days the offers are measured on, is neither
computed nor written (`gate_summary.gate_tables` with `selection` false; `NO_SELECTION`). The
split rule (`check_split_rule`): the held-out build's summary, its file and manifest checked
first as the tables are and verified at the checked-out commit, records the rule its calibration
days were drawn by (`calibration_rule`); the matrix's `split_seed`, `n_calibration` and
`calibration_half_below`, everything that decides a unit's split from its counting days, must
equal it, or the run could not run, naming both rules. The calibration days
(`check_split_days`), after the tables are read: for each scored unit whose window the build
searched on calibration days (`windows.<unit>.calibration_dates`; the a.s.r. units' window is the
workplace window and has none), the days the run's split draws (`gate_run.plan_unit`) must be
those days, or the run could not run (the matrix's readings `no_fit` and `small_fleet` can make
other counting days than the build's).

Abstaining units. The held-out truth build records a unit it could not calibrate as abstained
(`abstained[unit]` with a reason in its summary) and writes no tables for it (exit 0). On the
route `read_two_builds` leaves such a unit out of the held-out check: only a unit the summary
lacks AND records as abstained with a reason (`abstained_in_build`); a unit the summary lacks
without that record still could not run. The held-out check's record then holds
`ABSTAINED_KEY`, each such unit with its reason. Every command that reads the held-out build
scores the units left (`scored_units`; `scored_matrix` for the commands of a matrix, whose gate
units are then the units left, the matrix file unchanged), and prints and records one line per
abstained unit (`abstention_lines`: `<unit>: abstained in the held-out truth build <run>:
<reason>; not scored`), in the first lines of its summary (`summary_head` and the route heads of
the commands that write their own) and in its results under the route's record. A command with
no unit left could not run (`no_unit_left`), but `drift_run`, whose one unit the run list names,
goes on with none. The route's own off state is as before: nothing here is read or left out on
development units.

Readings of the route, each with its alternative in brackets:
  S1. A look at a scored unit's other days is a forecaster built from their outcomes: a model
      that declares `reads_other_day_outcomes` (`forecast_model.OUTCOME_READERS`) or `gate_run`'s
      empirical forecaster. A learned model's user-history inputs (`learn_features`: the user's
      earlier sessions at the same site, plugged out by the row's decision time, earlier test
      days included) are known at that time and are not refused, as in the cold start.
      [Alternative: refuse on the route every model that reads a history column.]
  S2. A scored unit whose window the build did not search on calibration days (an a.s.r. unit)
      is not compared day by day; its split rule is compared as every unit's is. [Alternative:
      refuse such a unit on the route until its build records the calibration days it drew.]
  S3. `gate_run` on the route could not run at all, since its one forecaster is refused (S1); a
      held-out unit's offer is scored by `learn_run`, and the other commands that score held-out
      units are named in the first paragraph. [Alternative: an empirical forecaster built from
      each unit's calibration days only, a new rule the plan does not state.]
  S4. A unit left out is not scored but stays in every count over the held-out units, in its
      denominator, counted as not passing and shown as "abstained (<reason>)", as `gate_summary`
      counts a development unit that cannot be calibrated: the gate decisions of `learn_run`
      (`abstained_gate_data`), the drift run's record that its measure could not be made, the
      pooled lines and the unit table of the intervals over days (`day_intervals` reading I18:
      "pooled over K of N units", a row for the unit; `abstained_counted` gives the words) and
      the line over all units of the corner check (`corner_summary`). Outcome O2's verdict over
      the units is the exception: the unit is listed there as having no day to check, as the
      owner's decision of 19 November 2026 lists a unit with none (`ceiling_report` reading
      O2-7). The count of an abstained unit is the plan's rule (Section 11, decided 22 November
      2026, before any sealed file was opened). The rule sits in one place of this module (above
      `abstention_lines`).
"""

from dataclasses import dataclass, replace
from pathlib import Path

from fleetreserve import (
    gate_mark,
    gate_run,
    gate_summary,
    heldout_loader,
    heldout_truth_build,
    truth,
)
from fleetreserve.days_from_truth import ReadFiles
from fleetreserve.run_record import MANIFEST_NAME
from fleetreserve.run_verify import NotVerified
from fleetreserve.sessions import TOP
from fleetreserve.truth_build import DEVELOPMENT_UNITS, SETTING_FILE, TRAIN_UNITS, read_setting
from fleetreserve.truth_build import SUMMARY_NAME as TRUTH_SUMMARY_NAME
from fleetreserve.truth_verify import CouldNotRun, load_summary

DEVELOPMENT = "development"
HELD_OUT = "held-out"
HELD_OUT_UNITS = heldout_truth_build.UNITS
# The held-out truth build's module, by name within the package: data only, read by the run
# list's checker (`run_list` reading L11), which may not name it; a test pins it to the module.
TRUTH_BUILD_MODULE = "heldout_truth_build"
# The tag the gate passes at, for the run list's start check (`run_starts`), which may not name the
# loader.
TAG = heldout_loader.TAG
SealRefused = heldout_loader.SealRefused  # the loader's refusal, for callers of this module


class UnitsRefused(Exception):
    """The units named cannot be scored by one run (module text)."""


def check_units(names) -> tuple[tuple[str, ...], str]:
    """(the names, their kind) of a run's units; raises `UnitsRefused` (module text)."""
    names = tuple(names)
    if not names:
        raise UnitsRefused("no unit is named")
    if len(set(names)) != len(names):
        raise UnitsRefused(f"a unit is named twice: {list(names)}")
    training = [n for n in names if n in TRAIN_UNITS]
    if training:
        raise UnitsRefused(
            f"{training} are training units, which the plan fits on and never scores"
        )
    unknown = [n for n in names if n not in DEVELOPMENT_UNITS and n not in HELD_OUT_UNITS]
    if unknown:
        raise UnitsRefused(
            f"{unknown} are neither development units ({', '.join(DEVELOPMENT_UNITS)}) nor "
            f"held-out units ({', '.join(HELD_OUT_UNITS)})"
        )
    held = [n for n in names if n in HELD_OUT_UNITS]
    if held and len(held) != len(names):
        raise UnitsRefused(
            "a run names development units or held-out units, never both: their tables come "
            "from two different truth builds"
        )
    return names, HELD_OUT if held else DEVELOPMENT


class GateCommit(str):
    """The tag's commit as `require_gate` returns it, with `fix_commit`: HEAD at a logged
    parsing fix after the tag, None at the tag (module text)."""

    fix_commit: str | None

    def __new__(cls, tag_commit: str, fix_commit: str | None = None):
        made = super().__new__(cls, tag_commit)
        made.fix_commit = fix_commit
        return made


def require_gate(root: Path = TOP) -> GateCommit:
    """The held-out loader's checks 2 to 8, in its order, with the look for links before its
    check 5 reads the logs (module text); the tag's commit, with the fix commit at a parsing
    fix."""
    try:
        root = Path(root)
        heldout_loader._check_working_copy(root)
        head, tag_commit, later = heldout_loader._head_after_tag(root)
        if head != tag_commit and later:
            heldout_loader.check_no_link(root)
        head, tag_commit, fix = heldout_loader._check_at_tag_or_fix(root)
        heldout_loader._check_clean(root)
        heldout_loader._check_code(root, head, "the fix commit" if fix else "the tag")
        gate_mark.mark_run(root, tag_commit, head if fix else None)
        return GateCommit(tag_commit, head if fix else None)
    except heldout_loader.SealRefused:
        raise
    except Exception as error:
        raise heldout_loader.SealRefused(f"refused: {type(error).__name__}: {error}") from error


FIX_KEY = "fix_commit"


def fix_record(found) -> dict:
    """What a route's record adds for a parsing fix, from a `Route` or a `GateCommit` (module
    text): the fix commit at a fix, nothing at the tag."""
    fix = getattr(found, FIX_KEY, None)
    return {} if fix is None else {FIX_KEY: fix}


def fix_text(record: dict) -> str:
    """What a summary line naming the tag's commit adds from a route's record (module text)."""
    fix = record.get(FIX_KEY)
    return "" if fix is None else f"; at the logged parsing fix {fix}"


def held_out_tables(root: Path = TOP, setting: str = truth.MAIN) -> Path:
    """The default folder of the held-out tables of `setting` under `root` (module text)."""
    return heldout_truth_build.default_out(root, setting)


class FoldersRefused(Exception):
    """A folder a run would read lies in a held-out place (module text)."""


def _resolved(path) -> Path:
    return Path(path).resolve()


def _under(where: Path, place: Path) -> int | None:
    """The number of parts of `place` when `where` is `place` or lies under it, the names
    compared in any spelling (`gate_run.folded`); else None."""
    if len(where.parts) < len(place.parts):
        return None
    for one, other in zip(where.parts, place.parts, strict=False):
        if gate_run.folded(one) != gate_run.folded(other):
            return None
    return len(place.parts)


def _under_on_disk(where: Path, place: Path) -> int | None:
    """The number of parts of the folder `where` is or lies under that is `place` on disk (the
    same device and inode, whatever it is called); else None."""
    mark = gate_run.identity(place)
    if mark is None:
        return None
    for folder in (*reversed(where.parents), where):
        if gate_run.identity(folder) == mark:
            return len(folder.parts)
    return None


def _outside_derived(where: Path, depth: int | None) -> bool:
    """Whether `where`, in a data folder of `depth` parts, lies outside its derived tables: the
    part after the data folder is not named `derived` exactly."""
    return depth is not None and (len(where.parts) == depth or where.parts[depth] != "derived")


def _refused(folder: Path, root: Path, builds: list[Path]) -> str | None:
    """Why `folder` may not be read, or None: by names, as written first with no call to the file
    system and then with links followed; last by identity on disk (module text)."""
    data = Path(root) / "data"
    derived = data / "derived"
    building = "the held-out build's folder {}"
    outside = f"the data folder {data} outside its derived tables {derived}"
    for form in (heldout_truth_build._written, _resolved):
        where = form(folder)
        for build in builds:
            if _under(where, form(build)) is not None:
                return building.format(build)
        if _outside_derived(where, _under(where, form(data))):
            return outside
    where = _resolved(folder)
    for build in builds:
        if _under_on_disk(where, build) is not None:
            return building.format(build)
    if _outside_derived(where, _under_on_disk(where, data)):
        return outside
    return None


def check_folders(root: Path, tables: Path, others=()) -> None:
    """Raises `FoldersRefused` when a folder a run reads is in a held-out place (module text)."""
    builds = [held_out_tables(root, setting) for setting in heldout_truth_build.OUT_NAMES]
    found = _refused(tables, root, builds)
    if found is not None:
        raise FoldersRefused(f"the tables folder {tables} lies in {found}")
    for folder in others:
        found = None if folder is None else _refused(folder, root, [])
        if found is not None:
            raise FoldersRefused(f"the folder {folder} lies in {found}")


def _one_setting(development: dict, held_out_tables) -> None:
    """Raises `CouldNotRun` when the held-out tables folder's setting is not the development
    build's; its setting file is checked first (module text)."""
    setting_file = Path(held_out_tables) / SETTING_FILE
    gate_run.check_files([setting_file])
    found = read_setting(held_out_tables)
    if found != development["setting"]:
        raise CouldNotRun(
            f"the held-out tables in {held_out_tables} are of the {found} setting but the "
            f"development build {development['truth_run']} is of the {development['setting']} "
            "setting; a run reads the two builds of one setting"
        )


ABSTAINED_KEY = "abstained"


def abstained_in_build(held_out_truth_run, units) -> dict[str, str]:
    """The reason of each of `units` that the held-out build `held_out_truth_run` recorded as
    abstained and wrote no tables for (module text: abstaining units); the build's summary is
    read as `gate_run.read_inputs` reads it. A unit the summary lacks and records no abstention
    for is not here, so reading it still could not run. When the summary cannot be read here,
    none is here: `gate_run.read_inputs` then refuses it, for the same cause."""
    if held_out_truth_run is None:
        return {}
    run = Path(held_out_truth_run)
    try:
        gate_run.check_files([run / TRUTH_SUMMARY_NAME, run / MANIFEST_NAME])
        if gate_run.foreign_manifest(run):
            return {}
        summary = load_summary(run, gate_run._head_commit())
    except (CouldNotRun, NotVerified):
        return {}
    written = summary.get("units") or {}
    recorded = summary.get(ABSTAINED_KEY) or {}
    found = {}
    for unit in units:
        entry = recorded.get(unit)
        reason = entry.get("reason") if isinstance(entry, dict) else None
        if unit not in written and isinstance(reason, str) and reason:
            found[unit] = reason
    return found


def abstained_of(held_out_inputs) -> dict:
    """The abstained units of a record of `read_two_builds`, whatever the record's type: none
    when it is no mapping or holds no mapping under `ABSTAINED_KEY`."""
    found = held_out_inputs.get(ABSTAINED_KEY) if isinstance(held_out_inputs, dict) else None
    return found if isinstance(found, dict) else {}


def scored_units(units, held_out_inputs: dict | None) -> tuple[str, ...]:
    """`units` without those the held-out build abstained, as the record of `read_two_builds`
    (`held_out_inputs`) lists them; the units themselves when none is listed (module text)."""
    left_out = abstained_of(held_out_inputs)
    return tuple(unit for unit in units if unit not in left_out)


# --- how an abstained unit counts (the one place; reading S4) -------------------------------------
# An abstained unit is not scored and stays in every count over the held-out units: it is in the
# denominator and counts as NOT PASSING, shown as "abstained (<reason>)", as `gate_summary` counts
# a development unit that cannot be calibrated. This is the plan's rule (Section 11, decided 22
# November 2026, before any sealed file was opened); it is kept in the names below: `COUNTS_AS`,
# `abstained_text` and `abstained_gate_data`. O2 alone lists the unit as having no day to check
# (`ceiling_report.abstained_units_o2`, the owner's decision of 19 November 2026).
COUNTS_AS = "not passing"


def abstained_text(reason: str) -> str:
    """How an abstained unit is shown in a count: `abstained (<reason>)`."""
    return f"abstained ({reason})"


def abstained_gate_data(held_out_inputs: dict | None) -> dict[str, dict]:
    """For each unit the held-out build abstained (the record of `read_two_builds`), the test data
    the gate decisions (`gate_run.assemble`, `counted_units`) take for it under every
    configuration: a reason and no number, so the unit passes neither rule and stays in the
    denominator (`gate_summary.unit_offers`)."""
    return {
        unit: {"reason": abstained_text(reason)}
        for unit, reason in abstained_of(held_out_inputs).items()
    }


def abstention_lines(held_out_inputs: dict | None) -> list[str]:
    """One line per unit the held-out build abstained, in the record of `read_two_builds`
    (`held_out_inputs`): `<unit>: abstained in the held-out truth build <run>: <reason>; not
    scored, counted as not passing`; none when no unit abstained (module text)."""
    record = held_out_inputs if isinstance(held_out_inputs, dict) else {}
    return [
        f"{unit}: abstained in the held-out truth build {record.get('truth_run')}: {reason}; "
        f"not scored, counted as {COUNTS_AS}"
        for unit, reason in abstained_of(record).items()
    ]


def abstained_counted(held_out_inputs: dict | None) -> dict[str, str]:
    """For each unit the held-out build abstained, the words a count over the held-out units shows
    for it, `abstained in the held-out truth build <run>: <reason>; counts as not passing`: the
    unit has a place in the count (a row, a name in the left-out part), and no number."""
    record = held_out_inputs if isinstance(held_out_inputs, dict) else {}
    return {
        unit: f"abstained in the held-out truth build {record.get('truth_run')}: {reason}; "
        f"counts as {COUNTS_AS}"
        for unit, reason in abstained_of(record).items()
    }


def no_unit_left(held_out_inputs: dict | None) -> str:
    """The text of a command that has no unit left to score (module text)."""
    return (
        "no unit is left to score: every unit named abstained in the held-out truth build "
        f"{(held_out_inputs or {}).get('truth_run')}; "
        + "; ".join(abstention_lines(held_out_inputs))
    )


def scored_matrix(matrix, held_out_inputs: dict | None):
    """`matrix` for the units of the run that are scored: its gate units without those the
    held-out build abstained (`scored_units`); the matrix itself when none is left out. Raises
    `truth_verify.CouldNotRun` when no unit is left (`no_unit_left`; module text)."""
    scored = scored_units(matrix.gate_units, held_out_inputs)
    if not scored:
        raise CouldNotRun(no_unit_left(held_out_inputs))
    if scored == tuple(matrix.gate_units):
        return matrix
    return replace(matrix, gate_units=scored)


def read_two_builds(
    truth_run, tables, fit_units, held_out_truth_run, held_out_tables, units
) -> tuple[dict, dict, dict[str, bytes]]:
    """(the development build's check of `fit_units`, the held-out build's check of `units`) and
    the bytes of every table the two checks hashed, by file name (`gate_run.read_inputs`; the
    fit units are training units and the scored units never are, so no name is in both), as a
    `days_from_truth.ReadFiles` that knows the folder of each; raises `truth_verify.CouldNotRun`.
    A unit the held-out build recorded as abstained (`abstained_in_build`) is left out of the
    held-out check, whose record then holds `ABSTAINED_KEY`: each such unit with its reason (module
    text: abstaining units)."""
    development, read = gate_run.read_inputs(truth_run, tables, tuple(fit_units))
    _one_setting(development, held_out_tables)
    try:
        held_out, held_read = gate_run.read_inputs(
            held_out_truth_run, held_out_tables, tuple(units)
        )
    except CouldNotRun:
        # Only a refusal can lead here, and only a unit the summary lacks and records as abstained
        # changes it: the summary is read again for that record, and a run with no such unit
        # raises what it raised (so a run with no abstention reads the build exactly once).
        abstained = abstained_in_build(held_out_truth_run, tuple(units))
        if not abstained:
            raise
        scored = tuple(unit for unit in units if unit not in abstained)
        held_out, held_read = gate_run.read_inputs(held_out_truth_run, held_out_tables, scored)
        held_out = {**held_out, ABSTAINED_KEY: abstained}
    folders, states = {**read.folders, **held_read.folders}, {**read.states, **held_read.states}
    return development, held_out, ReadFiles({**read, **held_read}, folders, states)


# --- the route of a command whose units are a matrix's gate units ---------------------------------

# Where the gate passes, in the words of every sealed-data text of the package (what a logged
# parsing fix is: the held-out loader's module text, the one place that says it).
AT_THE_TAG = f"at the tag {heldout_loader.TAG}, or at a logged parsing fix of it"
LABEL = f"Held-out units, through the sealed-data gate {AT_THE_TAG}, with a clean tree"
# What the route's checks stop and what they do not (module text, first paragraph); recorded in
# every results document on the route and shown as the summary's second line.
CHECKS = (
    "The route's checks stop ordinary mistakes by an honest operator; they are not a defence "
    "against deliberate evasion (links or copies made on purpose, a second working copy). The "
    "proof that the seal held is the plan's own: the committed hashes, the held-out loader, "
    f"which opens a sealed file only {AT_THE_TAG}, with a clean tree, and its openings log."
)
# What a run on the route records in place of a kappa selection per unit (module text).
NO_SELECTION = (
    "none: a selection per unit would pick kappa on the scored units' test days; on the sealed "
    "route kappa is that of each configuration of the matrix, fixed before the tag"
)
# The refusal of a forecaster built from a scored unit's own other days (module text); `what`
# names it.
OUTCOME_READER = (
    "{what} builds a scored unit's forecasts from that unit's own other days, test days "
    "included; on the sealed route a scored unit's days reach a model only through its inputs, "
    "as in the cold start (its reading C12); nothing read"
)
TRIAL_OPTIONS = (
    "{options} {verb} for trials on development units; a run on the sealed route scores every "
    "counting day of every scored unit under every configuration of its matrix; nothing read"
)


def trial_refusal(options: dict) -> str | None:
    """The refusal of the trial options given (`options`: option name to value, None when not
    given), or None when none is given (module text)."""
    given = [name for name, value in options.items() if value is not None]
    if not given:
        return None
    verb = "is" if len(given) == 1 else "are"
    return TRIAL_OPTIONS.format(options=" and ".join(given), verb=verb)


class RouteRefused(Exception):
    """A run on the route could not run (exit 2); the text follows "COULD NOT RUN: "."""


@dataclass(frozen=True)
class Route:
    """The outcome of `matrix_route` or `units_route` (module text)."""

    held_out: bool
    tag_commit: str | None = None
    tables: Path | None = None  # the held-out tables folder
    truth_run: Path | None = None  # the held-out build's recorded run
    fix_commit: str | None = None  # HEAD at a logged parsing fix after the tag, else None


def held_out_kind(units) -> bool:
    """True when every unit is held-out, False when none is; raises `UnitsRefused` for a mix."""
    units = tuple(units)
    held = [u for u in units if u in HELD_OUT_UNITS]
    if held and len(held) != len(units):
        raise UnitsRefused(
            f"the matrix's gate units {list(units)} mix held-out units ({held}) with others; a run "
            "scores development units or held-out units, never both: their tables come from two "
            "different truth builds"
        )
    return bool(held)


def gate_again(route: Route, root: Path) -> None:
    """The gate once more for a run on the route, at the commits the route passed at (module
    text); raises `RouteRefused`."""
    if route.held_out:
        _gate(root, route)


def passed_at(tag_commit: str, fix_commit: str | None) -> str:
    """Where the gate passed, in words: the tag's commit, or a logged parsing fix of it."""
    tag = f"the tag {heldout_loader.TAG} ({tag_commit})"
    return tag if fix_commit is None else f"the logged parsing fix {fix_commit} of {tag}"


def same_gate(tag_commit: str, fix_commit: str | None, found) -> None:
    """Raises `SealRefused` when the gate's commits now (`found`, as `require_gate` returns it)
    are not the tag's commit and fix commit it passed at before in the same run (module text)."""
    now = (str(found), getattr(found, FIX_KEY, None))
    if now != (str(tag_commit), fix_commit):
        raise SealRefused(
            f"the gate passes now at {passed_at(*now)}, not at "
            f"{passed_at(tag_commit, fix_commit)}, where it passed when this run started"
        )


def _gate(root: Path, route: Route | None = None) -> GateCommit:
    try:
        found = require_gate(root)
        if route is not None:
            same_gate(route.tag_commit, route.fix_commit, found)
        return found
    except SealRefused as problem:
        raise RouteRefused(f"REFUSED BY THE GATE: {problem}; nothing read") from problem


def matrix_route(root, units, tables, truth_run, scored_tables, scored_truth_run) -> Route:
    """Steps 1 to 4 of the module text; raises `RouteRefused`."""
    try:
        held = held_out_kind(units)
    except UnitsRefused as problem:
        raise RouteRefused(f"the units are refused: {problem}; nothing read") from problem
    if not held:
        given = [
            option
            for option, value in (
                ("--held-out-tables", scored_tables),
                ("--held-out-truth-run", scored_truth_run),
            )
            if value is not None
        ]
        if given:
            raise RouteRefused(
                f"{' and '.join(given)} {'is' if len(given) == 1 else 'are'} for held-out units, "
                f"and none of the matrix's gate units {list(units)} is a held-out unit; nothing "
                "read"
            )
        return Route(False)
    try:
        check_folders(root, tables, (truth_run, scored_tables, scored_truth_run))
    except FoldersRefused as problem:
        raise RouteRefused(f"a folder is refused: {problem}; nothing read") from problem
    tag_commit = _gate(root)
    if scored_truth_run is None:
        raise RouteRefused("held-out units need --held-out-truth-run")
    folder = scored_tables if scored_tables is not None else held_out_tables(root)
    return Route(True, tag_commit, Path(folder), Path(scored_truth_run), tag_commit.fix_commit)


def units_route(root, units, tables, truth_run, scored_tables, scored_truth_run) -> Route:
    """`matrix_route`'s steps 1 to 4 for the units named on a command line (module text);
    raises `RouteRefused`."""
    units = tuple(units)
    held = [u for u in units if u in HELD_OUT_UNITS]
    if held and len(held) != len(units):
        raise RouteRefused(
            f"the units are refused: the units named {list(units)} mix held-out units ({held}) "
            "with others; a run scores held-out units or others, never both: their tables come "
            "from two different truth builds; nothing read"
        )
    if not held:
        given = [
            option
            for option, value in (
                ("--held-out-tables", scored_tables),
                ("--held-out-truth-run", scored_truth_run),
            )
            if value is not None
        ]
        if given:
            raise RouteRefused(
                f"{' and '.join(given)} {'is' if len(given) == 1 else 'are'} for held-out units, "
                f"and none of the units named {list(units)} is a held-out unit; nothing read"
            )
        return Route(False)
    try:
        check_folders(root, tables, (truth_run, scored_tables, scored_truth_run))
    except FoldersRefused as problem:
        raise RouteRefused(f"a folder is refused: {problem}; nothing read") from problem
    tag_commit = _gate(root)
    if scored_truth_run is None:
        raise RouteRefused("held-out units need --held-out-truth-run")
    folder = scored_tables if scored_tables is not None else held_out_tables(root)
    return Route(True, tag_commit, Path(folder), Path(scored_truth_run), tag_commit.fix_commit)


def committed_text(root, tag_commit: str, relative) -> tuple[str, str]:
    """(the text, its git object id) of the file `relative` (a path from the top folder `root`)
    as the tag's commit holds it, read from git's objects and never from the working copy;
    raises `CouldNotRun` when that commit holds no such file (module text)."""
    name = Path(relative).as_posix()
    try:
        listing = heldout_loader._git_out(Path(root), "ls-tree", "-z", tag_commit, "--", name)
        entries = [entry for entry in listing.split("\0") if entry]
        if len(entries) != 1 or entries[0].split()[1] != "blob":
            raise CouldNotRun(
                f"{name} is not a file of the tag's commit {tag_commit}; on the sealed route a "
                "setting is read only from the file committed at the tag; nothing read"
            )
        blob = entries[0].split("\t", 1)[0].split()[2]
        return heldout_loader._git_out(Path(root), "cat-file", "blob", blob), blob
    except heldout_loader.SealRefused as problem:
        raise CouldNotRun(f"{name} cannot be read from the tag's commit: {problem}") from problem


def openings_log(root: Path = TOP) -> tuple[Path, list[dict]]:
    """(the openings log's path under `root`, every opening it holds), by the held-out loader's
    `log_path` and `logged_openings`; raises their `ValueError` or `OSError` (module text)."""
    root = Path(root)
    return heldout_loader.log_path(root), heldout_loader.logged_openings(root)


def refusals_log(root: Path = TOP) -> tuple[Path, list[dict]]:
    """(the refusal log's path under `root`, every layout refusal it holds), by the held-out
    loader's `refusals_path` and `logged_refusals`; raises their `ValueError` or `OSError`
    (module text)."""
    root = Path(root)
    return heldout_loader.refusals_path(root), heldout_loader.logged_refusals(root)


def openings_in(content: bytes) -> list[dict]:
    """Every opening in the bytes of an openings log, by the held-out loader's `log_lines`, as
    `openings_log` reads them from the file; reads no file; raises its `ValueError`."""
    return heldout_loader.log_lines(content, heldout_loader.LOG_NAME)


def refusals_in(content: bytes) -> list[dict]:
    """Every layout refusal in the bytes of a refusal log, by the held-out loader's `log_lines`,
    as `refusals_log` reads them from the file; reads no file; raises its `ValueError`."""
    return heldout_loader.log_lines(content, heldout_loader.REFUSALS_NAME)


def sealed_logs(root: Path = TOP) -> tuple[Path, Path]:
    """(the openings log's path, the refusal log's path) under `root`, by the held-out loader's
    `log_path` and `refusals_path`; reads nothing (module text)."""
    root = Path(root)
    return heldout_loader.log_path(root), heldout_loader.refusals_path(root)


def _rule_text(seed, n_calibration, half_below) -> str:
    return f"split_seed {seed}, n_calibration {n_calibration}, calibration_half_below {half_below}"


def check_split_rule(matrix, route: Route) -> dict:
    """The held-out build's summary, verified at the commit checked out now, when the matrix's
    split rule is the one the build recorded; raises `CouldNotRun` (module text)."""
    run = Path(route.truth_run)
    manifest = run / MANIFEST_NAME
    gate_run.check_files([run / TRUTH_SUMMARY_NAME, manifest])
    if not manifest.is_file():
        raise CouldNotRun(f"{manifest} is missing; a held-out build is read only with its manifest")
    summary = load_summary(run, gate_run._head_commit())
    recorded = summary.get("calibration_rule") or {}
    ours = (matrix.split_seed, matrix.n_calibration, matrix.calibration_half_below)
    theirs = tuple(recorded.get(key) for key in ("seed", "n_calibration", "half_below"))
    if ours != theirs:
        raise CouldNotRun(
            f"the matrix's split rule ({_rule_text(*ours)}) is not the rule the held-out build "
            f"{run.name} drew its calibration days by ({_rule_text(*theirs)}); on the sealed "
            "route every scored unit is split by its build's rule"
        )
    return summary


def check_split_days(matrix, units: dict, summary: dict, run_name: str) -> None:
    """Raises `CouldNotRun` when a scored unit's calibration days as the run draws them are not
    those its build searched the unit's window on (module text)."""
    for name, days in units.items():
        window = summary.get("windows", {}).get(name, {})
        if "calibration_dates" not in window:
            continue  # a window not searched on calibration days (an a.s.r. unit)
        plan = gate_run.plan_unit(name, days, matrix)
        drawn = sorted(key.split(" ", 1)[0] for key in plan.split.get("calibration", []))
        recorded = sorted(window["calibration_dates"])
        if drawn == recorded:
            continue
        here = sorted(set(drawn) - set(recorded))
        there = sorted(set(recorded) - set(drawn))
        raise CouldNotRun(
            f"{name}: the {len(drawn)} calibration days this run draws are not the "
            f"{len(recorded)} the held-out build {run_name} searched its window on ("
            f"{len(here)} drawn only here{', from ' + here[0] if here else ''}, {len(there)} only "
            f"in the build{', from ' + there[0] if there else ''}); the matrix's readings (no_fit "
            f"{matrix.no_fit}, small_fleet {matrix.small_fleet}) make other counting days than "
            "the build's, so on the sealed route the unit is not scored"
        )


def mark(document: dict, route: Route, held_out_inputs: dict | None, fit_units=()) -> None:
    """Records the route in `document` (module text); nothing when the route is off."""
    if not route.held_out:
        return
    document[gate_summary.ROUTE_KEY] = {
        "kind": HELD_OUT,
        "label": LABEL,
        "tag_commit": route.tag_commit,
        **fix_record(route),
        "held_out_inputs": held_out_inputs,
        "fit_units": list(fit_units),
        "kappa_selection": NO_SELECTION,
        "checks": CHECKS,
    }


def summary_head(document: dict) -> list[str]:
    """The summary's first two lines on the route (module text); none when the route is off."""
    found = document.get(gate_summary.ROUTE_KEY)
    if not found:
        return []
    fitted = found["fit_units"]
    fit = f"the fit units' ({', '.join(fitted)})" if fitted else "no fit unit's"
    return [
        f"{found['label']} (tag commit {found['tag_commit']}{fix_text(found)}): the scored units' "
        f"tables from the held-out build {found['held_out_inputs']['truth_run']}, {fit} from the "
        f"development build {document['inputs']['truth_run']}.",
        found["checks"],
        *abstention_lines(found["held_out_inputs"]),
    ]
