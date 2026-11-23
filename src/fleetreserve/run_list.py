"""The run list: every recorded run of the run from the tag and of its dress rehearsal, in order,
in one JSON file from which both are derived, and the checker of its form (plan section 13: WP11
"run matrix frozen", WP12 "Dress rehearsal of held-out scripts on development units", WP13
"Freeze protocol, held-out list, run matrix, seeds", WP14 "Held-out run, once, from the tag ...
logs hashed").

Command line:
    python -m fleetreserve.run_list <run list> [--no-starts]

ONE LIST, TWO PURPOSES. A list is driven for one purpose at a time (`python -m
fleetreserve.run_driver <run list> --purpose <purpose>`): `rehearsal` (WP12: the steps of the run
from the tag, run on development units) or `sealed-run` (WP14: the run from the tag). The driver
runs the list's instantiation for that purpose (`instantiate`): the entries that run in it, in the
list's order, each command with its placeholders filled in and the purpose's own arguments added
at its end. The two instantiations can differ in three ways only (reading L8): the units an entry
is given, the entries marked for one purpose only, and the arguments marked for one purpose only.
Every other part (the modules, every other argument, the fixed values, the settings, the time
limits, the labels, the order and the dependencies) is one text for both purposes, so the two run
the same scripts by construction. The checker prints exactly where the two differ
(`differences`).

THE LIST. The run list of the run from the tag and of its rehearsal is `run-lists/run-list.json`,
the one fixed path the WP12 check reads (`tools/rehearsal_check.py`). Its matrices are written by
`sealed_matrix` from each model's kappa-choice results (`gate/sealed-matrix/`,
`gate/rehearsal-matrix/`); until they are committed the checker reports each step that names one
as faulty (rule 5: its matrix cannot be read).

THE FORM. A run list is one JSON object with exactly these fields:
- `format`: the text `run-list-2`.
- `about`: text saying what the list is.
- `units`: the units of each purpose, an object with exactly the keys `rehearsal` and
  `sealed-run`, each an object of named groups: name -> a non-empty list of distinct unit names.
  A name is letters, digits, '-' and '_'. An entry names one group, and each purpose gives it
  that purpose's group of that name. Every unit of a rehearsal group is a training or a
  development unit (reading L1), and no group of the sealed run mixes those with other units. A
  group that no entry of its purpose names is a fault (reading L2).
- `fixed`: the values fixed for the whole list, each under a name, in three objects (each may be
  empty): `seeds` (name -> whole number >= 0), `counts` (name -> whole number >= 1) and `names`
  (name -> non-empty text). They hold, for example, the 2,000 resamples of the 95% bootstrap
  intervals and their seed, the count and seed of the day sample of outcome O3 and of the
  comparison methods measured on their own sets, and the name of a base set.
- `entries`: a non-empty list of entries, run in this order, one at a time; each purpose runs at
  least one of them.

An entry is a JSON object with exactly these fields:
- `label`: the label of its recorded run, one that `run_record` records when the driver calls it
  (`--name=<label>`): letters, digits, '-' and '_' only, and short enough that the run folder's
  name `<UTC start>-<label>` fits the file system's limit on a name (reading L6); no two entries
  share one, nor two labels that are equal when letter case is ignored (reading L7). One label
  serves both purposes, so a later step finds an earlier one's run by its label in either.
- `only`: null when the entry runs in both purposes; else the one purpose it runs in, as the build
  of the sealed units' truth tables runs in the sealed run only.
- `command`: the command of every purpose the entry runs in, a list of non-empty texts that is a
  `python -m fleetreserve.<module>` call: the pair `-m fleetreserve.<module>` appears exactly
  once, right after an argument that is a Python program (`python`, `python3` or `python3.<n>`,
  with or without folders in front), and `<module>` is a module of the package, not a subpackage.
  What comes before the program (such as `uv run --frozen --group learn`) is not checked. An
  argument that holds `{` or `}` must be exactly one placeholder:
  - `{units}`: the units of the entry's group in the purpose run, as separate arguments, in the
    group's order; the command holds it when, and only when, the entry names a group, and
    `extra` never does;
  - `{seed:<name>}`, `{count:<name>}`, `{name:<name>}`: a fixed value, a whole number as its
    decimal digits, a name as its text;
  - `{run:<label>}`: the run folder that the earlier entry `<label>` recorded in the same drive,
    from the top folder (`runs.noindex/<UTC start>-<label>`, reading L10); that entry must come
    before this one and run in every purpose where the placeholder is used.
- `units`: the name of the group whose units `{units}` takes, or null.
- `extra`: the arguments one purpose adds at the end of the command, as an object from a purpose
  the entry runs in to a non-empty list of non-empty texts (the placeholders above but
  `{units}`); `{}` for none, and always `{}` for an entry of one purpose only, whose command is
  its own. The two arguments of the sealed route of a scoring command, for example, are its
  sealed run's extra arguments.
- `setting`: the truth setting it runs on, `main` or `held` (plan section 3).
- `max_seconds`: the time limit of the whole run in seconds, a positive finite number that reads
  back exactly as the float `run_record --max-seconds` makes of it (so not a whole number above
  2**53 that no float equals, nor one too large for a float); the driver gives it to
  `run_record --max-seconds` (plan section 13: every linear-programme solve has a time limit, and
  a timeout is recorded, never a hang), and the WP12 check holds the manifest's float against it.
- `after`: the labels of the entries it depends on, each an earlier entry that runs in every
  purpose this one runs in; may be empty.

THE CHECKER. A list is faulty when any field above is missing, unknown or not of its form, and
in particular when an entry: names a module the package lacks; has no time limit, or one that is
not a positive finite number read back exactly; repeats an earlier entry's label, also when letter
case is ignored; has a label that `run_record`, called as the driver calls it, would not record
(one it refuses, or one whose run folder name would be longer than the file system allows); names
a group that a purpose it runs in lacks, or a group with no `{units}` in its command, or holds
`{units}` with no group or in `extra`; has extra arguments for a purpose it does not run in;
depends on, or refers to the run of, itself, a later entry, a label no entry has, or an entry that
does not run in a purpose where it is needed; refers to a fixed value the list's `fixed` block
lacks. It is also faulty when a fixed value or a group is used by no entry (reading L2), when a
rehearsal group names a unit that is no training or development unit, when a group of the sealed
run mixes those with other units, and when a purpose runs no entry. Every fault found is reported,
one line each.

THE SEALED RUN'S RULES. The driver runs a list's entries in order and has no resume, and the run
from the tag is made once (plan section 13, WP14). The rehearsal stops at the first entry that
does not pass. The sealed run stops at the first one that does not pass only before the first
entry that runs the sealed truth build and at that build; after the opening it logs a failed
entry and goes on with the entries that do not need its run (`run_driver`, module text). The
rules below are checked of the list's sealed-run instantiation whichever purpose it is checked or
driven for, so a rehearsal of the list checks its sealed run's list too; a list that breaks one
is faulty. They are what is checked, and no more: a list that keeps them can still stop part-way
through the sealed run for a reason listed under "Not checked". The rules read the sealed run's
steps as `instantiate` gives them (fixed values put in, `{run:<label>}` kept):
  1. Truth builds first (reading L11). Each `--held-out-truth-run` is `{run:<label>}`, never a
     literal path (a literal path names a build of no entry, so the list's order cannot hold the
     build before the step); each entry so named runs the sealed truth build; and every entry
     that runs it comes before the sealed run's first entry that reads the sealed units' tables
     (gives `--held-out-truth-run` or `--held-out-tables`), so a sealed file that cannot be
     parsed stops the drive before any scoring.
  3. One run for the pooled figure (reading L14). Each `day_intervals` entry reads its run by
     `{run:<label>}` (`--run`), never by a literal path. Each `learn_run` on the sealed route
     (gives `--held-out-truth-run`) whose run a
     `day_intervals` entry reads names its matrix, whose gate units are the nine held-out units,
     no more and no fewer: the intervals pool the units of one run folder (`day_intervals`
     reading I9), and outcome O1 is over the nine (plan section 11). A matrix is read from git's
     objects at the checked-out commit of the working copy the commands run from (`unit_faults`:
     for the driver its working copy, for the command line the git working copy holding the
     list), never from a path on disk; a path a run would refuse, or one that is no file
     committed there, is a fault, and no such working copy at all makes the check undecidable
     (exit 2).
  4. After a logged parsing fix every entry runs again at the fix commit. This is the driver's
     own behaviour, not a rule of the list: a drive starts from the first entry and fills each
     `{run:<label>}` with a run of the same drive (reading L10), so no run recorded at the tag
     is read by a drive at the fix.
  2. The interval seed (reading L13). Each `learn_run` or `gate_run` of the sealed run that is
     on the sealed route (gives `--held-out-truth-run`) or whose run a sealed `day_intervals`
     entry reads gives `--interval-seed S --interval-resamples 2000`; the sealed
     `day_intervals` entries give one `--seed`, S, and 2000 resamples (`--resamples`, 2000
     when not given), and each reads the run of a `learn_run` or `gate_run` entry: else the
     intervals of outcomes O1 and O4 would need the sealed scoring made again
     (`day_intervals` reading I12). With no sealed `day_intervals` entry, S cannot be read
     from the list, and a scoring entry of the sealed route is a fault.
  5. The route's decision at each command's start (reading L15). For each step of a command of
     the sealed route (`ROUTE_COMMANDS`), the decision that command makes at its start about the
     units it scores is made here, on the step's options and on the units of its matrix
     (`--matrix`, or the command's default) read from git's objects as in rule 3, or on the
     units it names (`--units`, or its default): `sealed_route.held_out_kind`, then, where every
     unit is a development unit or the units mix, the command's own deciding function
     (`sealed_route.matrix_route` or `units_route`), or for `cold_start`
     `cold_start_gate.check_units`. Any refusal they give is a fault. Where every unit is a
     held-out unit the deciding function is not called, since its next steps look at the
     folders on disk and pass the gate; the refusals that follow those steps are the one
     copied rule: no `--held-out-truth-run` is a fault, and a `gate_run` is a fault (it refuses
     every run on the route, `sealed_route` reading S3, with `sealed_route.trial_refusal` or
     `OUTCOME_READER`).
  6. Truth builds first, and the openings of sealed data in one step (reading L17). The sealed
     run starts with its truth builds (`truth_build` and the sealed truth build): no other entry
     comes before one of them; and the entries that run the sealed truth build, which alone open
     sealed data, follow one another with no other entry between them.
  7. No model that reads other days' outcomes (reading L18): no step of the sealed run gives
     `--model` or `--reference` a model of `forecast_model.OUTCOME_READERS`.
  8. Each step's dial and kappa are its matrix's (reading L19). For each step of the sealed run
     that gives `--dial` or `--kappa`, its matrix (`--matrix`, or the command's default) is read
     from git's objects as in rule 3, and its field `sealed_choice` (written by `sealed_matrix`)
     must hold the dials per site type. The `--dial` must be written as the fixed name of a
     site type's dial (`{name:dial-workplaces}` or `{name:dial-others}`, `DIAL_NAMES`), its
     value must be the dial `sealed_choice.dials` gives that site type, and every unit the step
     names by `--units` must be of that site type (`sealed_matrix.site_type_of`); a `--kappa`
     must be `sealed_choice.kappa`. A fixed name has one value, so the drift guard, whose units
     are of Norway and Korea, gives the dial of the other steps of that site type, and that
     dial is its matrix's. A fault names the step and both values. A matrix that cannot be read
     is reported by rule 5 alone.
  9. The rehearsal reads no sealed units' tables (reading L20): no step of the rehearsal's
     instantiation gives `--held-out-truth-run` or `--held-out-tables`. This rule alone is
     checked of the rehearsal's instantiation; like the others it is checked whichever purpose
     the list is checked or driven for.
  10. Outcome O2 on the nine held-out units, every day (reading L21): each `ceiling_report` step
     of the sealed run names by `--units` the nine held-out units, no development unit and none
     left out, and gives `--every-day`.
  11. Each dial name once per command, and a lone step only at one shared dial (reading L22): of
     the `compare_run` steps of the sealed run, no two give `--dial` the same fixed name, nor of
     the `corner_run` steps. A `compare_run` or `corner_run` step may stand alone for its command
     (no step of its command gives the other site type's fixed name) only when
     `sealed_choice.dials` in its matrix gives every site type the same value, equal to its
     `--dial`; when the dials differ, the list must hold one step per site type, and a lone step
     is a fault naming the step and both dials.
  12. Outcome O2's two steps (reading L23): when any purpose of the list runs a `ceiling_report`
     step, each purpose's instantiation holds exactly two, one of the `main` setting and one of
     the `held` setting, each at rho 0 (`--rho` as the command reads it, its default when not
     given) and giving `--every-day`, its development truth run (`--truth-run` or `--run`) the
     `{run:<label>}` of a `truth_build` entry of its setting (the build's `--setting` as the
     build reads it); in the sealed run, its `--held-out-truth-run` the `{run:<label>}` of a
     sealed truth build entry of its setting, and its `--held-out-tables` as the command reads it
     (not given: the main setting's folder) the sealed tables folder of its own setting
     (`sealed_route.held_out_tables`). Every step of both purposes writes each option in full
     and gives it once (reading L12), so the rule reads each option at its one place as the
     command's parser does. A missing, a second or
     an extra step is a fault naming the purpose and, when there is one, the step.
Every option of every step of the sealed run is written in full and given once, unless the
command declares it repeatable (reading L12).
THE START CHECK (reading L16). Once the list keeps every rule above, the command line starts each
step of the sealed run that runs a command of the sealed route, by the command's own code, at a
made-up tag in a made-up repository holding the checked-out commit and no data, up to its first
read of data (`run_starts`, its module text); any refusal before that read is a fault. It catches
what each command refuses at its start beyond rule 5: `compare_run --splits` other than 1,
`corner_run`'s seed, days and methods, `drift_run --etas` and `--calibration-days` and its
step-size file at the tag, a model that reads other days' outcomes, a dial not of the matrix's
family, a model that cannot be loaded. `--no-starts` leaves it out (for made-up lists whose
commands are stand-ins); the line `starts: ...` then says so. The driver makes the same start
check before its first step, for either purpose, with no option that leaves it out (`run_driver`,
module text).
Not checked:
- what comes before the program (`uv run ...`, or a shell wrapper that runs another command
  first). A wrapper written to run one command while the list names another is deliberate
  evasion, which these rules, like the route's checks, do not defend against (`sealed_route`,
  module text, first paragraph);
- options built at run time: a value only a run gives (`{run:<label>}`) is taken as given, never
  read, and an option a command adds by a name built at run time is not seen (reading L12);
- the route's refusals after its deciding function looks at the file system: a folder in a
  held-out place (`sealed_route.check_folders`), the gate itself (checked by the driver at its
  start, `run_driver`), and everything a command refuses once it starts opening data (a table
  folder or run that is missing or of another setting, a split rule, a day that cannot be
  built);
- by the rules above, each command's own checks of its other options, before or after the
  route's decision: the start check makes them, by the commands' own code, for the commands of
  the sealed route; the commands of the list that are not of the sealed route are not started.

Output: one line `FAULT: <where>: <what>` per fault and a last line `faulty: ...`; or, for a
well-formed list, one line `difference: <where>: <what>` per difference between the purposes
(`differences`) and a last line `well formed: ...` with the number of entries of each purpose,
the sum of their time limits and the number of differences, after a line `starts: ...` saying
whether the starts were made; or `COULD NOT RUN: ...`.

Exit status: 0 well formed; 1 faulty; 2 could not decide (the file is missing or cannot be read,
the package's modules cannot be listed, the file system's limit on a name cannot be read, or rule 3
must read a matrix and no git working copy holds the list, or a start cannot be decided:
`run_starts`). An unexpected error is 2 too, never Python's own 1.

Readings (alternatives in brackets):

  L1. Unit kinds. A unit is of the `development` kind when it is a training or a development unit
      of the truth build (`truth_build.TRAIN_UNITS`, `truth_build.DEVELOPMENT_UNITS`): open data,
      fitted on or scored before the tag. Every other name counts as the other kind, `sealed`.
      The full list of the sealed units is held by a module whose name only the files on the
      allow list of the repository rule on the sealed data may write (`tools/repo_rules.py`), so
      this module does not import it. A rehearsal group that names any other unit is therefore
      faulty (a misspelt development unit among them), but a misspelt sealed unit in a group of
      the sealed run is not caught here. [Alternative: take the sealed units' names from the
      module of the cold start's gate, which may name them, and refuse a name of neither kind.]
  L2. A fixed value or a group that no entry uses is a fault: a seed or a count fixed for the run
      and used by no command would not be applied. [Alternative: allow it and list it.]
  L3. Text that is not JSON, and the numbers NaN and Infinity, make a list faulty (exit 1), not
      undecidable: the file is there and its text is wrong. [Alternative: exit 2.]
  L4. The units reach a command through `{units}` alone, but the checker does not read the other
      arguments for unit names (a unit written out in an argument, or a matrix that names units,
      is not caught), nor compare an entry's setting with its arguments: those are the command's
      own inputs, which it checks when it runs. [Alternative: refuse any argument that is a unit
      name, the sealed ones taken as in L1's alternative.]
  L5. The module lives under `src/fleetreserve/`, not `tools/`, because the driver imports it to
      refuse a faulty list before it runs anything.
  L6. A label's run folder name is built as the recorder builds it (`run_record.run_folder_name`),
      and its length in bytes is held against the file system's own limit on a name
      (`os.pathconf(<folder>, "PC_NAME_MAX")`), read for the folder where the run folders will
      be: for the driver and the WP12 check, the working copy's `runs.noindex/` (its top folder
      before the first run); for the command line, the list's own folder, which in a working copy
      is on the same file system. The length of the whole path (the working copy's own path, the
      runs folder, the run folder's name and a file in it) is not checked: the list does not
      know where it will be driven. [Alternative: a fixed limit of 255 bytes; or the driver also
      checking the whole path against `PC_PATH_MAX`.]
  L7. Two labels are one when they are equal after Unicode caseless matching (`label_key`: NFD,
      case folding, NFD), whatever file system the list is driven on: on one that ignores letter
      case, two entries starting in one UTC second would get one run folder and the second would
      be refused part-way through a drive. A label is letters, digits, '-' and '_' of ASCII, so
      the normalisation changes nothing today; it is there for a wider label rule. [Alternative:
      refuse them only where the runs folder's file system ignores case, found by a probe file.]
  L8. A difference between the purposes is one of a closed list of three: the units of a group
      (each purpose has its own groups under the same names), an entry marked for one purpose
      only (`only`), and arguments marked for one purpose only (`extra`, added at the end of the
      command). Everything else is one text for both purposes, so a module, a fixed value, a
      setting, a time limit, a label, the order or a dependency cannot differ between them: a
      step of the run from the tag that needs a longer time limit than its rehearsal gets that
      limit in both. The checker lists the differences and judges none of them.
      [Alternatives: two lists, compared entry by entry, which the reviewed form before this one
      did and no real pair could pass, since the run from the tag needs other modules and
      arguments; or one list with a table of corresponding modules and of the arguments allowed
      to differ.]
  L9. "The held-out scripts on development units" (WP12) is the list's rehearsal instantiation:
      every entry not marked for the sealed run only, each with the same command, given the
      units of the rehearsal's groups (all of the development kind, L1) and none of the sealed
      run's own arguments. The entries and arguments of the sealed run only (the build of the
      sealed units' truth tables, the sealed route's arguments) are therefore not rehearsed:
      their code is tested by their own tests on made-up data alone. An entry of the rehearsal
      only is allowed, and listed as a difference. [Alternatives: rehearse the sealed run's own
      entries and arguments too, pointed at development tables, which needs those commands to
      accept them; or refuse entries of the rehearsal only.]
  L10. `{run:<label>}` is the run folder that the entry `<label>` recorded earlier in the same
      drive, as its recorder reported it, from the top folder (`runs.noindex/<UTC start>-<label>`),
      since every command runs from the top folder. A run of another drive is never named this
      way. [Alternative: the folder's absolute path; or the newest run with that label.]
  L11. The sealed truth build is the module `sealed_route.TRUTH_BUILD_MODULE` names (this
      module may not write that name: the repository rule on the sealed data lets only the
      files on its allow list name it, `tools/repo_rules.py`). Each entry a sealed
      `--held-out-truth-run` names must run it, and every sealed entry that runs it, named or
      not (a build of the held setting), comes before the sealed run's first entry that reads
      the sealed units' tables: one that gives `--held-out-truth-run` or `--held-out-tables`. A
      command of the sealed route that gives neither refuses itself when it runs, since the
      route requires the build's run. The constant is imported when a list is parsed, for
      either purpose, which loads the route's modules and calls nothing of the gate.
      [Alternatives: the checker on the seal rule's allow list; or order by module, every
      module of the sealed route after every truth build.]
  L12. A command's long options are the literal texts its own source file passes to
      `add_argument`, read with `ast` from the package's file and never imported (one parse
      per module, a few milliseconds). On every step of the sealed run and of the rehearsal
      (whose results are read before the tag, and whose truth builds the O2 steps read) an
      argument that is a shortened form of one of them (`--mat` for `--matrix`, which
      `argparse` reads as `--matrix`) is a fault, and so is one of them given twice unless its
      `action` repeats (`append`, `count`, ...), since `argparse` keeps the last; the names one
      `add_argument` call gives (`--run` and `--truth-run` of `ceiling_report`), or calls with
      one literal `dest`, are one option, so a step giving two of them gives it twice
      (`option_dests`); the rules then read each option at its one place. A flag given twice
      (`--every-day`), which `argparse` reads as once, is a fault too. An option a command
      adds by a name built at run time, or through another module, is not seen, so an
      argument that shortens it is not caught, and one that is a prefix of a seen option is a
      fault even if the command would read it whole. An
      argument that names no option of the command is left to the command. [Alternative: build
      each command's own parser, which imports every module of the list, model packages
      included, into the checker's process, about seconds per list and the modules' import
      effects; most commands build their parser inside `main`, so it cannot be taken without
      running it.]
  L13. The interval seed is asked of every `learn_run` and `gate_run` on the sealed route,
      not only of the one whose run the headline is read from, since no field of the list
      says which that is and every slot of plan section 11 has its interval; a `gate_run`
      refuses itself on the route when it runs (`sealed_route` reading S3), so a list whose
      `gate_run` keeps this rule is still faulty by rule 5. The resamples are
      `day_bootstrap.RESAMPLES` (the owner's 2,000).
      [Alternatives: ask it only of the runs a sealed `day_intervals` entry reads; or allow
      a list with no `day_intervals` entry.]
  L14. The nine held-out units are their one home's (`sealed_route.HELD_OUT_UNITS`), read
      through the cold start's names (`cold_start_gate`), which this module may name; the import
      loads the route's modules but calls nothing of the gate. Each `learn_run` whose run a
      sealed `day_intervals` entry reads must score all nine, not only the one the headline is
      read from: no field of the list says which that is, and every slot of plan section 11 is
      read over the held-out units. A matrix is read as the checked-out commit holds it
      (`sealed_route.committed_text`, git's objects), never from the working copy, after the
      path checks a run makes before it opens a path argument (`gate_run.argument_path`,
      `gate_run.check_files`; a path written with `~` is a fault, since it names no committed
      file). So a matrix path inside the sealed folder, or anywhere a run may not open, is
      refused and no file under it is opened; the commit's file is the one a drive, on a clean
      tree, reads. A `learn_run` with no `--matrix` reads its default, the committed development
      matrix (`learn_run`, module text): rule 3 does not read it (it is a fault of rule 3 when a
      `day_intervals` reads its run), and rule 5 does. [Alternatives: refuse only when
      the nine are split over several runs, leaving a run of fewer units; or read the matrix
      from the working copy after the same path checks.]
  L15. The route's commands are the modules under `src/` that reach the sealed-data gate, as
      the scan of `tests/test_run_evidence_gate_callers.py` finds them, less the held-out truth
      build (rule 1's build, which passes the gate itself) and the driver; no other home lists
      them, so `ROUTE_COMMANDS` lists each with the deciding function it calls, and a test pins
      the table to the scan (`tests/wp12/test_run_list_route_parity.py`). A command's default
      (`--matrix`, `--units`) is the expression its own `add_argument` call gives, evaluated in
      the imported module, a path under the package's top folder taken from the working copy
      the commands run from (`_default_text`). What the functions touch when called here:
      `held_out_kind` and `check_units` read their arguments only; `matrix_route` and
      `units_route` read their arguments only on the branches called (a mix, or no held-out
      unit), which return or refuse before `check_folders` (which resolves and stats the
      folders, the data folder's own included) and `_gate` (git and the running code). The
      copied rule for held-out units (the truth run; `gate_run`'s refusal) is pinned against
      the real commands' starts by that test, run at a made-up tag. [Alternatives: run each
      command's start in a made-up repository at a made-up tag, which needs every option's
      folders made up; or call the deciding functions on held-out units too, which stats the
      real data folder and fails before the tag.] The first alternative is now made too, as the
      start check (L16).
  L16. The start check runs after the rules, from the command line and in the driver before its
      first step (`run_starts` reading T5); the WP12 check, which reads a drive's log, does not
      run it. [Alternative: the command line only, the list being frozen at the tag, which
      leaves a drive of a list never checked with the starts free to start its steps.]
  L17. "Every opening of sealed data in the first step that opens any" is read as the sealed
      truth builds following one another: a build is of one setting, so the main and the held
      setting's builds are two entries. [Alternative: one entry building both settings, which
      the sealed truth build does not do.]
  L18. A model reads other days' outcomes when its module is in `forecast_model.OUTCOME_READERS`,
      the list the route refuses by (`sealed_route` reading S1), named by `--model` or by
      `cold_start`'s `--reference`. [Alternative: rule 5's route alone, which refuses it only
      when the command starts.]
  L19. A step's site type is the one its dial's fixed name holds (`dial-workplaces`,
      `dial-others`), since `compare_run` and `corner_run` score every unit of their matrix at
      the one dial given and the list runs each once per site type, or once for both when the
      matrix gives both site types one dial (reading L22); a literal `--dial` is a
      fault, as its site type cannot be read. The rule applies to every step of the sealed run
      that gives `--dial` or `--kappa`, whatever its command; a matrix with no `sealed_choice`
      is a fault, as no dial of it can be checked. Values are compared as the commands read
      them (`float`). [Alternatives: compare a dial with either site type's dial, which passes a
      workplace step at the other type's dial; or take the site type from the step's label.]
  L20. The rehearsal runs on development units alone (WP12), and a command of the route refuses
      either held-out option on development units when it runs (`sealed_route.units_route`,
      `matrix_route`); the rule refuses it in the list, before any step. A `{run:<label>}` of the
      sealed truth build in the rehearsal is already a fault of the references rule; the rule also
      catches a literal path and the run of another build. [Alternative: rule 5's decision made
      on the rehearsal's steps too, which reads the rehearsal's matrices as well.]
  L21. O2 is stated over the held-out units (plan section 11), and the route's command refuses a
      held-out run without `--every-day` only when it runs (`ceiling_report`,
      `ROUTE_NEEDS_EVERY_DAY`), after the gate. A sealed `ceiling_report` step on development
      units alone and with no held-out option passes every other rule and would report O2 of
      development units as the sealed run's; one on only some of the nine would report O2 of
      fewer units. The nine are those of rule 3 (reading L14). With no `--units` the command's
      default, the development units, is read. [Alternative: every unit a held-out unit, some of
      the nine allowed.]
  L22. `compare_run` and `corner_run` score every unit of their matrix at the one dial given.
      Where `sealed_choice.dials` gives the site types one dial, one step of each stands for
      both site types: a second step at the other type's name would repeat the first exactly,
      and a name given twice (swapped by hand, the two values equal) is still a fault. Where the
      dials differ, one step per site type is needed (a step scores every unit at its own
      dial, so a lone one leaves the other site type's units at a dial that is not theirs),
      and a lone step passes rule 8, which holds a step only against the dial of its own name's
      site type, so rule 11 holds it against every site type's dial of the matrix. Only names
      are compared: a step whose dial is not a fixed name is rule 8's fault. [Alternatives:
      each of the two names exactly once, which refuses a list with one step at a shared dial,
      the form the owner's decision 103 of 22 November 2026 asks for; or the lone step passing
      whatever the dials, which leaves a site type's dial unchecked.]
  L23. The run from the tag and its rehearsal hold two O2 steps, `ceiling_report --every-day` at
      rho 0, one in the main and one in the held setting, and none at rho 0.1 or 0.2 (the owner's
      decision 79 (4) of 19 November 2026; plan section 11 names no rho and no setting for O2,
      and the held setting is run in full beside the main one for the main model, plan section
      3). A step's setting is the entry's `setting`. Its truth runs and sealed tables are the
      command's own inputs (reading L4), but a held step without `--held-out-tables`, or given
      the main setting's truth run, would read the other setting's truth, which the command
      refuses only once it runs, after the sealed truth builds (the review of 2054956, finding
      2), so they are held against the setting here; a truth run given as a literal path names
      no build of the list and is a fault too. Its development `--tables` folder is left to the
      command (reading L4), which refuses tables of another setting than its truth build's,
      held here to the step's. The rule is
      checked of a list that runs a `ceiling_report` step in some purpose: a list with none
      (made-up lists of other commands) is not held to it, and the committed list's two entries
      are held by its own test (`tests/wp12/test_run_list_file.py`). [Alternatives: the rule on
      every list, which refuses a list with no O2 step; or per group of units rather than per
      purpose.]
"""

import argparse
import ast
import importlib
import json
import math
import os
import pkgutil
import re
import subprocess
import sys
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import fleetreserve
from fleetreserve import gate_config
from fleetreserve.day_bootstrap import RESAMPLES  # the resamples of the intervals over days
from fleetreserve.run_record import _LABEL as RUN_RECORD_LABEL  # the recorder's own label rule
from fleetreserve.run_record import RUNS_FOLDER, run_folder_name, utc_now
from fleetreserve.truth import SETTINGS
from fleetreserve.truth_build import DEVELOPMENT_UNITS, TRAIN_UNITS

FORMAT = "run-list-2"
REHEARSAL = "rehearsal"
SEALED_RUN = "sealed-run"
PURPOSES = (REHEARSAL, SEALED_RUN)
DEVELOPMENT = "development"
SEALED = "sealed"
PACKAGE = fleetreserve.__name__
OPEN_UNITS = frozenset((*TRAIN_UNITS, *DEVELOPMENT_UNITS))  # reading L1
TOP_FIELDS = ("format", "about", "units", "fixed", "entries")
ENTRY_FIELDS = ("label", "only", "command", "units", "extra", "setting", "max_seconds", "after")
# The three objects of `fixed`, the reference word of each, and the smallest whole number allowed
# (None: a text).
FIXED = {"seeds": ("seed", 0), "counts": ("count", 1), "names": ("name", None)}
UNITS = "{units}"
RUN = "run"
WELL_FORMED, FAULTY, COULD_NOT_RUN = 0, 1, 2
# What the sealed run's rules read (module text, THE SEALED RUN'S RULES)
TRUTH_RUN = "--held-out-truth-run"  # the sealed truth build's run, on the sealed route
TABLES = "--held-out-tables"  # the sealed truth build's tables, on the sealed route
INTERVALS_MODULE = "day_intervals"
RUN_OPTION, SEED_OPTION, RESAMPLES_OPTION = "--run", "--seed", "--resamples"  # day_intervals'
SCORING_MODULES = ("gate_run", "learn_run")  # the runs day_intervals reads (its module text)
LEARN_MODULE = "learn_run"
GATE_MODULE = "gate_run"
INTERVAL_SEED, INTERVAL_RESAMPLES = "--interval-seed", "--interval-resamples"
MATRIX_OPTION = "--matrix"
UNITS_OPTION = "--units"
# The commands of the sealed route, each with the deciding function of `sealed_route` its start
# calls (reading L15); a test pins the table to the modules that call them.
MATRIX_ROUTE, UNITS_ROUTE, CHECK_UNITS = "matrix_route", "units_route", "check_units"
ROUTE_COMMANDS = {
    "learn_run": MATRIX_ROUTE,
    "gate_run": MATRIX_ROUTE,
    "compare_run": MATRIX_ROUTE,
    "corner_run": MATRIX_ROUTE,
    "ledger_run": MATRIX_ROUTE,
    "drift_run": UNITS_ROUTE,
    "o3_run": UNITS_ROUTE,
    "ceiling_report": UNITS_ROUTE,
    "cold_start": CHECK_UNITS,
}
ROUTE_OPTIONS = ("--tables", "--truth-run", TABLES, TRUTH_RUN)  # the deciding functions' order
GATE_TRIALS = ("--limit-days", "--only")  # `gate_run`'s trial options, in its order

_NAME = re.compile(r"[A-Za-z0-9_-]+")
_PYTHON = re.compile(r"python(?:3(?:\.\d+)?)?")
_REFERENCE = re.compile(r"\{(seed|count|name|run):([A-Za-z0-9_-]+)\}")
_FIXED_OF = {word: block for block, (word, _) in FIXED.items()}


class CannotDecide(Exception):
    """The checker cannot decide the list (exit status 2)."""


class Faulty(Exception):
    """The list is faulty; `faults` holds one line per fault (exit status 1)."""

    def __init__(self, faults: list[str]) -> None:
        super().__init__(f"{len(faults)} fault{'s' if len(faults) != 1 else ''}")
        self.faults = faults


@dataclass(frozen=True)
class Entry:
    label: str
    only: str | None
    command: tuple[str, ...]
    units: str | None
    extra: dict  # purpose -> tuple of arguments
    setting: str
    max_seconds: float
    after: tuple[str, ...]

    def runs_in(self, purpose: str) -> bool:
        return self.only is None or self.only == purpose


@dataclass(frozen=True)
class RunList:
    about: str
    units: dict  # purpose -> group -> tuple of units
    fixed: dict  # block -> name -> value
    entries: tuple[Entry, ...]


@dataclass(frozen=True)
class Step:
    """One entry as one purpose runs it (`instantiate`)."""

    position: int  # the entry's place in the list, from 0
    label: str
    command: tuple[str, ...]  # filled in but for each `{run:<label>}`, which `runs` places
    runs: tuple[tuple[int, str], ...]  # (place in `command`, label) of each `{run:<label>}`
    setting: str
    max_seconds: float
    after: tuple[str, ...]


def step_modules() -> frozenset[str]:
    """The modules of the package (not its subpackages), by name."""
    try:
        found = [m for m in pkgutil.iter_modules(fleetreserve.__path__) if not m.ispkg]
    except Exception as problem:
        raise CannotDecide(f"the modules of {PACKAGE} cannot be listed: {problem}") from problem
    if not found:
        raise CannotDecide(f"no module of {PACKAGE} was found")
    return frozenset(m.name for m in found)


def name_limit(folder: Path) -> int:
    """The longest name, in bytes, that the file system of `folder` allows (reading L6)."""
    try:
        return os.pathconf(folder, "PC_NAME_MAX")
    except (OSError, ValueError) as problem:
        raise CannotDecide(
            f"the file system's limit on a name in {folder} cannot be read: {problem}"
        ) from problem


def run_folder_limit(top: Path) -> int:
    """`name_limit` for the run folders of the working copy at `top` (reading L6)."""
    runs = Path(top) / RUNS_FOLDER
    return name_limit(runs if runs.is_dir() else Path(top))


def _label_faults(label, name_max: int) -> list[str]:
    """Why `run_record`, called as the driver calls it, would not record a run under `label`."""
    if not isinstance(label, str) or not RUN_RECORD_LABEL.fullmatch(label):
        return [f"its label {label!r} is refused by run_record"]
    size = len(run_folder_name(utc_now(), label).encode("utf-8"))
    if size > name_max:
        return [
            f"its run folder name would be {size} bytes long, more than the {name_max} bytes the "
            "file system allows for a name (<UTC start>-<label>, reading L6)"
        ]
    return []


def unit_kind(name: str) -> str:
    """`DEVELOPMENT` for a training or development unit, `SEALED` for any other name (L1)."""
    return DEVELOPMENT if name in OPEN_UNITS else SEALED


def _is_whole(value, least: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= least


def _limit_fault(value) -> str | None:
    """Why `value` is no time limit the recorder reads back exactly (`--max-seconds` is read as a
    float, and the WP12 check holds the manifest's float against the list's value); None when it
    is one."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "is not a positive finite number of seconds"
    try:
        seconds = float(value)
    except OverflowError:
        return "is too large to be read as a number of seconds"
    if not math.isfinite(seconds) or seconds <= 0:
        return "is not a positive finite number of seconds"
    if seconds != value:
        return f"is not read back exactly: it reads back as {seconds!r} seconds"
    return None


def label_key(label: str) -> str:
    """The label as a file system that ignores letter case and Unicode normalisation sees it
    (caseless matching: NFD, case folding, NFD); two labels with one key may name one run folder."""
    return unicodedata.normalize("NFD", unicodedata.normalize("NFD", label).casefold())


def _field_faults(document, fields: tuple[str, ...], where: str) -> list[str]:
    if not isinstance(document, dict):
        return [f"{where}: is not a JSON object"]
    found = [f"{where}: lacks the field {name!r}" for name in fields if name not in document]
    found += [f"{where}: has the unknown field {name!r}" for name in document if name not in fields]
    return found


def _fixed_faults(fixed) -> list[str]:
    found = _field_faults(fixed, tuple(FIXED), "fixed")
    if found:
        return found
    for block, (_, least) in FIXED.items():
        values = fixed[block]
        if not isinstance(values, dict):
            found.append(f"fixed.{block}: is not a JSON object")
            continue
        for name, value in values.items():
            if not _NAME.fullmatch(name):
                found.append(f"fixed.{block}: the name {name!r} is not letters, digits, - and _")
            if least is None:
                if not isinstance(value, str) or not value:
                    found.append(f"fixed.{block}.{name}: is not a non-empty text")
            elif not _is_whole(value, least):
                found.append(f"fixed.{block}.{name}: is not a whole number >= {least}")
    return found


def _units_faults(block) -> list[str]:
    """Faults of the list's `units` (module text)."""
    found = _field_faults(block, PURPOSES, "units")
    if found:
        return found
    for purpose in PURPOSES:
        groups = block[purpose]
        if not isinstance(groups, dict):
            found.append(f"units.{purpose}: is not a JSON object")
            continue
        for name, units in groups.items():
            where = f"units.{purpose}.{name}"
            if not _NAME.fullmatch(name):
                found.append(f"units.{purpose}: the name {name!r} is not letters, digits, - and _")
            if (
                not isinstance(units, list)
                or not units
                or not all(isinstance(u, str) and u for u in units)
                or len(set(units)) != len(units)
            ):
                found.append(f"{where}: is not a non-empty list of distinct unit names")
                continue
            other = [u for u in units if unit_kind(u) == SEALED]
            if purpose == REHEARSAL and other:
                found.append(
                    f"{where}: names {', '.join(other)}, which is no training or development "
                    "unit: the rehearsal runs on those alone (reading L1)"
                )
            elif other and len(other) != len(units):
                found.append(
                    f"{where}: mixes the two kinds of units: training or development units and "
                    f"others ({', '.join(other)})"
                )
    return found


def _is_texts(value) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(p, str) and p for p in value)


def _brace_faults(parts: list[str], *, units_allowed: bool) -> list[str]:
    found = []
    for part in parts:
        if "{" not in part and "}" not in part:
            continue
        if part == UNITS:
            if not units_allowed:
                found.append(f"its extra arguments hold {UNITS}, which only its command may hold")
        elif not _REFERENCE.fullmatch(part):
            found.append(
                f"its argument {part!r} holds a brace but is not one placeholder {UNITS}, "
                "{seed:<name>}, {count:<name>}, {name:<name>} or {run:<label>}"
            )
    return found


def _module_faults(command: list[str], modules: frozenset[str]) -> list[str]:
    prefix = f"{PACKAGE}."
    calls = [
        k
        for k in range(len(command) - 1)
        if command[k] == "-m" and command[k + 1].startswith(prefix)
    ]
    if (
        len(calls) != 1
        or calls[0] == 0
        or not _PYTHON.fullmatch(command[calls[0] - 1].split("/")[-1])
    ):
        return [f"its command is not one `python -m {PACKAGE}.<module>` call"]
    module = command[calls[0] + 1][len(prefix) :]
    if module not in modules:
        return [f"its command runs {prefix}{module}, which is no module of the package"]
    return []


def _references(parts) -> list[tuple[str, str]]:
    """(word, name) of every placeholder `{<word>:<name>}` among `parts`."""
    found = []
    for part in parts:
        match = _REFERENCE.fullmatch(part)
        if match:
            found.append((match.group(1), match.group(2)))
    return found


def _purposes_of(entry) -> frozenset[str] | None:
    """The purposes an entry runs in, or None when its `only` is missing or not of its form."""
    if not isinstance(entry, dict) or "only" not in entry:
        return None
    only = entry["only"]
    if only is None:
        return frozenset(PURPOSES)
    return frozenset((only,)) if only in PURPOSES else None


def _shown(purposes) -> str:
    return " and the ".join(p for p in PURPOSES if p in purposes)


class _Context:
    """What the faults of one entry are read against: the list's earlier entries and blocks."""

    def __init__(self, document, modules: frozenset[str], name_max: int) -> None:
        self.modules = modules
        self.name_max = name_max
        units = document.get("units")
        self.units = units if "units" in document and not _units_faults(units) else None
        fixed = document.get("fixed")
        self.fixed = fixed if "fixed" in document and not _fixed_faults(fixed) else None
        self.labels: list = []
        self.purposes: list = []
        self.used_fixed: set[tuple[str, str]] = set()
        self.used_groups: set[tuple[str, str]] = set()


def _earlier_faults(kind: str, name: str, label, used_in, context: _Context) -> list[str]:
    """Faults of a dependency (`kind` "depends on") or of a run placeholder (`kind` "refers to the
    run of") on the entry `name`, used in the purposes `used_in`."""
    if name == label:
        return ["depends on itself" if kind == "depends on" else "refers to its own run"]
    if name not in context.labels:
        return [f"{kind} {name!r}, which is no earlier entry (a later or a missing one)"]
    theirs = context.purposes[context.labels.index(name)]
    if theirs is None or used_in is None:
        return []
    missing = [p for p in PURPOSES if p in used_in and p not in theirs]
    if missing:
        return [f"{kind} {name!r} in the {_shown(missing)}, where {name!r} does not run"]
    return []


def _entry_faults(entry, k: int, context: _Context) -> list[str]:
    """Faults of entry `k` (0-based) of the list (module text)."""
    where = f"entry {k + 1}"
    found = _field_faults(entry, ENTRY_FIELDS, where)
    if not isinstance(entry, dict):
        return found
    label = entry.get("label")
    if isinstance(label, str):
        where = f"entry {k + 1} ({label})"
    found = [f"{where}{line[len(f'entry {k + 1}') :]}" for line in found]
    runs_in = _purposes_of(entry)
    own: list[str] = []
    if "label" in entry:
        refused = _label_faults(label, context.name_max)
        keys = [label_key(a) if isinstance(a, str) else None for a in context.labels]
        if refused:
            own += refused
        elif label in context.labels:
            own.append("its label is that of an earlier entry")
        elif label_key(label) in keys:
            j = keys.index(label_key(label))
            own.append(
                f"its label is that of an earlier entry (entry {j + 1}, {context.labels[j]}) when "
                "letter case is ignored: on a file system that ignores it the two run folders can "
                "have one name (reading L7)"
            )
    if "only" in entry and runs_in is None:
        own.append(f"its only {entry['only']!r} is not null or one of {', '.join(PURPOSES)}")
    command = entry.get("command")
    command_ok = "command" in entry and _is_texts(command)
    if "command" in entry and not command_ok:
        own.append("its command is not a non-empty list of non-empty texts")
    if command_ok:
        own += _module_faults(command, context.modules)
        own += _brace_faults(command, units_allowed=True)
    group = entry.get("units")
    if "units" in entry:
        if group is not None and (not isinstance(group, str) or not _NAME.fullmatch(group)):
            own.append(f"its units {group!r} is not null or the name of a group")
        elif group is not None:
            if command_ok and UNITS not in command:
                own.append(f"names the units group {group!r} but its command has no {UNITS}")
            for purpose in PURPOSES if runs_in is None else [p for p in PURPOSES if p in runs_in]:
                context.used_groups.add((purpose, group))
                if context.units is not None and group not in context.units[purpose]:
                    own.append(f"names the units group {group!r}, which units.{purpose} lacks")
        elif command_ok and UNITS in command:
            own.append(f"its command holds {UNITS} but it names no units group")
    extra = entry.get("extra")
    uses = [(runs_in, command)] if command_ok else []
    if "extra" in entry:
        if not isinstance(extra, dict):
            own.append("its extra is not a JSON object")
        elif extra and runs_in is not None and len(runs_in) == 1:
            own.append(
                f"runs in the {_shown(runs_in)} only, so its arguments go in its command and its "
                "extra is {}"
            )
        else:
            for purpose, parts in extra.items():
                if purpose not in PURPOSES:
                    own.append(f"its extra names {purpose!r}, which is no purpose")
                elif not _is_texts(parts):
                    own.append(
                        f"its extra arguments for the {purpose} are not a non-empty list of "
                        "non-empty texts"
                    )
                else:
                    own += _brace_faults(parts, units_allowed=False)
                    uses.append((frozenset((purpose,)), parts))
    if "setting" in entry and entry["setting"] not in SETTINGS:
        own.append(f"its setting {entry['setting']!r} is not one of {', '.join(SETTINGS)}")
    if "max_seconds" not in entry:
        own.append("has no time limit (max_seconds)")
    elif (why := _limit_fault(entry["max_seconds"])) is not None:
        own.append(f"its time limit {entry['max_seconds']!r} {why}")
    if "after" in entry:
        after = entry["after"]
        if not isinstance(after, list) or not all(isinstance(a, str) for a in after):
            own.append("its after is not a list of labels")
        else:
            for name in after:
                own += _earlier_faults("depends on", name, label, runs_in, context)
    for used_in, parts in uses:
        for word, name in _references(parts):
            if word == RUN:
                kind = "refers to the run of"
                own += _earlier_faults(kind, name, label, used_in, context)
            elif context.fixed is not None:
                block = _FIXED_OF[word]
                context.used_fixed.add((block, name))
                if name not in context.fixed[block]:
                    own.append(f"names the {word} {name!r}, which fixed.{block} lacks")
    return found + [f"{where}: {what}" for what in own]


def faults(document, *, name_max: int) -> list[str]:
    """Every fault of a parsed run list, one line each (module text); empty when well formed.
    `name_max` is the file system's limit on a name where its runs will be (`name_limit`)."""
    found = _field_faults(document, TOP_FIELDS, "the list")
    if not isinstance(document, dict):
        return found
    if "format" in document and document["format"] != FORMAT:
        found.append(f"the list: its format {document['format']!r} is not {FORMAT!r}")
    if "about" in document and not isinstance(document["about"], str):
        found.append("the list: its about is not a text")
    if "units" in document:
        found += _units_faults(document["units"])
    if "fixed" in document:
        found += _fixed_faults(document["fixed"])
    entries = document.get("entries")
    if "entries" not in document:
        return found
    if not isinstance(entries, list) or not entries:
        return [*found, "the list: its entries are not a non-empty list"]
    context = _Context(document, step_modules(), name_max)
    for k, entry in enumerate(entries):
        found += _entry_faults(entry, k, context)
        context.labels.append(entry.get("label") if isinstance(entry, dict) else None)
        context.purposes.append(_purposes_of(entry))
    if context.fixed is not None:
        for block in FIXED:
            for name in context.fixed[block]:
                if (block, name) not in context.used_fixed:
                    found.append(f"fixed.{block}.{name}: no entry refers to it (reading L2)")
    if context.units is not None:
        for purpose in PURPOSES:
            for name in context.units[purpose]:
                if (purpose, name) not in context.used_groups:
                    found.append(
                        f"units.{purpose}.{name}: no entry of the {purpose} names it (reading L2)"
                    )
    if all(p is not None for p in context.purposes):
        for purpose in PURPOSES:
            if not any(purpose in p for p in context.purposes):
                found.append(f"the list: no entry runs in the {purpose}")
    return found


def parse(document, *, name_max: int) -> RunList:
    """The run list of a parsed JSON document; raises `Faulty` (or `CannotDecide`). `name_max` as
    in `faults`."""
    found = faults(document, name_max=name_max)
    if found:
        raise Faulty(found)
    entries = tuple(
        Entry(
            label=e["label"],
            only=e["only"],
            command=tuple(e["command"]),
            units=e["units"],
            extra={purpose: tuple(parts) for purpose, parts in e["extra"].items()},
            setting=e["setting"],
            max_seconds=e["max_seconds"],
            after=tuple(e["after"]),
        )
        for e in document["entries"]
    )
    units = {
        purpose: {name: tuple(group) for name, group in document["units"][purpose].items()}
        for purpose in PURPOSES
    }
    fixed = {block: dict(values) for block, values in document["fixed"].items()}
    parsed = RunList(document["about"], units, fixed, entries)
    found = sealed_faults(parsed)
    if found:
        raise Faulty(found)
    return parsed


# --- the sealed run's rules (THE SEALED RUN'S RULES in the module text) ---------------------------


def _module_at(command: Sequence[str]) -> int:
    """The place in a well-formed command of its `fleetreserve.<module>` argument."""
    prefix = f"{PACKAGE}."
    return next(
        k + 1
        for k in range(len(command) - 1)
        if command[k] == "-m" and command[k + 1].startswith(prefix)
    )


def step_module(step: "Step") -> str:
    """The module a step runs, without the package's name."""
    return step.command[_module_at(step.command)][len(PACKAGE) + 1 :]


@dataclass(frozen=True)
class _Option:
    place: int | None  # the place of its value in the step's command; None when it has none
    value: str | None  # its value as instantiated; None when it has none


def step_options(step: "Step", names: Sequence[str]) -> dict:
    """The options `names` as the step's command gives them, `--name value` or `--name=value`,
    read after its module: name -> list of `_Option`. Whether the command's options are written
    in full and once is `option_form_faults`' (reading L12)."""
    found: dict[str, list[_Option]] = {name: [] for name in names}
    command = step.command
    k = _module_at(command) + 1
    while k < len(command):
        part = command[k]
        if part == "--":
            break
        key, equals, value = part.partition("=")
        if key in found:
            if equals:
                found[key].append(_Option(None, value))
            elif k + 1 < len(command):
                found[key].append(_Option(k + 1, command[k + 1]))
                k += 1
            else:
                found[key].append(_Option(None, None))
        k += 1
    return found


_TREES: dict[str, ast.Module] = {}
_OPTIONS: dict[str, tuple[frozenset[str], frozenset[str]]] = {}
_REPEATABLE = ("append", "append_const", "count", "extend")


def _source_tree(module: str) -> ast.Module:
    """The module's source file parsed with `ast`, never imported (reading L12)."""
    if module not in _TREES:
        path = Path(fleetreserve.__path__[0]) / f"{module}.py"
        try:
            _TREES[module] = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError) as problem:
            raise CannotDecide(
                f"the options of {PACKAGE}.{module} cannot be read: {problem}"
            ) from problem
    return _TREES[module]


def _additions(module: str):
    """(the long options given as literal texts, the call) of each `add_argument` call."""
    for node in ast.walk(_source_tree(module)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            longs = [
                a.value
                for a in node.args
                if isinstance(a, ast.Constant)
                and isinstance(a.value, str)
                and a.value.startswith("--")
            ]
            yield longs, node


def command_options(module: str) -> tuple[frozenset[str], frozenset[str]]:
    """(the long options, those of them that may be given more than once) that the module's
    source text passes to `add_argument` as literal texts, read from its file with `ast`, never
    imported (reading L12)."""
    if module not in _OPTIONS:
        names: set[str] = set()
        repeatable: set[str] = set()
        for longs, node in _additions(module):
            names.update(longs)
            for keyword in node.keywords:
                if (
                    keyword.arg == "action"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value in _REPEATABLE
                ):
                    repeatable.update(longs)
        _OPTIONS[module] = (frozenset(names), frozenset(repeatable))
    return _OPTIONS[module]


_DESTS: dict[str, dict[str, str]] = {}


def option_dests(module: str) -> dict[str, str]:
    """Each long option of `command_options` -> the destination `argparse` stores it in: the
    call's `dest` when a literal text, else its first long option without the leading dashes,
    '-' read as '_'. Names of one destination (`--run` and `--truth-run`) are one option."""
    if module not in _DESTS:
        found: dict[str, str] = {}
        for longs, node in _additions(module):
            if not longs:
                continue
            dest = longs[0].lstrip("-").replace("-", "_")
            for keyword in node.keywords:
                if keyword.arg == "dest" and isinstance(keyword.value, ast.Constant):
                    dest = str(keyword.value.value)
            found.update(dict.fromkeys(longs, dest))
        _DESTS[module] = found
    return _DESTS[module]


_DEFAULT_NODES = (ast.Expression, ast.Name, ast.Attribute, ast.Call, ast.Constant, ast.Load)
_DEFAULT_NODES += (ast.List, ast.Tuple, ast.BinOp, ast.Div)


def command_default(module: str, option: str):
    """The default the module's source gives `option` in its `add_argument` call, evaluated in
    the imported module (reading L15): a constant, or an expression of the module's names
    (`MATRIX`, `dial_choice.MATRIX`, `list(TRAIN_UNITS)`); None when it gives none. Raises
    `CannotDecide` for another form."""
    for longs, node in _additions(module):
        if option not in longs:
            continue
        for keyword in node.keywords:
            if keyword.arg != "default":
                continue
            expression = ast.Expression(keyword.value)
            if not all(isinstance(n, _DEFAULT_NODES) for n in ast.walk(expression)):
                raise CannotDecide(
                    f"the default of {option} of {PACKAGE}.{module} is not a form this check "
                    f"reads: {ast.unparse(keyword.value)}"
                )
            namespace = vars(importlib.import_module(f"{PACKAGE}.{module}"))
            code = compile(expression, f"<{module} {option}>", "eval")
            return eval(code, {"__builtins__": {"list": list, "tuple": tuple}}, namespace)
        return None
    return None


def option_form_faults(
    steps: Sequence["Step"], said: str = "on the sealed run", why: str = "(reading L12)"
) -> list[str]:
    """Every option of every step (the sealed run's, or `said` of others) written in a shortened
    form or given more than once, the names of one destination counted as one option
    (`argparse` would read the first as the longer option and keep the last of the second), one
    line each (reading L12)."""
    problems = []
    for s in steps:
        names, repeatable = command_options(step_module(s))
        dests = option_dests(step_module(s))
        command = s.command
        counts: dict[str, int] = {}
        given: dict[str, set[str]] = {}
        for part in command[_module_at(command) + 1 :]:
            if part == "--":
                break
            key = part.partition("=")[0]
            if not key.startswith("--") or len(key) <= 2:
                continue
            if key in names:
                dest = dests.get(key, key)
                counts[dest] = counts.get(dest, 0) + 1
                given.setdefault(dest, set()).add(key)
                continue
            shortened = sorted(name for name in names if name.startswith(key))
            if shortened:
                problems.append(
                    f"{_where(s)}: its argument {part!r} shortens {' or '.join(shortened)}: {said} "
                    f"every option is written in full {why}"
                )
        for dest, count in sorted(counts.items()):
            keys = sorted(given[dest])
            if count > 1 and not set(keys) & repeatable:
                shown = keys[0] if len(keys) == 1 else f"{' and '.join(keys)}, one option,"
                problems.append(
                    f"{_where(s)}: gives {shown} {count} times: {said} an option is given once "
                    f"{why}"
                )
    return problems


def _one(found: dict, name: str) -> "_Option | None":
    given = found[name]
    return given[0] if len(given) == 1 else None


def _run_of(step: "Step", option: "_Option | None") -> str | None:
    """The label of the `{run:<label>}` that is the option's value, or None."""
    if option is None or option.place is None:
        return None
    return dict(step.runs).get(option.place)


def _where(step: "Step") -> str:
    return f"entry {step.position + 1} ({step.label})"


def _builds_first_faults(steps: Sequence["Step"]) -> list[str]:
    """Rule 1 (reading L11): on the sealed run each `--held-out-truth-run` is the run of an
    earlier entry, and every such run comes before the first entry that reads the sealed units'
    tables."""
    problems = []
    builds: dict[str, Step] = {}  # label -> the step named as a truth run
    readers: list[Step] = []
    by_label = {s.label: s for s in steps}
    for s in steps:
        found = step_options(s, (TRUTH_RUN, TABLES))
        if found[TRUTH_RUN] or found[TABLES]:
            readers.append(s)
        for option in found[TRUTH_RUN]:
            label = _run_of(s, option)
            if label is None:
                problems.append(
                    f"{_where(s)}: on the sealed run its {TRUTH_RUN} is {option.value!r}, a "
                    "literal path, where {run:<label>} of the sealed truth build belongs, so the "
                    "list's order cannot hold the build before it (rule 1, reading L11)"
                )
            elif label in by_label:
                builds[label] = by_label[label]
    # the module of the sealed truth build, by the name the route holds (reading L11)
    from fleetreserve.sealed_route import TRUTH_BUILD_MODULE

    for build in builds.values():
        if step_module(build) != TRUTH_BUILD_MODULE:
            problems.append(
                f"{_where(build)}: is named by a {TRUTH_RUN} of the sealed run but runs "
                f"{PACKAGE}.{step_module(build)}, not the sealed truth build "
                f"{PACKAGE}.{TRUTH_BUILD_MODULE} (rule 1, reading L11)"
            )
    for s in steps:
        if step_module(s) == TRUTH_BUILD_MODULE:
            builds.setdefault(s.label, s)
    if readers:
        first = readers[0]
        for build in sorted(builds.values(), key=lambda s: s.position):
            if build.position >= first.position:
                problems.append(
                    f"{_where(build)}: a sealed truth build comes at or after {_where(first)}, "
                    "the sealed run's first entry that reads the sealed units' tables: every "
                    "truth build comes before it, so a file that cannot be parsed stops the drive "
                    "before any scoring (rule 1, reading L11)"
                )
    return problems


def _pooled_run_faults(steps: Sequence["Step"]) -> list[str]:
    """Rule 3, the part read from the list alone (reading L14): on the sealed run each
    `day_intervals` entry reads the run of an earlier entry by `{run:<label>}`, so the run whose
    units it pools can be found."""
    problems = []
    for s in steps:
        if step_module(s) != INTERVALS_MODULE:
            continue
        found = step_options(s, (RUN_OPTION,))
        for option in found[RUN_OPTION]:
            if _run_of(s, option) is None:
                problems.append(
                    f"{_where(s)}: on the sealed run its {RUN_OPTION} is {option.value!r}, a "
                    "literal path, where {run:<label>} of the scoring entry belongs, so the run "
                    "whose units it pools cannot be read from the list (rule 3, reading L14)"
                )
    return problems


def _interval_value(option: "_Option | None") -> int | None:
    if option is None or option.value is None or not re.fullmatch(r"\d+", option.value):
        return None
    return int(option.value)


def _interval_faults(steps: Sequence["Step"]) -> list[str]:
    """Rule 2 (reading L13): the sealed scoring runs carry the interval seed and resamples that
    the sealed `day_intervals` entries use."""
    problems = []
    by_label = {s.label: s for s in steps}
    seeds: dict[int, list[Step]] = {}
    wants: list[tuple[Step, int | None, int | None, str | None]] = []  # day_intervals' readings
    for s in steps:
        if step_module(s) != INTERVALS_MODULE:
            continue
        found = step_options(s, (RUN_OPTION, SEED_OPTION, RESAMPLES_OPTION))
        seed = _interval_value(_one(found, SEED_OPTION))
        given = _one(found, RESAMPLES_OPTION)
        resamples = RESAMPLES if not found[RESAMPLES_OPTION] else _interval_value(given)
        run = _one(found, RUN_OPTION)
        label = _run_of(s, run)
        if seed is None:
            problems.append(
                f"{_where(s)}: on the sealed run it gives no whole-number {SEED_OPTION} (rule 2, "
                "reading L13)"
            )
        else:
            seeds.setdefault(seed, []).append(s)
        if resamples != RESAMPLES:
            problems.append(
                f"{_where(s)}: on the sealed run its {RESAMPLES_OPTION} is not {RESAMPLES} "
                "(rule 2, reading L13)"
            )
        if label in by_label:
            wants.append((s, seed, resamples, label))
    if len(seeds) > 1:
        shown = "; ".join(
            f"{seed} in {', '.join(_where(s) for s in at)}" for seed, at in sorted(seeds.items())
        )
        problems.append(
            f"the sealed run: its {INTERVALS_MODULE} entries use more than one {SEED_OPTION} "
            f"({shown}); one seed S serves every interval (rule 2, reading L13)"
        )
    the_seed = next(iter(seeds)) if len(seeds) == 1 else None
    for s in steps:
        if step_module(s) not in SCORING_MODULES:
            continue
        found = step_options(s, (TRUTH_RUN, INTERVAL_SEED, INTERVAL_RESAMPLES))
        read_by = [w for w in wants if w[3] == s.label]
        if not found[TRUTH_RUN] and not read_by:
            continue  # neither on the sealed route nor read for intervals
        seed = _interval_value(_one(found, INTERVAL_SEED))
        resamples = _interval_value(_one(found, INTERVAL_RESAMPLES))
        if seed is None or resamples != RESAMPLES:
            problems.append(
                f"{_where(s)}: on the sealed run it scores without {INTERVAL_SEED} S "
                f"{INTERVAL_RESAMPLES} {RESAMPLES}, so the intervals of O1 and O4 would need its "
                "scoring made again (rule 2, reading L13)"
            )
            continue
        if not seeds:
            problems.append(
                f"{_where(s)}: no {INTERVALS_MODULE} entry of the sealed run gives the seed S of "
                f"its {INTERVAL_SEED}, so whether the intervals can read it cannot be decided "
                "(rule 2, reading L13)"
            )
        elif the_seed is not None and seed != the_seed:
            problems.append(
                f"{_where(s)}: on the sealed run its {INTERVAL_SEED} is {seed}, not the seed "
                f"{the_seed} its {INTERVALS_MODULE} entries use (rule 2, reading L13)"
            )
        elif the_seed is None:  # several seeds, already a fault: name each reader that differs
            for reader, wanted, _, _ in read_by:
                if wanted is not None and wanted != seed:
                    problems.append(
                        f"{_where(reader)}: its {SEED_OPTION} {wanted} is not the {INTERVAL_SEED} "
                        f"{seed} of {_where(s)}, whose run it reads (rule 2, reading L13)"
                    )
    for reader, _, _, label in wants:
        if step_module(by_label[label]) not in SCORING_MODULES:
            problems.append(
                f"{_where(reader)}: reads the run of {_where(by_label[label])}, which is no run "
                f"of {' or '.join(SCORING_MODULES)} (rule 2, reading L13)"
            )
    return problems


TRUTH_MODULE = "truth_build"  # the truth build of the training and development units
MODEL_OPTIONS = ("--model", "--reference")  # the options that name a forecast model


def _openings_faults(steps: Sequence["Step"]) -> list[str]:
    """Rule 6 (reading L17): the sealed run starts with its truth builds, and its entries that
    open sealed data (the sealed truth build) follow one another."""
    from fleetreserve.sealed_route import TRUTH_BUILD_MODULE

    builds = (TRUTH_MODULE, TRUTH_BUILD_MODULE)
    problems = []
    first_other = next((s for s in steps if step_module(s) not in builds), None)
    for s in steps:
        if (
            step_module(s) in builds
            and first_other is not None
            and s.position > first_other.position
        ):
            problems.append(
                f"{_where(s)}: a truth build comes after {_where(first_other)}, which is no "
                "truth build: the sealed run starts with its truth builds (rule 6, reading L17)"
            )
    openers = [k for k, s in enumerate(steps) if step_module(s) == TRUTH_BUILD_MODULE]
    if openers:
        between = [
            s for s in steps[openers[0] : openers[-1] + 1] if step_module(s) != TRUTH_BUILD_MODULE
        ]
        for s in between:
            problems.append(
                f"{_where(s)}: comes between two entries that open sealed data "
                f"({_where(steps[openers[0]])} and {_where(steps[openers[-1]])}): every opening "
                "of sealed data is made in the first step that opens any, the sealed truth "
                "builds one after another (rule 6, reading L17)"
            )
    return problems


def _outcome_reader_faults(steps: Sequence["Step"]) -> list[str]:
    """Rule 7 (reading L18): no step of the sealed run names a model that reads other days'
    outcomes (`forecast_model.OUTCOME_READERS`)."""
    from fleetreserve.forecast_model import OUTCOME_READERS

    problems = []
    for s in steps:
        for name, given in step_options(s, MODEL_OPTIONS).items():
            for option in given:
                if option.value in OUTCOME_READERS:
                    problems.append(
                        f"{_where(s)}: its {name} {option.value} reads other days' outcomes "
                        "(forecast_model.OUTCOME_READERS); on the sealed run a scored unit's days "
                        "reach a model only through its inputs (rule 7, reading L18)"
                    )
    return problems


def rehearsal_faults(run_list: RunList) -> list[str]:
    """Rule 9 (reading L20): no step of the rehearsal gives a held-out option."""
    problems = []
    for s in instantiate(run_list, REHEARSAL):
        for name, given in step_options(s, (TRUTH_RUN, TABLES)).items():
            if given:
                problems.append(
                    f"{_where(s)}: in the rehearsal it gives {name}: the rehearsal runs on "
                    "development units and reads no sealed units' tables (rule 9, reading L20)"
                )
    return problems


O2_MODULE, EVERY_DAY = "ceiling_report", "--every-day"  # outcome O2's command (rule 10)


def _o2_faults(steps: Sequence["Step"]) -> list[str]:
    """Rule 10 (reading L21): each `ceiling_report` step of the sealed run names the nine
    held-out units and gives `--every-day`."""
    problems = []
    for s in steps:
        if step_module(s) != O2_MODULE:
            continue
        from fleetreserve.cold_start_gate import HELD_OUT_UNITS  # rule 3's nine (reading L14)

        units = _values(s, UNITS_OPTION)
        if units is None:
            default = command_default(O2_MODULE, UNITS_OPTION)
            units = [] if default is None else [str(u) for u in default]
        said = []
        development = [u for u in units if unit_kind(u) == DEVELOPMENT]
        if development:
            said.append(f"names the development units {', '.join(development)}")
        missing = [u for u in HELD_OUT_UNITS if u not in units]
        if missing:
            said.append(f"lacks {', '.join(missing)}")
        if said:
            problems.append(
                f"{_where(s)}: in the sealed run it {' and '.join(said)}: outcome O2 is measured "
                f"on the {len(HELD_OUT_UNITS)} held-out units (rule 10, reading L21)"
            )
        if not step_options(s, (EVERY_DAY,))[EVERY_DAY]:
            problems.append(
                f"{_where(s)}: in the sealed run it gives no {EVERY_DAY}: outcome O2 is measured "
                "on every counting day with at most 8 service hours (rule 10, reading L21)"
            )
    return problems


DIAL_ONCE_MODULES = ("compare_run", "corner_run")  # a dial name once per command (rule 11)


def _dial_once_faults(run_list: RunList, steps: Sequence["Step"]) -> list[str]:
    """Rule 11's first part (reading L22): of the sealed run's steps of each of
    `DIAL_ONCE_MODULES`, no two give `--dial` one fixed name. Its second part, which reads the
    matrices, is `dial_alone_faults`."""
    problems = []
    for module in DIAL_ONCE_MODULES:
        first: dict[str, Step] = {}
        for s in steps:
            if step_module(s) != module:
                continue
            name = _fixed_name(run_list.entries[s.position], SEALED_RUN, DIAL_OPTION)
            if name not in DIAL_NAMES:
                continue  # rule 8's fault
            if name in first:
                problems.append(
                    f"{_where(s)}: its {DIAL_OPTION} is the fixed name {name}, which "
                    f"{_where(first[name])} of {module} gives too: each {module} step scores at "
                    "one site type's dial, so no site type's dial is left without one "
                    "(rule 11, reading L22)"
                )
            else:
                first[name] = s
    return problems


RHO_OPTION, O2_TABLES_OPTION = "--rho", TABLES  # outcome O2's rho and sealed tables (rule 12)
O2_TRUTH_RUN = ("--run", "--truth-run")  # the names of `ceiling_report`'s development truth run
SETTING_OPTION = "--setting"  # a truth build's setting
O2_RULE = "(rule 12, reading L23)"


def _o2_build_faults(s: "Step", said: str, by_label: dict, sealed: bool) -> list[str]:
    """Rule 12's faults of one `ceiling_report` step's truth run: the development one (`sealed`
    false) or the sealed one: given once, as the run of an entry that runs that truth build with
    the step's setting (the build's `--setting` as it reads it)."""
    from fleetreserve.sealed_route import TRUTH_BUILD_MODULE

    module = TRUTH_BUILD_MODULE if sealed else TRUTH_MODULE
    names = (TRUTH_RUN,) if sealed else O2_TRUTH_RUN
    found = step_options(s, names)
    given = [(name, o) for name in names for o in found[name]]
    if len(given) > 1 or (given and given[0][1].value is None):
        return []  # `option_form_faults`' fault (reading L12)
    name = given[0][0] if given else names[-1]
    label = _run_of(s, given[0][1]) if given else None
    build = by_label.get(label)
    if build is None or step_module(build) != module:
        shown = "not given" if not given else repr(given[0][1].value)
        if build is not None:
            shown = f"the run of {_where(build)}, which runs {PACKAGE}.{step_module(build)}"
        return [
            f"{_where(s)}: {said} its {name} is {shown}, not the run of a {module} entry: the "
            f"step reads the truth build of its setting, {s.setting}, by its run {O2_RULE}"
        ]
    setting = _one(step_options(build, (SETTING_OPTION,)), SETTING_OPTION)
    if setting is None and step_options(build, (SETTING_OPTION,))[SETTING_OPTION]:
        return []  # `option_form_faults`' fault (reading L12)
    setting = setting.value if setting is not None else command_default(module, SETTING_OPTION)
    if setting == s.setting:
        return []
    return [
        f"{_where(s)}: {said} its {name} is the run of {_where(build)}, a build of the {setting} "
        f"setting, not of its setting, {s.setting}: the step would read another setting's "
        f"truth {O2_RULE}"
    ]


def _o2_step_faults(s: "Step", purpose: str, said: str, by_label: dict) -> list[str]:
    """Rule 12's faults of one `ceiling_report` step of `purpose`: its rho, `--every-day`, its
    development truth run and, in the sealed run, its sealed truth run and tables folder."""
    from fleetreserve.sealed_route import held_out_tables
    from fleetreserve.truth import MAIN

    problems = _o2_build_faults(s, said, by_label, sealed=False)
    if purpose == SEALED_RUN:
        problems += _o2_build_faults(s, said, by_label, sealed=True)
    # The flag is read alone: `step_options` takes the argument after a name as its value.
    found = step_options(s, (RHO_OPTION, O2_TABLES_OPTION)) | step_options(s, (EVERY_DAY,))
    # An option given twice, or with no value, is `option_form_faults`' fault (reading L12).
    if not found[RHO_OPTION]:
        rho = command_default(O2_MODULE, RHO_OPTION)
    else:
        rho = found[RHO_OPTION][0].value if len(found[RHO_OPTION]) == 1 else 0.0
    if rho is not None and _float(rho) != 0.0:
        problems.append(
            f"{_where(s)}: {said} its {RHO_OPTION} is {rho}: outcome O2 runs at rho 0 alone "
            f"{O2_RULE}"
        )
    if not found[EVERY_DAY]:
        problems.append(
            f"{_where(s)}: {said} it gives no {EVERY_DAY}: outcome O2 is measured on every "
            f"counting day with at most 8 service hours {O2_RULE}"
        )
    tables = found[O2_TABLES_OPTION]
    if purpose != SEALED_RUN or len(tables) > 1 or (tables and tables[0].value is None):
        return problems
    wanted = held_out_tables(Path(), s.setting)
    if tables:
        read, shown = Path(tables[0].value), ""
    else:
        read, shown = held_out_tables(Path(), MAIN), " (not given: the command's default)"
    if read != wanted:
        problems.append(
            f"{_where(s)}: {said} its {O2_TABLES_OPTION} is {read}{shown}, not {wanted} of its "
            f"setting, {s.setting}: the step would read another setting's sealed tables {O2_RULE}"
        )
    return problems


def _o2_two_faults(run_list: RunList) -> list[str]:
    """Rule 12 (reading L23): when any purpose runs a `ceiling_report` step, each purpose holds
    exactly two, one per setting, each at rho 0 with `--every-day`, the sealed run's on the
    sealed tables of its own setting. Each option is read at its one place, as the command's
    parser reads it: `sealed_faults` holds every step of both purposes to options written in
    full and once (`option_form_faults`, reading L12)."""
    every = {purpose: instantiate(run_list, purpose) for purpose in PURPOSES}
    by_purpose = {
        purpose: [s for s in steps if step_module(s) == O2_MODULE]
        for purpose, steps in every.items()
    }
    if not any(by_purpose.values()):
        return []
    problems = []
    for purpose, steps in by_purpose.items():
        by_label = {s.label: s for s in every[purpose]}
        said = f"in the {purpose.replace('-', ' ')}"
        first: dict[str, Step] = {}
        for s in steps:
            if s.setting in first:
                problems.append(
                    f"{_where(s)}: {said} it is a second {O2_MODULE} step of the {s.setting} "
                    f"setting, beside {_where(first[s.setting])}: outcome O2 is one step in "
                    f"each setting {O2_RULE}"
                )
            else:
                first[s.setting] = s
            problems += _o2_step_faults(s, purpose, said, by_label)
        for setting in SETTINGS:
            if setting not in first:
                problems.append(
                    f"entries: {said} no {O2_MODULE} step is of the {setting} setting: outcome "
                    f"O2 is one step in each setting {O2_RULE}"
                )
    return problems


def sealed_faults(run_list: RunList) -> list[str]:
    """The faults of the list's sealed-run instantiation that the list alone decides (rules 1,
    2, 3, 6, 7, 10 and 11 of the module text), and of its rehearsal's (rule 9), and of both
    purposes' outcome O2 steps (rule 12), and every step of both purposes with an option
    shortened or given twice (reading L12), one line each; empty when there is none."""
    steps = instantiate(run_list, SEALED_RUN)
    return (
        option_form_faults(steps)
        + option_form_faults(instantiate(run_list, REHEARSAL), said="in the rehearsal")
        + _builds_first_faults(steps)
        + _openings_faults(steps)
        + _interval_faults(steps)
        + _pooled_run_faults(steps)
        + _outcome_reader_faults(steps)
        + _o2_faults(steps)
        + _dial_once_faults(run_list, steps)
        + rehearsal_faults(run_list)
        + _o2_two_faults(run_list)
    )


def committed_matrix(top: Path, text: str) -> tuple[frozenset[str] | None, str | None]:
    """(the gate units of the matrix `text` names, None), or (None, why not): `read_matrix`."""
    matrix, why = read_matrix(top, text)
    return (None, why) if matrix is None else (frozenset(matrix.gate_units), None)


def read_matrix(top: Path, text: str):
    """(the matrix `text` names, None) as the checked-out commit of the working
    copy `top` holds it, read from git's objects and never from a path on disk; or (None, why
    not). The path is first held to the checks a run makes of a path argument before it opens
    one (`gate_run.argument_path`'s refusal of a URL, `gate_run.check_files`), with no `~`
    expanded: a path with `~` names no committed file (reading L14)."""
    found, why = read_matrix_document(top, text)
    return (None, why) if found is None else (found[0], None)


def read_matrix_document(top: Path, text: str):
    """((the matrix, its JSON document), None) or (None, why not), read as `read_matrix` reads
    it: the document holds the fields no run reads, `sealed_choice` among them (rule 8)."""
    from fleetreserve import gate_run, sealed_route  # the route's own checks and reader

    if text.startswith("~"):
        return None, "it is written with ~, so it names no file committed in the working copy"
    try:
        gate_run.argument_path(text)
        path = Path(text)
        full = Path(os.path.normpath(path if path.is_absolute() else Path(top) / path))
        gate_run.check_files([full])
        relative = full.relative_to(Path(os.path.normpath(top)))
    except gate_run.CouldNotRun as problem:
        return None, str(problem)
    except ValueError:
        return None, f"it lies outside the working copy {top}"
    try:
        head = subprocess.run(
            ["git", "-C", str(top), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        data, _ = sealed_route.committed_text(top, head, relative)
        document = json.loads(data)
        matrix = gate_config.parse(document)
    except (OSError, subprocess.CalledProcessError) as problem:
        raise CannotDecide(
            f"the checked-out commit of {top} cannot be read: {problem}"
        ) from problem
    except sealed_route.CouldNotRun as problem:
        return None, f"it is not a file committed at the checked-out commit: {problem}"
    except ValueError as problem:  # not JSON, or no matrix (`gate_config.MatrixError`)
        return None, f"its committed text is no matrix: {problem}"
    return (matrix, document), None


def _values(step: "Step", option: str) -> list[str] | None:
    """The values of an option that takes several (`--units a b`), as the step gives it once, or
    None when it is not given."""
    command = step.command
    for k in range(_module_at(command) + 1, len(command)):
        key, equals, value = command[k].partition("=")
        if key != option:
            continue
        if equals:
            return [value]
        found = []
        for part in command[k + 1 :]:
            if part.startswith("-"):
                break
            found.append(part)
        return found
    return None


def _default_text(value) -> str:
    """A command's default path as the working copy the commands run from names it: a path under
    the package's own top folder (`sessions.TOP`) from that folder (reading L15)."""
    from fleetreserve.sessions import TOP as PACKAGE_TOP

    path = Path(value)
    try:
        return str(path.relative_to(PACKAGE_TOP))
    except ValueError:
        return str(path)


def route_fault(step: "Step", top: Path | None) -> str | None:
    """The refusal the sealed route's deciding function gives at the start of the step's command,
    called on the step's options and on the units of its matrix read from git's objects, or None
    (reading L15). Steps of other modules give None. Raises `CannotDecide`."""
    from fleetreserve import cold_start_gate, sealed_route

    module = step_module(step)
    deciding = ROUTE_COMMANDS.get(module)
    if deciding is None:
        return None
    found = step_options(step, (MATRIX_OPTION, *ROUTE_OPTIONS, *GATE_TRIALS))
    given = {name: None if _one(found, name) is None else _one(found, name).value for name in found}
    where = _where(step)
    reference = None
    if deciding == MATRIX_ROUTE:
        text = given[MATRIX_OPTION]
        if not found[MATRIX_OPTION]:
            default = command_default(module, MATRIX_OPTION)
            if default is None:
                return f"{where}: it names no {MATRIX_OPTION}, which {module} requires"
            text = _default_text(default)
        if text is None:
            return f"{where}: its {MATRIX_OPTION} is given no value or more than once"
        if top is None:
            raise CannotDecide(
                f"{where} names the matrix {text}, which cannot be found: the list is not in a "
                "git working copy, the folder its commands run from"
            )
        matrix, why = read_matrix(top, text)
        if matrix is None:
            return f"{where}: its matrix {text} cannot be read, so its route is not known: {why}"
        units, reference = tuple(matrix.gate_units), matrix.reference
        named = f"the units of its matrix {text}"
    else:
        units = _values(step, UNITS_OPTION)
        if units is None:
            default = command_default(module, UNITS_OPTION)
            units = [] if default is None else [str(u) for u in default]
        named = f"the units it names ({' '.join(units)})"
    arguments = [given[name] for name in ROUTE_OPTIONS]
    try:
        if deciding == CHECK_UNITS:
            held = cold_start_gate.check_units(units)[1] == cold_start_gate.HELD_OUT
        else:
            try:
                held = sealed_route.held_out_kind(units)
            except sealed_route.UnitsRefused:
                held = False  # a mix: the deciding function refuses it before it touches anything
            if not held:  # the branches of the function that touch nothing (reading L15)
                getattr(sealed_route, deciding)(top, units, *arguments)
    except sealed_route.UnitsRefused as problem:
        return (
            f"{where}: on {named}, refused at its start: the units are refused: {problem}; "
            "nothing read"
        )
    except sealed_route.RouteRefused as problem:
        return f"{where}: on {named}, refused at its start by the sealed route: {problem}"
    if not held:
        return None
    # Held-out units: the function's next steps look at the folders and pass the gate, so it is
    # not called; the refusal that follows them is the one rule copied (reading L15).
    if given[TRUTH_RUN] is None:
        return f"{where}: on {named}, refused at its start: held-out units need {TRUTH_RUN}"
    if module == GATE_MODULE:
        trials = {name: given[name] for name in GATE_TRIALS}
        forecaster = f"the empirical forecaster of gate_run (reference {reference})"
        refusal = sealed_route.trial_refusal(trials) or sealed_route.OUTCOME_READER.format(
            what=forecaster
        )
        return f"{where}: on {named}, refused at its start on the sealed route: {refusal}"
    return None


def route_faults(steps: Sequence["Step"], top: Path | None) -> list[str]:
    """`route_fault` of every step of the sealed run, one line each (reading L15)."""
    found = (route_fault(s, top) for s in steps)
    return [f"{line} (reading L15)" for line in found if line is not None]


DIAL_OPTION, KAPPA_OPTION = "--dial", "--kappa"
# The fixed names of the dials, each with the site type of `sealed_choice.dials` whose dial it
# holds (`sealed_matrix.WORKPLACES`, `sealed_matrix.OTHERS`; rule 8, reading L19).
DIAL_NAMES = {"dial-workplaces": "workplaces", "dial-others": "Norway and Korea"}


def _fixed_name(entry: Entry, purpose: str, option: str) -> str | None:
    """The fixed name `<name>` of `{name:<name>}` written right after `option` in the entry's
    command or its extra arguments for `purpose`, or None when the value is written otherwise."""
    parts = (*entry.command, *entry.extra.get(purpose, ()))
    for k in range(_module_at(parts) + 1, len(parts) - 1):
        if parts[k] == option:
            match = _REFERENCE.fullmatch(parts[k + 1])
            return match.group(2) if match and match.group(1) == "name" else None
    return None


def _float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _same(text: str, number) -> bool:
    """`text` and `number` are one number (a dial or kappa as `argparse` reads it, `type=float`)."""
    return _float(text) is not None and _float(text) == _float(number)


def dial_faults(run_list: RunList, steps: Sequence["Step"], top: Path | None) -> list[str]:
    """Rule 8 (reading L19): each `--dial` of a sealed-run step is the fixed name of a site
    type's dial whose value is the dial `sealed_choice.dials` gives that site type in the matrix
    the step reads, every unit it names is of that site type, and each `--kappa` is that
    matrix's `sealed_choice.kappa`. One line per fault. Raises `CannotDecide`."""
    from fleetreserve.sealed_matrix import site_type_of

    problems = []
    for s in steps:
        found = step_options(s, (DIAL_OPTION, KAPPA_OPTION, MATRIX_OPTION))
        dial, kappa = _one(found, DIAL_OPTION), _one(found, KAPPA_OPTION)
        if (dial is None or dial.value is None) and (kappa is None or kappa.value is None):
            continue  # no dial and no kappa, or one given twice (reading L12)
        where, module = _where(s), step_module(s)
        text = None if _one(found, MATRIX_OPTION) is None else _one(found, MATRIX_OPTION).value
        if not found[MATRIX_OPTION]:
            default = command_default(module, MATRIX_OPTION)
            text = None if default is None else _default_text(default)
        if text is None:
            problems.append(
                f"{where}: it gives a dial or a kappa but names no {MATRIX_OPTION} whose "
                "sealed_choice it can be held against (rule 8, reading L19)"
            )
            continue
        if top is None:
            raise CannotDecide(
                f"{where} names the matrix {text}, which cannot be found: the list is not in a "
                "git working copy, the folder its commands run from"
            )
        read, _ = read_matrix_document(top, text)
        if read is None:
            continue  # its route decision reports the matrix that cannot be read
        choice = read[1].get("sealed_choice")
        if not isinstance(choice, dict) or not isinstance(choice.get("dials"), dict):
            problems.append(
                f"{where}: its matrix {text} has no sealed_choice with the dials per site type, "
                "so its dial or kappa cannot be held against the matrix's own (rule 8, reading L19)"
            )
            continue
        if (
            kappa is not None
            and kappa.value is not None
            and not _same(kappa.value, choice.get("kappa"))
        ):
            problems.append(
                f"{where}: its {KAPPA_OPTION} {kappa.value} is not the kappa "
                f"{choice.get('kappa')} of sealed_choice in its matrix {text} (rule 8, reading L19)"
            )
        if dial is None or dial.value is None:
            continue
        name = _fixed_name(run_list.entries[s.position], SEALED_RUN, DIAL_OPTION)
        site_type = DIAL_NAMES.get(name)
        if site_type is None:
            written = f"{{name:{name}}}" if name else "a literal value"
            problems.append(
                f"{where}: its {DIAL_OPTION} {dial.value} is {written}, not one of the fixed "
                f"names {', '.join(DIAL_NAMES)}, so the site type whose dial it must be cannot "
                "be read (rule 8, reading L19)"
            )
            continue
        wanted = choice["dials"].get(site_type)
        if not _same(dial.value, wanted):
            problems.append(
                f"{where}: its {DIAL_OPTION} {dial.value} (the fixed name {name}) is not the "
                f"dial {wanted} that sealed_choice.dials gives {site_type} in its matrix {text}: "
                f"the step would score at {dial.value}, not at its matrix's {wanted} (rule 8, "
                "reading L19)"
            )
        units = _values(s, UNITS_OPTION) or []
        other = [u for u in units if site_type_of(u) != site_type]
        if other:
            problems.append(
                f"{where}: it scores {', '.join(other)}, of another site type than {site_type}, "
                f"whose dial its {DIAL_OPTION} {name} holds (rule 8, reading L19)"
            )
    return problems


def dial_alone_faults(run_list: RunList, steps: Sequence[Step], top: Path | None) -> list[str]:
    """Rule 11's second part (reading L22): a `compare_run` or `corner_run` step of the sealed
    run that is the only one of its command for the site type of its fixed dial name stands
    alone only when `sealed_choice.dials` in its matrix gives every site type the dial its
    `--dial` has; else one line naming the step and both dials. Steps whose matrix cannot be
    read, or has no `sealed_choice`, are left to rules 5 and 8. Raises `CannotDecide`."""
    problems = []
    for module in DIAL_ONCE_MODULES:
        mine = [s for s in steps if step_module(s) == module]
        names = {
            s.position: _fixed_name(run_list.entries[s.position], SEALED_RUN, DIAL_OPTION)
            for s in mine
        }
        covered = {DIAL_NAMES[n] for n in names.values() if n in DIAL_NAMES}
        missing = [t for t in DIAL_NAMES.values() if t not in covered]
        if not missing:
            continue  # one step per site type, as rule 8 holds each to its own dial
        for s in mine:
            found = step_options(s, (DIAL_OPTION, MATRIX_OPTION))
            dial, matrix = _one(found, DIAL_OPTION), _one(found, MATRIX_OPTION)
            if names[s.position] not in DIAL_NAMES or dial is None or dial.value is None:
                continue  # rule 8's fault
            text = None if matrix is None else matrix.value
            if not found[MATRIX_OPTION]:
                default = command_default(module, MATRIX_OPTION)
                text = None if default is None else _default_text(default)
            if text is None:
                continue  # rule 8's fault
            if top is None:
                raise CannotDecide(
                    f"{_where(s)} names the matrix {text}, which cannot be found: the list is "
                    "not in a git working copy, the folder its commands run from"
                )
            read, _ = read_matrix_document(top, text)
            choice = None if read is None else read[1].get("sealed_choice")
            if not isinstance(choice, dict) or not isinstance(choice.get("dials"), dict):
                continue  # rules 5 and 8
            differ = [t for t in missing if not _same(dial.value, choice["dials"].get(t))]
            if differ:
                own = DIAL_NAMES[names[s.position]]
                gives = " and ".join(f"{t} {choice['dials'].get(t)}" for t in (own, *differ))
                problems.append(
                    f"{_where(s)}: it is the only {module} step, at the dial {dial.value} of "
                    f"{own}, but sealed_choice.dials in its matrix {text} gives {gives}: a "
                    f"{module} step scores every unit of its matrix at the one dial given, so it "
                    "may stand alone only when every site type has that dial, and the list must "
                    f"hold one {module} step per site type (rule 11, reading L22)"
                )
    return problems


def unit_faults(run_list: RunList, top: Path | None) -> list[str]:
    """The faults read from the committed matrices: the sealed route's decision at the start of
    each sealed step's command (`route_faults`, reading L15); rule 8, each dial and kappa its
    matrix's (`dial_faults`, reading L19); rule 11's second part, a `compare_run` or `corner_run`
    step that stands alone only where the site types' dials are equal (`dial_alone_faults`,
    reading L22); and rule 3 (reading L14), each
    `learn_run` of the sealed route whose run a sealed `day_intervals` entry reads names a
    matrix whose gate units are the nine held-out units. `top` is the working copy the commands
    run from (None: not known, which is undecidable when a matrix must be read). Raises
    `CannotDecide`."""
    steps = instantiate(run_list, SEALED_RUN)
    problems = (
        route_faults(steps, top)
        + dial_faults(run_list, steps, top)
        + dial_alone_faults(run_list, steps, top)
    )
    by_label = {s.label: s for s in steps}
    read: dict[str, list[Step]] = {}
    for s in steps:
        if step_module(s) == INTERVALS_MODULE:
            label = _run_of(s, _one(step_options(s, (RUN_OPTION,)), RUN_OPTION))
            if label in by_label:
                read.setdefault(label, []).append(s)
    nine = None
    for s in steps:
        found = step_options(s, (TRUTH_RUN, MATRIX_OPTION))
        readers = read.get(s.label, [])
        if step_module(s) != LEARN_MODULE or not readers or not found[TRUTH_RUN]:
            continue  # its run feeds no interval, or it scores no sealed unit
        option = _one(found, MATRIX_OPTION)
        if option is None or option.value is None or _run_of(s, option) is not None:
            problems.append(
                f"{_where(s)}: on the sealed run its {INTERVALS_MODULE} reads its run, but it "
                f"names no {MATRIX_OPTION} file whose units can be read (rule 3, reading L14)"
            )
            continue
        if nine is None:
            # the one home of the held-out units, through the cold start's names (reading L14)
            from fleetreserve.cold_start_gate import HELD_OUT_UNITS

            nine = frozenset(HELD_OUT_UNITS)
        if top is None:
            raise CannotDecide(
                f"{_where(s)} names the matrix {option.value}, which cannot be found: the list is "
                "not in a git working copy, the folder its commands run from"
            )
        units, _ = committed_matrix(top, option.value)
        if units is None:
            continue  # its route decision reports the matrix that cannot be read
        if units != nine:
            missing = sorted(nine - units)
            other = sorted(units - nine)
            said = [f"lacks {', '.join(missing)}"] if missing else []
            said += [f"also names {', '.join(other)}"] if other else []
            problems.append(
                f"{_where(s)}: its matrix {option.value} scores {len(units & nine)} of the "
                f"{len(nine)} held-out units ({'; '.join(said)}), and "
                f"{', '.join(_where(r) for r in readers)} pools the units of this one run alone: "
                f"outcome O1 is over the {len(nine)} held-out units, so they are scored in one "
                f"{LEARN_MODULE} (rule 3, reading L14)"
            )
    return problems


def git_top(folder: Path) -> Path | None:
    """The top folder of the git working copy holding `folder`, or None."""
    try:
        done = subprocess.run(
            ["git", "-C", str(folder), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return Path(done.stdout.strip()) if done.returncode == 0 and done.stdout.strip() else None


def _no_constant(name: str):
    raise ValueError(f"the number {name} is not allowed")


def decode(data: bytes):
    """The JSON document of a run list's bytes; raises `Faulty` (reading L3)."""
    try:
        return json.loads(data.decode("utf-8"), parse_constant=_no_constant)
    except ValueError as problem:
        raise Faulty([f"the list: its text is not JSON: {problem}"]) from problem


def load(path: Path) -> RunList:
    """The run list in the file `path`, its labels held against the limit on a name in the list's
    own folder (reading L6); raises `CannotDecide` or `Faulty`."""
    try:
        data = Path(path).read_bytes()
    except OSError as problem:
        raise CannotDecide(f"the run list {path} cannot be read: {problem}") from problem
    return parse(decode(data), name_max=name_limit(Path(path).parent))


def instantiate(run_list: RunList, purpose: str) -> tuple[Step, ...]:
    """The entries that run in `purpose`, in the list's order, each with its command and the
    purpose's extra arguments filled in: `{units}` by the purpose's group, a fixed value by its
    value; each `{run:<label>}` is kept, and its place noted, for `fill_runs`."""
    if purpose not in PURPOSES:
        raise ValueError(f"{purpose!r} is not one of {', '.join(PURPOSES)}")
    steps = []
    for k, entry in enumerate(run_list.entries):
        if not entry.runs_in(purpose):
            continue
        command: list[str] = []
        runs: list[tuple[int, str]] = []
        for part in (*entry.command, *entry.extra.get(purpose, ())):
            match = _REFERENCE.fullmatch(part)
            if part == UNITS:
                command.extend(run_list.units[purpose][entry.units])
                continue
            if match and match.group(1) == RUN:
                runs.append((len(command), match.group(2)))
            elif match:
                part = str(run_list.fixed[_FIXED_OF[match.group(1)]][match.group(2)])
            command.append(part)
        steps.append(
            Step(
                position=k,
                label=entry.label,
                command=tuple(command),
                runs=tuple(runs),
                setting=entry.setting,
                max_seconds=entry.max_seconds,
                after=entry.after,
            )
        )
    return tuple(steps)


def fill_runs(step: Step, runs: dict[str, str]) -> list[str]:
    """The step's command with each `{run:<label>}` replaced by `runs[<label>]`, the run folder
    that entry recorded in this drive (reading L10); raises `KeyError` for a label `runs` lacks."""
    command = list(step.command)
    for place, label in step.runs:
        command[place] = runs[label]
    return command


def differences(run_list: RunList) -> list[str]:
    """Where the two purposes' instantiations differ, one line each (reading L8)."""
    found = []
    for k, entry in enumerate(run_list.entries):
        where = f"entry {k + 1} ({entry.label})"
        if entry.only is not None:
            found.append(f"{where}: runs in the {entry.only} only")
            continue
        if entry.units is not None:
            ours, theirs = (run_list.units[purpose][entry.units] for purpose in PURPOSES)
            if ours != theirs:
                found.append(
                    f"{where}: its units, the group {entry.units!r}: {REHEARSAL} {' '.join(ours)}; "
                    f"{SEALED_RUN} {' '.join(theirs)}"
                )
        for purpose in PURPOSES:
            if purpose in entry.extra:
                found.append(
                    f"{where}: arguments for the {purpose} only: {' '.join(entry.extra[purpose])}"
                )
    return found


def entries_text(count: int) -> str:
    """`1 entry`, `2 entries`."""
    return f"{count} entr{'y' if count == 1 else 'ies'}"


def differences_text(count: int) -> str:
    """`1 difference`, `2 differences`."""
    return f"{count} difference{'' if count == 1 else 's'}"


def summary(run_list: RunList) -> str:
    parts = []
    for purpose in PURPOSES:
        steps = instantiate(run_list, purpose)
        total = sum(s.max_seconds for s in steps)
        parts.append(
            f"the {purpose} {entries_text(len(steps))}, time limits adding up to {total:g} s "
            f"({total / 3600:.1f} h)"
        )
    return (
        f"{'; '.join(parts)}; {differences_text(len(differences(run_list)))} between the purposes"
    )


STARTS_NOT_MADE = "the starts not made (--no-starts)"


def route_steps(run_list: RunList) -> list[Step]:
    """The steps of the sealed run that run a command of the sealed route (reading L16)."""
    return [s for s in instantiate(run_list, SEALED_RUN) if step_module(s) in ROUTE_COMMANDS]


def starts_text(count: int) -> str:
    """What the last line says of the start check made (reading L16)."""
    return (
        f"the starts of {count} step{'' if count == 1 else 's'} of the sealed route made at a "
        "made-up tag, none refused"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fleetreserve.run_list",
        description="Check the form of a run list (exit 0 well formed, 1 faulty, 2 could not run).",
    )
    parser.add_argument("run_list", type=Path, help="the run list, a JSON file")
    parser.add_argument(
        "--no-starts",
        action="store_true",
        help="leave out the start check (made-up lists whose commands are stand-ins)",
    )
    options = parser.parse_args(argv)
    try:
        run_list = load(options.run_list)
        top = git_top(options.run_list.resolve().parent)
        units = unit_faults(run_list, top)
        if units:
            raise Faulty(units)
        starts = STARTS_NOT_MADE
        if not options.no_starts:
            from fleetreserve import run_starts  # the start check (reading L16)

            started = run_starts.start_faults(run_list, top)
            if started:
                raise Faulty(started)
            starts = starts_text(len(route_steps(run_list)))
        found = differences(run_list)
        last = f"well formed: {options.run_list}: {summary(run_list)}"
    except Faulty as problem:
        for line in problem.faults:
            print(f"FAULT: {line}")
        print(f"faulty: {problem} in {options.run_list}")
        return FAULTY
    except CannotDecide as problem:
        print(f"COULD NOT RUN: {problem}")
        return COULD_NOT_RUN
    except Exception as problem:  # an unexpected error: could not run, never Python's own 1
        print(f"COULD NOT RUN: the check stopped on {type(problem).__name__}: {problem}")
        return COULD_NOT_RUN
    for line in found:
        print(f"difference: {line}")
    print(f"starts: {starts}")
    print(last)
    return WELL_FORMED


if __name__ == "__main__":
    sys.exit(main())
