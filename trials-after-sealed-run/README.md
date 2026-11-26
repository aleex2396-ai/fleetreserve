# Trials made after the held-out results were seen

Everything in this folder was written on 24 November 2026 or 25 November 2026,
after held-out results were seen. It fills no pre-registered slot, changes no held-out number and
no plan definition or pass mark, and the run matrix of the run from the tag
(`gate/sealed-matrix/`) is not touched. The trials run on development units only, each once, and
are reported apart from the plan's results.

Every output of the commands below says so: the key `made_after_sealed_run` and a label in each
results file, the first two lines of each summary, and the file `made-after-sealed-run.txt` in the
run folder (`fleetreserve.after_sealed_run`). The flag `--made-after-sealed-run` turns it on in
`learn_run` and `dial_choice`; `pooled_calibration` always carries it. Only `learn_run` has a route
to held-out units, and it refuses the flag there. `dial_choice` has no such route: a held-out unit
named in its matrix has no development table under the default `--tables`, so the run could
not run there (a `--tables` folder holding such a table is not guarded). `pooled_calibration`
refuses any name that is not a development unit before reading anything.
This folder is outside `gate/` on purpose: tests read every JSON file under `gate/` as a
committed run matrix of the plan.

Each command starts from a clean commit, as every recorded run does. `<label>` is the name of the
run folder. Run one heavy run at a time.

## Trial a: forecast energy at level 0.2 with dial 0.02, and trial c: no hourly re-solving at level 0.1 with dial 0.02

One run, one matrix file (`lgbm_tail-trials-a-c.json`, six configurations: the three below and their
oracle-departure partners):

```
python -m fleetreserve.run_record --name zeros-trials-ac --long --max-seconds 172800 -- uv run --frozen --group learn python -m fleetreserve.learn_run --model lgbm_tail --matrix trials-after-sealed-run/lgbm_tail-trials-a-c.json --made-after-sealed-run
```

Output: `learn.json` in the run folder (and `learn-gate.json`, `learn-gate-days.csv`,
`learn-forecasts.csv`, `learn-timing.json`, `learn-day-losses.json`, `made-after-sealed-run.txt`).

- Trial a is the configuration `forecast-energy-k0.2-d0.02-empirical`: energy bounds from the
  model's own per-car quantiles (`energy_bounds: empirical`), level 0.2, dial 0.02. Its comparison
  is `main-k0.2-d0.02-empirical`: energy bounds known, the same level, dial and departure, so the
  two differ in the energy bounds alone. Read, per unit and for both, the `one-way` rung's
  `beta_hat` (calibrated scale) and `test.mean_offer_kwh_per_day`, and for trial a the share of
  days in E' (`e_prime`).
- Trial c is the configuration `main-k0.1-d0.02-empirical`: energy bounds known, level 0.1, dial
  0.02. Read, per unit, its `fixed` rung (the fixed plan, no hourly re-solving): `beta_hat` and
  `test.mean_offer_kwh_per_day`, beside the `one-way` rung of the same configuration. The
  comparison is the fixed plan at level 0.1 against that `one-way` rung; the level-0.2 pair above
  is not part of trial c.
- The same fields are printed in the summary, one line per unit and configuration.

## Trial b: no driver-risk dial at levels 0.05 and 0.1

Two runs of the committed gate matrix; the row "no rows" is the trial (the other rows are the dials,
measured in the same run):

```
python -m fleetreserve.run_record --name zeros-trials-b005 --long --max-seconds 172800 -- uv run --frozen --group learn python -m fleetreserve.dial_choice --model lgbm_tail --kappa 0.05 --splits 200 --made-after-sealed-run
python -m fleetreserve.run_record --name zeros-trials-b010 --long --max-seconds 172800 -- uv run --frozen --group learn python -m fleetreserve.dial_choice --model lgbm_tail --kappa 0.1 --splits 200 --made-after-sealed-run
```

Output: `dial-choice.json` and `dial-choice-summary.txt` of each run folder (also
`dial-choice-split-offers.json`, `dial-choice-forecasts.csv`, `dial-choice-timing.json`,
`dial-choice-certificate-timeouts.json`, `made-after-sealed-run.txt`). Read, per unit and tier, the
configuration `main-rounded-hour-k<level>-empirical` (no driver-risk rows): `mean_offer_kwh_per_test_day`
and `share_of_splits_that_offer`.

## Trial d: pooled calibration across the five Norwegian development sites

```
python -m fleetreserve.run_record --name zeros-trials-d --long --max-seconds 172800 -- uv run --frozen --group learn python -m fleetreserve.pooled_calibration --model lgbm_tail --splits 200
```

Units pooled: `OSL_1`, `BER`, `KRO`, `OSL_2`, `BAR_2` (the default; `--units` names others). A name
that is not a development unit is refused before anything is read. Level 0.5, dial 0.02.

Output: `pooled-calibration.json` and `pooled-calibration-summary.txt` (also
`pooled-calibration-forecasts.csv`, `pooled-calibration-timing.json`,
`made-after-sealed-run.txt`). Read, per unit and tier, `own_scale` (the unit alone) beside
`pooled_scale` (one scale from the pooled offering calibration days), and per tier the `pool`
block: how many of the 200 splits have a scale above 0, the scales chosen, and the mean test losses
over the offering test days of all units together.

## Trial forecast-mix: the tree model and the neural model averaged

The main model `lgbm_tail` (a tree model) is compared on the development gate units with the
average of its departure quantiles and those of the neural model `hazard_tuned_inputs`. The trial
is the module `fleetreserve.models.mix_tail_inputs`; the command `fleetreserve.forecast_mix`
measures both models on the same days, the same level, the same dial family, the same calibration
rule and the same seeded splits, and pairs the two runs split by split. Nothing is adopted: the
main model, the plan and every held-out figure stay as they are.

Command (one run, under the run recorder, from a clean commit, nothing else running). The trial
is this command run with no option. The command has options, which each change a setting and are
not part of the trial: `--kappa` (the level), `--extra-level` (the level of the extra line), `--splits` (the number of splits), `--main-model` and
`--trial-model` (the two models compared), `--settings` (the settings file read), `--matrix` (the
run matrix: units, dial family, calibration rule), `--tables` and `--truth-run` (the tables read);
`--workers` changes the number of processes only. The output says what was run (`run`: level,
splits, models, and the members of the trial model as run) and whether every value equals the
committed settings file's (`settings_check`; the summary says so, and names each difference).

```
python -m fleetreserve.run_record --name forecast-mix --long --max-seconds 172800 -- uv run --frozen --group learn python -m fleetreserve.forecast_mix
```

Output files in the run folder: `forecast-mix.json` (settings, inputs, both models' dial choices
and the paired comparison; read `comparison`), `forecast-mix-summary.txt`, `forecast-mix-timing.json`,
`made-after-sealed-run.txt`, and per run (`main`, `trial` and `extra`, the trial model at the extra level) `forecast-mix-<run>-dial-choice.json`
(the full results of `dial_choice.run`), `forecast-mix-<run>-split-offers.json` and
`forecast-mix-<run>-forecasts.csv`. Every file carries the after-sealed-run label (the key
`made_after_sealed_run` and the label in each results file, the first two lines of the summary,
the label file, which covers the tables).

What is fixed, all in `forecast-mix.json` in this folder with its source, before any result:

- level 0.5, 200 seeded splits (seeds 1 to 200), tiers A and B apart, the dial family of
  `gate/matrix.json` at the level, the one-way re-solve rung, energy bounds known;
- the members `lgbm_tail` and `hazard_tuned_inputs` (the neural variant of plan section 7), the
  neural member at its tuned settings file and its one seed, 0; both fitted on the training units;
- equal weights, one half each, at every level; no weight is a setting, so none is searched or
  tuned (`forecast_average.average_tables` takes no weight); a cell that is NaN in either member is
  NaN in the mix;
- the headline: tier A, per unit, each model at the dial with its largest mean tier A offer
  (`own_dial`), the mean paired difference `trial - main` in kWh per counting test day and its
  standard error over the splits. `same_dial` (both at the main model's dial) and every dial are
  reported beside it; tier B is given as numbers only, as information;
- the reading rule (`forecast_mix_reading`, the `reading` block of the settings file), printed in
  the summary and in `definitions`: the share is the paired mean difference of the offer (mix minus
  main) over the 200 splits as a share of the main arm's mean offer. A unit's bar is the spread the
  neural model's seed alone gives there: 8% at the workplace unit (`nrel-from-may-2019`) and 30% at
  the pooled Norwegian unit (`docs/prior-looks.md`, the entry beginning "The neural model's seed
  test"). The bars are the neural model's range over three seeds (with the plan's own runs) at one
  level, kappa 0.3, applied here to a mean over 200 splits that share days, at kappa 0.5 and 0.4;
  they are an analogy and not this trial's measured noise, and the rule was fixed before any
  result. A share above its bar "shows a gain",
  below minus the bar "shows a loss", else "no measurable difference". One line: "the mix helps"
  only if one unit shows a gain and none a loss; "the mix hurts" if one shows a loss and none a
  gain; otherwise "no measurable difference; a mixed result is reported as mixed". No Dutch or
  Korean site is a development unit, so the trial cannot speak for them;
- one extra line, always reported beside the first and labelled "extra line: the mix at level
  0.4": the mix at level 0.4 against the main model at its own level 0.5, same units, splits,
  seeds, calibration rule and dial rule, judged by the same rule and never part of the one-line
  reading. The neural model did best at 0.4 at workplaces on development data
  (`docs/prior-looks.md`, the entry beginning "The neural model at kappa 0.4 and 0.5, seeds 1 to 3"), so 0.5 alone may understate the mix. The command fits the trial
  model a second time for it (the fit repeats) and measures the dial family at 0.4 again;
- the splits share days, so the standard error over them understates the uncertainty of the whole
  experiment, and the mix uses one training (seed 0) of the neural member, so part of any
  difference is that one training's chance (both sentences are in the summary and in
  `definitions`); a model's own dial is picked on the same 200 splits that score it, for both
  models alike, as `dial_choice` reports it.

A command that gives `--kappa`, `--splits`, `--main-model`, `--trial-model`, `--extra-level`,
`--settings` or `--matrix` with a value other than the committed settings' ends with exit 2,
unless `--not-the-trial` is also given; the label of every output and the first line of the
summary then say that the run is not the trial of the committed settings (the file
`not-the-trial.txt` too, and `made-after-sealed-run.txt` carries the same words as its third line). `settings_check` compares with the committed file, whatever `--settings`
names, and names `--tables` and `--truth-run` when they differ from the defaults.

The command refuses a matrix that holds a unit that is not a development unit of the truth build
before it opens a truth build, a table or a model file (`forecast_mix.check_units`, exit 2); it
names no loader of the sealed data.

Why no test day's outcome reaches its own forecast or calibration:

- Both members are fitted on the training units only: `mix_tail_inputs.MixModel` refuses other fit
  units (the `TRAIN_UNITS` check in `__init__`), `learn_run.check_model` refuses a model that
  names a unit that is not a training unit (`outside`, line 287) or that is scored
  (`FitOnScored`, line 296), and `forecast_mix.measure` hands each model only the tables of its
  `fit_units`.
- A model forecasts from `learn_run.forecast_frame`, which drops every label column
  (`learn_features.feature_frame`, line 329 of `learn_run`): only values known at the decision
  time reach `predict`; the mix adds no input and no outcome.
- A split's scale is calibrated on that split's calibration days alone and its test days are
  scored at that scale (`dial_splits.score_split`); a seed's test days are not in its calibration
  days (`dial_choice.split_indices`).

What has to exist before it runs: a recorded, verified truth build of the training and development
tables, as `dial_choice` needs it; the `learn` dependency group; the context files the neural
member reads (weather and OxCGRT under `data/raw/`, fetched by `tools/fetch_data.py --context`;
the school terms, the tuned settings and the 9/80 parity are committed). No stored fit exists, so
the command fits both members inside the run and no other run has to come first. The run costs
about two `dial_choice` runs plus one fit of each member (a guess, not measured).

Not tested by the author of this trial: the author ran nothing here, including the tests under
`tests/forecast_trials_mix/`, the linter and the formatter.

## Forecast trial local: the tree model refitted with a unit's own first days

Question: does the main tree model (`lgbm_tail`) forecast later days of a unit better when it is
refitted once with that unit's own first days added to its training rows, either with each such row
counting as much as a training row or with the first days counting as much in total as all the
training rows? The cold start of plan
section 11 (`fleetreserve.cold_start`) leaves this out (its reading C2). The trial is paired: each
refitted model and the same model not refitted are scored on the same later days, at the same level
and dial, with the same calibration rule and the same seeded splits, tier A and tier B in separate
blocks. It runs on development units only; a name that is not a development unit is refused before
any data file is read, in `--units` and in the settings file (the settings file is read first, to
check the units it names).

Every setting is in `forecast-local.json`, written before any result. The trial is the command
below run with no option. Options exist, and each changes what is read: `--settings` another
settings file (every value in it), `--matrix` another run matrix (the configurations, the
calibration rule and the gate's readings), `--units` fewer units than the file names, `--tables`
and `--truth-run` the tables and the truth build read, `--workers` the number of processes. No
option sets a single value of the file. The results record the settings file's name and SHA-256
and the matrix's, so a run with an option shows it. Source of each setting:

- Model `lgbm_tail`: the main forecasting model of plan section 7.
- Level 0.5 and dial 0.02: the level and dial that the rules of plan sections 7 and 6 gave the
  main model at both site types in the development runs (`docs/prior-looks.md`, the six-level
  runs: workplaces 17.70, Norway 2.48 kWh per test day). The refitted model gets no level or dial
  of its own.
- k = 56 counting days in time order (date, then site, so a pooled unit counts site-days): the
  largest k of plan section 11, outcome O9 ("k=0, 7, 14, 28, 56 local days in time order"), taken
  from the plan's text and not from any result. These are the days the cold start already sets
  aside as local for every k (its reading C6), so the days after them are the days that
  outcome scores; and the largest k gives a refit the most local rows to learn from.
- Two refit arms (`refits`), both always run and both always reported, whatever they show; the
  settings file lists exactly these two and the command refuses a file that drops, adds or
  reorders one, and never gives the better one alone:
  - `unweighted-pool`: the model's own `fit`, once per unit, on the training units' rows followed
    by the unit's local fit rows, each row once with weight 1.
  - `equal-mass`: the same rows in one fit, each training-pool row with weight 1 and each local
    row with weight (number of training-pool rows) / (number of local fit rows)
    (`forecast_local.equal_mass_weight`), so the local rows weigh as much in total as the pool.
    The weight is the row weight of the LightGBM data set of every booster the model fits (each
    quantity and level; `models.lgbm_tail_equal_mass`, `WeightedQuantileSet.fit`); features,
    labels and settings are those of `lgbm_tail`, which is not changed. The equal total holds over
    the rows of the fit frame, not for each quantity's subset of rows (a quantity is fitted on the
    rows that have its label).
    A check of this fit (`tests/learn/test_lgbm_tail_equal_mass.py`) fits `lgbm_tail`'s own
    unweighted booster set and the weighted one with every weight 1 on the same made-up rows and
    reports the largest absolute difference of their predictions at every level, passing only
    below 1e-9. It is not assumed that they agree: the library may treat a weighted quantile fit
    differently (a weighted-percentile leaf rule) even when every weight is 1. If that test fails,
    the equal-mass arm differs from the unweighted arm by the fitting route as well as by the
    weights, and the difference must be read before any result of the trial is read. Neither arm
    is changed to force equality. The consequence is fixed now: if that test fails, the
    `equal-mass` arm is reported under the name "equal-mass (weights and fitting route)", its
    difference from `unweighted-pool` is not credited to the weights, and no further arm is added
    after results are seen.
- Local fit rows: the rows of the unit's cars table at or before the k-th counting day (also those
  of days that do not count among them) whose session had plugged out before the decision time of
  the earliest later counting day (`forecast_local.local_fit_rows`). The decision time is taken
  over the whole unit, so every outcome a fit row holds is known before the first forecast day
  begins, and no day at or after a forecast day enters the fit. Rows of the first k site-days can
  be left out at any unit: a session still plugged in at that decision time, or with no readable
  plug-out, is left out. At a pooled unit two sites can also have the same date, so the k-th
  site-day may share its date with the first forecast site-day and its sessions are still plugged
  in then. The results and the summary give, per unit and per refit arm, the local site-days asked
  for (k), the site-days with at least one kept row, the rows kept and the rows left out by this
  cutoff. Rows kept and left out count all local rows, those of days that do not count included;
  the site-day count covers the first k counting site-days only.
- Days scored: the counting days after the first k, split into calibration and test days by the
  matrix's calibration-days rule with seeds 1 to 200 (`splits`, plan section 6), on those days
  alone. The first k days are in no calibration set and no test set of either model
  (`forecast_local.tail_splits`; their losses are blanked, `forecast_local.without_first`). A test
  day's outcome reaches neither its forecast (the fit rows are the first k days' rows) nor its
  calibration (only calibration days' losses are read, `dial_splits.score_split`).
- Units: the three gate units of `gate/matrix.json`. `nrel-before-may-2019` is left out and
  listed, by the rule of the code and not from any result: the days scored are the counting days
  after the first k, so a unit needs more than k = 56 days, and the unit has 9 counting days
  (`docs/prior-looks.md`, the day counts of the recorded runs). Its days after the first 56 are
  none (`tail_of`), `gate_run.plan_unit` finds no split of none (`day_split.is_split` needs a
  calibration count of at least 1 and at most one below the number of days), and the unit is
  "not calibratable". Any unit with 56 days or fewer is left out in the same way; `nrel-from-may-2019` has 278 days left (100 calibration, 178
  test) and `norway-development-pooled` 186 (93 and 93, the half rule). The run records its own
  counts.

Command (from a clean commit; needs a recorded truth build at that commit, and the `learn`
dependency group):

```
python -m fleetreserve.run_record --name forecast-local --long --max-seconds 172800 -- uv run --frozen --group learn python -m fleetreserve.forecast_local
```

Output, in the run folder: `forecast-local.json`, `forecast-local-summary.txt`,
`forecast-local-forecasts-main.csv`, `forecast-local-forecasts-unweighted-pool.csv`,
`forecast-local-forecasts-equal-mass.csv`, `forecast-local-timing.json` and
`made-after-sealed-run.txt`. Read, per unit and tier: the mean offers and losses of the main model
and of both refits side by side, and each refit's paired difference from main with its 5th and
95th percentile over the splits (the spread over the days of one unit, not an interval for new days).
The line "main, all days" is the main model scored on all the unit's days with the splits of
`dial_choice`, to be read against the recorded main figures; it is not part of the comparison.

How the results are read (block `reading` of `forecast-local.json`, fixed before any result and
printed in the summary and the results' `definitions`). The trial gives eight comparisons (two
refit arms, two units that are scored, two tiers); four are judged:

- Judged: tier A, per unit and per refit arm: the paired mean difference of the offer (arm minus
  main) over the 200 splits, as a share of the main arm's mean offer.
- Noise bar per unit: 8% at the workplace unit `nrel-from-may-2019` and 30% at the pooled
  Norwegian unit `norway-development-pooled`, the shares by which the seed alone moved an offer
  (`docs/prior-looks.md`, the entry that begins with the words "The neural model's seed test").
  The bars are the neural model's range over three seeds (with the plan's own runs) at one level,
  kappa 0.3. The refitted tree model has one fixed seed, so the bars are an analogy and not this
  trial's measured noise. At the pooled Norwegian unit a difference of up to 30% therefore reads
  "no measurable difference".
- An arm "shows a gain at a unit" only if that share is above the unit's bar; "shows a loss" only
  if it is below minus the bar; otherwise "no measurable difference". The summary prints that
  word for each of the four tier A cells, and tier B's four cells as numbers only, labelled
  information.
- The trial's one line: "the refit helps" only if at least one arm shows a gain at a unit and no
  arm shows a loss at any unit; "the refit hurts" if any arm shows a loss and none a gain;
  otherwise "no measurable difference; a mixed result is reported as mixed". The line is "not
  read", with the reason stated in the summary and the results, when a unit that has a bar in the
  settings file has no judged cell (left out for too few days, failed, or not run); the unit
  `nrel-before-may-2019` has no bar and is not judged, so its absence does not stop the reading.
- The 200 splits share the unit's days, so the standard error over splits understates the
  uncertainty; and the trial cannot speak for the Dutch or Korean sites, none of which is a
  development unit.

Choices left open, not taken: another weight than the two; a refit before each forecast day; a site
term in place of a refit; another k or several k; a dial chosen again for a refitted model.

## Trial forecast-noset: the neural model without its set encoder

APPROVED TO RUN by the owner's decision of 24 November 2026: the trial runs after the approved
trials, on development data only, and its figures are reported apart. The committed settings file
says `approved_to_run: true`; the command ends with exit 2 ("not approved to run") if that file
does not say `true`. The approval is read from the committed settings file only: a copy of the
settings given with `--settings`, with or without `--not-the-trial`, cannot grant it (a test pins
both routes). The trial was run once, on 24 November 2026, at commit 0f07d1b: start 21:21:18 UTC,
end 22:12:49 UTC, exit 0. One earlier start, at commit f5ab5fb at 21:11:44 UTC, ended with exit 2
because the committed settings file did not yet say `approved_to_run: true`, and made no figure.

The question: is the neural departure model without its set encoder (`hazard_tuned_no_set`, the
ablation `no_set_encoder` of plan section 7) better than the same model with its set encoder
(`hazard_tuned`), than the neural model the project reports (`hazard_tuned_inputs`) and than the
main model (`lgbm_tail`), with the neural models run on three seeds? One earlier look (the
`docs/prior-looks.md` entry beginning "Dial-choice runs: as in the dial-choice runs at b9dd078",
one seed, the plan's dial rule) gave the no-encoder model 20.21 and 3.19 kWh per day at the two
development units; the entry beginning "at a8de44f: `20261113T190537Z-wp2-truth`" gives 17.26 and
2.73 for `hazard_tuned` and 17.08 and 2.24 for `lgbm_tail`. The seed alone moved a neural figure
by about 8% at workplaces and 30% at pooled Norway (the entry beginning "The neural model's seed
test"), so one seed proves nothing. Nothing is adopted: the main model, the plan, the pass marks
and every held-out figure stay as they are.

What differs between the models (read from the source, not guessed; a test reads the same text):

- `hazard_tuned_no_set` against `hazard_tuned`: one difference. `hazard_tuned_no_set` is built with
  the same tuned settings file and the same inputs, fit units and levels, and its settings are
  replaced by `tuned.hazard(**PARTS['no_set_encoder'])`, where `PARTS['no_set_encoder']` is
  `{'set_encoder': False}` (`models/hazard_tuned_no_set.py`, `hazard_parts.py`). The judged
  comparison is this pair.
- `hazard_tuned_inputs` against `hazard_tuned`: one difference. It is `hazard_tuned` with fourteen
  extra input columns (`EXTRA_INPUTS` of `models/hazard_tuned_inputs.py`: energy drawn before the
  decision time, four regime flags, four weather columns, one stringency index, public holidays,
  9/80 Fridays, school holidays and academic breaks), each given as a standardised value and a
  missing flag; the same tuned settings file, the set encoder on.
- `hazard_tuned_no_set` against `hazard_tuned_inputs`: two differences (the set encoder is off, and
  the extra input columns are absent). A comparison against it cannot say which difference counts.
- `lgbm_tail` against the neural models: many differences. `lgbm_tail` is LightGBM quantile
  boosters, one per level, on the feature columns of `learn_features` plus the lower-tail history
  of the user's earlier sessions (`models/lgbm_tail.py`); it is a different kind of model.
- Every neural model is run at seeds 1, 2 and 3; the seed replaces only the seed of the model's
  settings (`dial_choice.with_fit_seed`). `lgbm_tail` has no seed option and runs once.

Command (one run, under the run recorder, from a clean commit, nothing else running; `--workers`
changes the number of processes only; every option that changes a setting (`--kappa`, `--splits`,
`--seeds`, `--model-tail`, `--model-tuned`, `--model-inputs`, `--model-noset`, `--settings`,
`--matrix`) is refused unless `--not-the-trial` is also given, which labels the run as not the
trial; `--tables` and `--truth-run` change the data read and are reported, not refused):

```
python -m fleetreserve.run_record --name forecast-noset --long --max-seconds 172800 -- uv run --frozen --group learn python -m fleetreserve.forecast_noset --workers 6
```

Output files in the run folder: `forecast-noset.json` (settings, inputs, the runs, the paired
comparison on both lines and the reading; read `reading` first), `forecast-noset-summary.txt`,
`forecast-noset-timing.json`, `made-after-sealed-run.txt`, and per run (`tail`, `tuned-seed<N>`,
`inputs-seed<N>`, `noset-seed<N>`) `forecast-noset-<run>-dial-whatif.json` (the full results of
`site_dial_whatif.run`) and `forecast-noset-<run>-forecasts.csv`. Every file carries the
after-sealed-run label. A run given `--not-the-trial` also writes `not-the-trial.txt`, and its
`made-after-sealed-run.txt` carries the same words as its third line.

Duration, a guess and not a measurement: about 2 hours, between 1.5 and 3.5 hours, so under the
four hours at which a note would be due. Basis: ten runs (`lgbm_tail`, then each of three neural
models at three seeds), each one fit of its model on the training units and one measurement of the
eight dials of the family. The first version of this trial had seven runs and was guessed at about
1.5 hours; three more runs add about 40%. The judged line shares its fits and evaluations with the
check line (both are read from the one measurement of each run, see below), so the two lines do not
double the run. A recorded `dial_choice` run of the tuned neural model (nine configurations and
their oracle partners, 200 splits, six workers) took about 6 minutes (the dial-choice runs at
a8de44f, the entry beginning "at a8de44f: `20261113T190537Z-wp2-truth`" of `docs/prior-looks.md`);
the measurement here has fewer configurations. The fit time alone is not recorded, and a fit of a
neural model with the plan's inputs may be slower than the recorded one.

What is fixed, all in `forecast-noset.json` in this folder with its source, before any result:

- development units only: `nrel-from-may-2019` and `norway-development-pooled` are judged,
  `nrel-before-may-2019` is printed and not judged; tier A is the judged tier, tier B is
  information; a matrix with any other unit is refused before a table is read, and every held-out
  name is refused (`forecast_mix.check_units`, and a test of the whole list);
- level 0.5, 200 seeded splits, the dial family of `gate/matrix.json` at that level, the same
  splits for every run, so every comparison is paired by split;
- ten runs: `lgbm_tail` once (its own fixed seed), and `hazard_tuned`, `hazard_tuned_inputs` and
  `hazard_tuned_no_set` each at seeds 1, 2 and 3, the seeds of the earlier seed tests (the entry
  beginning "The neural model's seed test" and the entry beginning "The neural model at kappa 0.4
  and 0.5, seeds 1 to 3" of `docs/prior-looks.md`); every model is fitted on the training units only;
- two lines of figures, both read from the one measurement of each run (`site_dial_whatif.run`
  stores, per split, the offer of every dial calibrated on all the calibration days, and the offer
  of the dial chosen from calibration days alone), so the second line needs no further fit or
  evaluation:
  - the JUDGED line: the project's standard development rule for the dial (plan section 6,
    `dial_splits.pick_dial`: per run, unit and tier, the dial with the largest mean offer over the
    same 200 splits, a tie to the smaller dial), applied identically to every arm and every seed
    (each seed of a neural arm has its own dial). This rule reads the test-day offers of the
    development units, for every arm alike, as every development figure of the project does; it is
    not free of test-day information and it is the line the earlier figures of the project were
    made on. The bars of the reading apply to this line;
  - the CHECK line: variant S1 of the site-dial what-if (`site_dial_whatif`), the dial chosen from
    calibration days alone. Per split, the calibration days are cut in two by a seeded draw that
    looks at no loss; the dial with the largest calibrated offer on the first half is taken, its
    scale is calibrated on the second half and scored on the test days. The dial is a function of
    the first half only, so no test day's outcome reaches it (a test changes the test days' losses
    and sees the choice unchanged). When no dial has an offer above 0 on the first half, S1 takes
    a fallback dial, fixed as the smallest dial of the family, which reads no outcome. Its offers
    may be lower than the judged line's (in an earlier look at level 0.3 under S1, the entry
    beginning "The site dial (the third attack of the owner's decision of 16 Nov" of
    `docs/prior-looks.md`, `lgbm_tail` offered about half of the fixed dial's figure), so the bars
    do not apply to it and no verdict is formed on it: only the sign and the size of each
    difference are reported, beside the judged line;
- what a disagreement means. Only a check-line difference of the opposite sign to a gain or a loss
  that the judged line shows at a judged unit (a gain with a negative check difference, a loss with
  a positive one) makes the one-line reading of that comparison "mixed": the gain, or the loss, may
  come from the choice of the dial on the test days that the judged line allows, and the check
  line, which does not, contradicts it. A check-line difference of exactly 0 (within one billionth
  of a kWh; for example every arm offers 0 under the stricter rule at the pooled Norwegian unit) or
  an undefined one (both arms offer 0, or the check has no figure) is silent: it neither confirms
  nor contradicts, the judged line's word stands, and the output says for that unit "check line
  silent (difference 0)" or "check line silent (no figure for the check line)". Where the judged
  word is "no measurable difference" the check line cannot change it; its sign and size are
  printed beside it. The output counts, per comparison, the judged units at which the check line
  was silent. A reading that rests on the judged line alone has not been checked against dial
  selection on test days;
- the reading (`forecast_noset_reading`, the `reading` block): per judged unit a bar, 8% at the
  workplace unit and 30% at the pooled Norwegian unit; a neural model's figure is the mean over its
  three seeds; the share is the paired mean difference of the offer (no-encoder arm minus the
  comparison arm) over the 200 splits as a share of the comparison arm's mean offer. Words as in
  the mix trial: a gain only above the bar, a loss only below minus the bar, else no measurable
  difference; and a gain only if every one of the no-encoder arm's three seed figures is above
  every seed figure of the comparison arm (for `lgbm_tail`, its one figure), a loss only if every
  one is below. One line per comparison, each named for what it is: against `hazard_tuned` (the
  judged comparison, one difference), against `hazard_tuned_inputs` (the neural model the project
  reports, two differences) and against `lgbm_tail` (the main model, a different kind of model):
  better (a gain at one unit and no loss at the other), worse, no measurable difference, or mixed
  (a gain at one unit and a loss at the other, or a check-line sign opposite to a gain or a loss of
  the judged line; a silent check line never makes it mixed). The
  overall line is "better than all three" only if all three lines read better, "worse than all
  three" only if all three read worse, otherwise "no clear ordering". An arm that fails gives
  NOT_READ for what needs it. The reported neural model against `lgbm_tail` is printed as
  information and never read.

Limits. No Dutch or Korean site is a development unit, so the trial cannot speak for them, though
the Dutch units hold most of the gap between the offer and its ceiling. The bars are the neural
model's seed spread from an earlier look at kappa 0.3, taken at the best dial of each fit: an
analogy, not the spread of this trial. With three seeds per neural arm the spread of this trial is
reported as well, the least and the greatest seed figure of each arm and unit, on both lines, beside
the mean; three seeds measure it only roughly. The seeded splits share days, so the standard error
over the splits understates the uncertainty of the whole experiment. The judged line reads
test-day offers (as every development figure of the project does) and so cannot show whether a gain
would survive a dial chosen without them; the check line is there for that, at the price of lower
offers that the bars do not fit. The command's `--tables` and `--truth-run` are reported in the
output (`inputs_that_differ_from_the_defaults`), not refused: a run on other tables is not
detected as such by the settings check beyond that report. No arm, level, seed or setting is added
after results are seen.

Why no test day's outcome reaches a model's forecast or a split's calibration: each model is fitted
on the training units only (`learn_run.check_model` refuses another fit unit and a model fitted on
a scored unit); it forecasts from `learn_run.forecast_frame`, which holds only values known at the
decision time; on both lines a split's scale is calibrated on calibration days only
(`dial_splits.score_split`); the test days are not in the calibration days
(`dial_choice.split_indices`). On the check line the dial too is chosen from calibration days only
(`site_dial_whatif.split_arms`); on the judged line it is chosen by the plan's rule from mean
offers that include the test days, as stated above.

What has to exist before it runs: a recorded, verified truth build of the training and development tables; the `learn` dependency
group; the context files the neural model with the plan's inputs reads (weather and OxCGRT under
`data/raw/`, fetched by `tools/fetch_data.py --context`). No stored fit exists, so the command fits
each model inside the run.

Not tested by the author of this trial: the author ran nothing here, including the tests under
`tests/forecast_trials_noset/`, the linter and the formatter.

## Trial forecast-energy: more cautious energy bounds

APPROVED TO RUN by the owner's decision of 24 November 2026: this one trial, as fixed in the
settings file, on development data only, once, reported apart from the plan's results. The
committed settings file says `approved_to_run: true`; the command ends with exit 2 ("not approved to
run") if that file does not say `true`. The approval is read from the committed settings file only:
a copy of the settings given with `--settings`, with or without `--not-the-trial`, cannot grant it
(a test pins both routes). The trial was run once, on 25 November 2026, at commit e288876: start
01:17:57 UTC, end 01:33:00 UTC, exit 0.

The question: the held-out units offered nothing. Is part of the loss at the development units due to
the energy bounds, that is, to the car's top charging speed and storable energy being forecast too
high? The model forecasts three energy bounds per car beside the departure: the top charging speed
(rate), the storable energy (the car's room) and the needed energy. Known energy takes the car's
true values (plan section 7, first bullet); the project's empirical bounds take rate and room as the
lower quantile at the run's level (0.2 here) and the need as the 0.8 upper quantile. The trial adds
one value, `cautious` (`forecast_empirical.CAUTIOUS`, `gate_config.ENERGY_BOUNDS`): rate and room at
the plan's lowest level, 0.05, whatever the level of the departure forecast, and the needed energy
unchanged. The departure forecast is the same in every arm.

Three arms, each a `dial_choice` measurement of `lgbm_tail` at departure level 0.2 (the dial family
of `gate/matrix.json` at 0.2, the one-way re-solve rung, 200 seeded splits, the same splits in every
arm), each read at its own dial (the dial with its largest mean tier A offer, a tie to the smaller):

1. `known`: known energy bounds, the reference line;
2. `empirical`: the project's empirical energy bounds at level 0.2;
3. `cautious`: the new bounds.

The model is fitted once on the training units and every arm reads the same forecast table; the
table holds level 0.05 (`dial_choice.make_factory` predicts at every plan level up to the run's
level), so no new fit is needed. The arms differ in the energy bounds alone.

Command (one run, under the run recorder, from a clean commit, nothing else running; `--workers`
changes the number of processes only; every option that changes a setting (`--kappa`, `--splits`,
`--model`, `--settings`, `--matrix`) is refused unless `--not-the-trial` is also given, which labels
the run as not the trial; `--tables` and `--truth-run` change the data read and are reported, not
refused):

```
python -m fleetreserve.run_record --name forecast-energy --long --max-seconds 172800 -- uv run --frozen --group learn python -m fleetreserve.forecast_energy --workers 6
```

Output files in the run folder: `forecast-energy.json` (settings, inputs, the three arms' dial
choices, the paired comparison and the reading; read `reading` first), `forecast-energy-summary.txt`,
`forecast-energy-timing.json`, `forecast-energy-forecasts.csv` (the one forecast table, its hash in
every arm's record), `made-after-sealed-run.txt`, and per arm (`known`, `empirical`, `cautious`)
`forecast-energy-<arm>-dial-choice.json` and `forecast-energy-<arm>-split-offers.json`. Every file
carries the after-sealed-run label, and the label line is on top of the summary and the label file.
A run given `--not-the-trial` also writes `not-the-trial.txt`, and its `made-after-sealed-run.txt`
carries the same words as its third line.

Duration, a guess and not a measurement: about 40 minutes, between 25 and 70 minutes. Basis: the
mix trial (three measurement runs and two fits) took about 44 minutes and trial a about 16 minutes
(as the written design of this trial records them, not checked again here); this trial has
three measurement runs and one fit. An arm with
more empty forecast sets may run faster, because a day that abstains is not evaluated at every
scale.

What is fixed, all in `forecast-energy.json` in this folder with its source, before any result:

- development units only: `nrel-from-may-2019` and `norway-development-pooled` are judged,
  `nrel-before-may-2019` is printed and not judged (it has no bar); tier A is the judged tier, tier B
  is information; a matrix with any other unit is refused before a table is read, and every held-out
  name is refused (`forecast_mix.check_units`, and a test of the whole list);
- the model `lgbm_tail`, departure level 0.2, rate and room level 0.05, need level 0.2, 200 seeded
  splits, the three arms above; no arm, level, bar or setting is added or changed afterwards;
- the reading (`forecast_energy_reading`, the `reading` block). G is the paired mean difference of
  the offer, arm 3 (cautious) minus arm 2 (empirical), over the 200 splits. The bar is the unit's
  share (8% at the workplace unit, 30% at the pooled Norwegian unit, the neural model's seed spread
  at kappa 0.3 from `docs/prior-looks.md`, an analogy and not this trial's measured noise) times the
  mean offer of arm 1. "Shows a gain" if G is above the bar and arm 3's mean test slot loss is at
  most 0.10 and its mean test driver loss at most 0.05; "shows a loss" if G is below minus the bar;
  otherwise "no measurable difference" (also when a gain above the bar has a loss over its limit,
  where the output says so, or when arm 1's offer is 0). One line: "the cautious bounds help" if one
  judged unit shows a gain and none a loss; "the cautious bounds hurt" if one shows a loss and none a
  gain; otherwise "no measurable difference; a mixed result is reported as mixed". Also printed: the
  mean offers of arms 2 and 3 as a share of arm 1's, G as a share of arm 1's offer, and G as a share
  of the gap between arm 1 and arm 2 (none unless that gap is above 0).

What a result would and would not show:

- A gain: on development data, at these two units and at departure level 0.2, cautious rate and room
  bounds recover offer that the project's empirical bounds lose against known energy. It would not
  show that the held-out zero came from the energy bounds, it would not change the main model, the
  plan, a pass mark or a held-out figure, and it would be a reason to ask for a further trial.
- A loss: only that level 0.05 for rate and room, at these two units and at departure level 0.2,
  makes the offer smaller than the empirical bounds do. The trial tries one setting, so a loss does
  not show that other settings would do the same. Still untried: levels between 0.05 and the level
  of the empirical bounds, a stricter need bound, other departure levels and other models. The offer
  need not move in one direction with the level, so a loss at 0.05 says nothing about a level
  between.
- No measurable difference: the paired difference is inside a bar that is an analogy to another
  model's seed spread. It does not show that the bounds are equal, and with these bars a small real
  difference would not be seen.

Limits. No Dutch or Korean site is a development unit, so the trial cannot speak for them. The
seeded splits share days, so the standard error over the splits understates the uncertainty of the
whole experiment. The dial is chosen by the plan's rule from offers that include the test days, for
every arm alike, so the figures are not free of test-day information. One fit serves the three
arms, so a different fit could give a different difference. The guarantee is untouched: a bound
that is exceeded is counted in the replayed slot and driver losses that set the scale, so a smaller
bound costs offer and not validity; calibration, the certificate, the loss limits, the plan's
definitions and every pass mark are as they are.

Why no test day's outcome reaches the forecast or a split's calibration: the model is fitted on the
training units only (`learn_run.check_model` refuses another fit unit and a model fitted on a scored
unit); it forecasts from `learn_run.forecast_frame`, which holds only values known at the decision
time; a split's scale is calibrated on calibration days only (`dial_splits.score_split`); the test
days are not in the calibration days (`dial_choice.split_indices`).

What has to exist before it runs: a recorded, verified truth build of the training and development
tables, and the `learn` dependency group. No stored fit exists, so the command fits the model inside
the run.

Not run by its author: the author did not run the trial (it was run later, once, as stated above),
and no `data/` or `runs.noindex/` folder was opened; the tests under
`tests/forecast_trials_energy/` use made-up data only.

## Trial forecast-energy-2: all three energy bounds cautious, and the needed energy alone

APPROVED TO RUN by the owner's decision of 25 November 2026: "Yes, build and run"; this one trial,
as fixed in the settings file, on development data only, once, reported apart from the plan's
results and apart from the first energy trial, whose figures stand as its result and are not
replaced. The committed settings file (`forecast-energy-2.json`) says `approved_to_run: true`; the
command ends with exit 2 ("not approved to run") if that file does not say `true`. The approval is
read from the committed settings file only: a copy of the settings given with `--settings`, with or
without `--not-the-trial`, cannot grant it (a test pins both routes). The trial was run once, on 25
November 2026, at commit 4ebe5fe: start 12:43:30 UTC, end 13:06:01 UTC, exit 0.

The question: the first energy trial made the top charging speed and the storable energy cautious
(level 0.05) and left the needed energy at its 0.8 quantile; at the two judged units the paired
difference stayed inside the bar. One setting does not show that no setting helps. This trial adds
two values of `energy_bounds` beside `cautious`, written `cautious-need` and `cautious-all`
(`forecast_empirical.CAUTIOUS_NEED`, `CAUTIOUS_ALL`; `gate_config.MATRIX_ENERGY_BOUNDS`):

- `cautious-need`: the needed energy alone at its level-0.05 upper quantile (the 0.95 quantile); the
  top charging speed and the storable energy as in the empirical bounds at level 0.2;
- `cautious-all`: all three bounds at level 0.05: the top charging speed and the storable energy at
  the 0.05 lower quantile, the needed energy at the 0.95 upper quantile.

The departure forecast stays at level 0.2 in every arm. Five arms, each a `dial_choice` measurement
of `lgbm_tail` at departure level 0.2 (the dial family of `gate/matrix.json` at 0.2, the one-way
re-solve rung, 200 seeded splits, the same splits in every arm), each read at its own dial (the dial
with its largest mean tier A offer, a tie to the smaller):

1. `known`: known energy bounds, the reference line;
2. `empirical`: the project's empirical energy bounds at level 0.2;
3. `cautious`: the first trial's setting, re-run so that every arm is paired split by split;
4. `cautious-need`: the needed energy alone cautious; information only;
5. `cautious-all`: all three bounds cautious; the judged arm.

The model is fitted once on the training units and every arm reads the same forecast table; the
table holds level 0.05 in every column (`dial_choice.make_factory` predicts at every plan level up to
the run's level, and the needed energy of a row at level 0.05 is the 0.95 quantile), so no new fit
is needed. The arms differ in the energy bounds alone. The three settings tried, as the output
defines them: `cautious` (arm 3), `cautious-need` (arm 4) and `cautious-all` (arm 5).

Command (one run, under the run recorder, from a clean commit, nothing else running; `--workers`
changes the number of processes only; every option that changes a setting (`--kappa`, `--splits`,
`--model`, `--settings`, `--matrix`) is refused unless `--not-the-trial` is also given, which labels
the run as not the trial; `--tables`, `--truth-run` and `--first-run` change the data read or the
comparison made and are reported, not refused). The option `--first-run` names the run folder of the
first energy trial; the command reads that folder's `forecast-energy.json` and no other file. The
trial is run with `--first-run`, so that arms 1 to 3 are compared with the first run; a run without
it still works and says that the first run was not compared. In the command, the placeholder after
`--first-run` stands for the run folder of the first energy trial:

```
python -m fleetreserve.run_record --name forecast-energy-2 --long --max-seconds 172800 -- uv run --frozen --group learn python -m fleetreserve.forecast_energy_2 --workers 6 --first-run <run folder of the first energy trial>
```

Output files in the run folder: `forecast-energy-2.json` (settings, inputs, the five arms' dial
choices, the paired comparisons, the reading and the check against the first run; read `reading`
first, then `first_run_check`), `forecast-energy-2-summary.txt`, `forecast-energy-2-timing.json`,
`forecast-energy-2-forecasts.csv` (the one forecast table, its hash in every arm's record),
`made-after-sealed-run.txt`, and per arm (`known`, `empirical`, `cautious`, `cautious-need`,
`cautious-all`) `forecast-energy-2-<arm>-dial-choice.json` and
`forecast-energy-2-<arm>-split-offers.json`. Every file carries the after-sealed-run label, and the
label line is on top of the summary and the label file. A run given `--not-the-trial` also writes
`not-the-trial.txt`, and its `made-after-sealed-run.txt` carries the same words as its third line.
The label sentence names the UTC date on which the run started, read from the clock once at the
start of the run and used in every output of the run; the rest of the sentence is the one every
other trial carries. The sealed run ended on 24 November 2026.

Duration, a guess and not a measurement: about 25 minutes, between 15 and 45. Basis: the first
energy trial took about 15 minutes for three measurement runs and one fit (a figure recorded
for that trial, not checked again here); this trial has five measurement runs and one fit, so about five
thirds of that, with room for the arms that have more empty forecast sets and so run faster or
slower.

What is fixed, all in `forecast-energy-2.json` in this folder with its source, before any result:

- development units only: `nrel-from-may-2019` and `norway-development-pooled` are judged,
  `nrel-before-may-2019` is printed and not judged (it has no bar); tier A is the judged tier, tier B
  is information; a matrix with any other unit is refused before a table is read, and every held-out
  name is refused (`forecast_mix.check_units`, and a test of the whole list);
- the model `lgbm_tail`, departure level 0.2, rate and room level 0.05, needed energy level 0.05 in
  arms 4 and 5 and 0.2 in arms 2 and 3, 200 seeded splits, the five arms above; no arm, level, bar,
  limit or rule is added or changed afterwards;
- ONE judged comparison (`forecast_energy_2_reading`, the `reading` block): arm 5 minus arm 2, per
  split. G is the paired mean difference of the offer over the 200 splits. The bar is the unit's
  share (8% at the workplace unit, 30% at the pooled Norwegian unit, the neural model's seed spread
  at kappa 0.3 from `docs/prior-looks.md`, an analogy and not this trial's measured noise) times the
  mean offer of arm 1. "Shows a gain" if G is above the bar and arm 5's mean test slot loss is at
  most 0.10 and its mean test driver loss at most 0.05; "shows a loss" if G is below minus the bar;
  otherwise "no measurable difference" (also when a gain above the bar has a loss over its limit,
  where the output says so). A unit where arm 1's mean offer is 0 is printed as not judged, since no
  bar can be defined, and gets no word. One line: "all three energy bounds at level 0.05 help" if
  one judged unit shows a gain and none a loss; "all three energy bounds at level 0.05 hurt" if one
  shows a loss and none a gain; otherwise "no measurable difference; a mixed result is reported as
  mixed". The line names the setting;
- arm 4 minus arm 2 is printed with the same figures (G, the shares of arm 1's offer, the losses) as
  information only: it gets no gain, loss or no-difference word, it is read by no rule, and it enters
  no one-line reading;
- the re-run arms: arms 1 to 3 are measured again here and the first trial's figures stand. With
  `--first-run` the command prints, for arms 1 to 3 and each judged unit, this run's mean offer
  beside the first run's and the word FAULT with both numbers for any difference at all (exact
  equality, no tolerance). A fault is printed and is not chosen between; the reading of arm 5
  against arm 2 is given from this run's own arms either way. Without the option the output says
  the first run was not compared. The field `first_run_faults` of `reading` in
  `forecast-energy-2.json` holds the list of the faults (the entries of `first_run_check.faults`):
  an empty list when the first run was compared and every figure is equal, null when the first run
  was not compared. The first lines of the summary, before any figure, say one of: the count of
  FAULT lines, "First run compared: equal", or "First run not compared". A fault changes no figure,
  no reading word and not the exit status.

What a result would and would not show:

- A gain: on development data, at these two units and at departure level 0.2, all three energy
  bounds at level 0.05 recover offer that the project's empirical bounds lose against known energy.
  It would not show that the held-out zero came from the energy bounds, it would not change the main
  model, the plan, a pass mark or a held-out figure, and it would be a reason to ask for a further
  trial.
- A loss: only that this one setting, all three bounds at level 0.05, at these two units and at
  departure level 0.2, makes the offer smaller than the empirical bounds do. The judged comparison
  tries one setting, so a loss does not show that other settings would do the same, and it does not
  show that no change of levels helps. Still untried: levels between 0.05 and 0.2, other departure
  levels, another model and energy rows in the certificate. The offer need not move in one direction
  with the level.
- No measurable difference: the paired difference is inside a bar that is an analogy to another
  model's seed spread, at the setting tried. It does not show that the bounds are equal or that no
  setting helps, and with these bars a small real difference would not be seen.
- Arm 4 against arm 2 shows nothing by a rule: its figures are information.

Limits. No Dutch or Korean site is a development unit, so the trial cannot speak for them. The bars
are an analogy to another model's seed spread at another level and not this trial's measured noise.
The seeded splits share days, so the standard error over the splits understates the uncertainty of
the whole experiment. The dial is chosen by the plan's rule from offers that include the test days,
for every arm alike, so the figures are not free of test-day information. One fit serves the five
arms, so a different fit could give a different difference. The guarantee is untouched: a bound that
is exceeded is counted in the replayed slot and driver losses that set the scale, so a smaller bound
costs offer and not validity; calibration, the certificate, the loss limits, the plan's definitions
and every pass mark are as they are. The run guard compares the values of a run with the committed
settings file on disk, not with git, so a run starts from a clean commit; a test pins that the
committed file is the one the code and the plan give.

Why no test day's outcome reaches the forecast or a split's calibration: the model is fitted on the
training units only (`learn_run.check_model` refuses another fit unit and a model fitted on a scored
unit); it forecasts from `learn_run.forecast_frame`, which holds only values known at the decision
time; a split's scale is calibrated on calibration days only (`dial_splits.score_split`); the test
days are not in the calibration days (`dial_choice.split_indices`).

What has to exist before it runs: a recorded, verified truth build of the training and development
tables, and the `learn` dependency group. No stored fit exists, so the command fits the model inside
the run.

Not run by its author: the author did not run the trial (it was run later, once, as stated above),
and no `data/` or `runs.noindex/` folder was opened; the tests under
`tests/forecast_trials_energy/` use made-up data only.
