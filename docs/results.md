# Results

This page gives the results of the one run made on held-out data (data that no loader opened until the plan was frozen; the earlier looks, among them the program that split the Norwegian file by site, are in section 1), and, in section 15, trials made after that run. Each part opens with what failed, what is missing and what limits the reading, and gives the figures after that. The held-out run did not meet its own pass mark (the test the plan set for it); read section 1 first.

The plan numbers its outputs, and sections 3 to 13 follow them: O1 the offer, O2 the exact-ceiling check, O3 simpler published sets against the ceiling, O4 the guarantee, O5 properties that must hold, O6 the ledger of where the offer is lost, O7 worst-case request paths, O8 two-way charging, O9 cold start, O10 models against each other. A work package (WP) is a unit of the plan with its own pass mark.

Every table named below is a file written by one command (section 17) whose manifest file `wp16-manifest.json`, written beside the tables, names the commit of the code, and the rows shown here are copied from those files cell by cell, with nothing added up or rounded on this page. A figure that is not in a table file is marked "read from the run's results file" and is given only where the table files leave it out. The trial figures of section 15 come from the trials' own summary files, not from that command.

## 1. Status of the run

**What failed.**
- The run has 30 steps. 26 ended with exit status 0 (passed) and 4 ended with exit status 1 (failed). The four are interval steps: the steps that compute 95% intervals for the main model's headline offer, the neural model's headline offer, the forecast-energy setting and the held setting. The run is therefore not a passed held-out run, and the plan's pass mark for it (work package 14) returned status 1, not passed.
- The cause, read from the run's results file, is the same in all four steps. At four Korean units the largest possible offer on a few test days (the hindsight ceiling, exactly zero in truth) was computed as about -1e-14 kWh. The learning step then wrote a missing share of the ceiling, the interval step recomputed it as 0.0, found the two unequal and stopped. It is a rounding at zero, not a sign error and not a timeout.
- The plan's second pass mark (work package 15, the sensitivity settings) also returned status 1, because it counts as not passed after a run whose first mark fails. The rows it covers are printed as information only (section 5, and the drift guard in section 10).
- Under reading A (section 6) no unit passes the check of the guarantee, and six of the nine units offer nothing (section 3).
- Intervals computed again afterwards, apart from the run, exist for the failed interval steps. They were made after held-out output had been seen, fill no slot of the plan, and are named as "made apart" wherever they appear. The headline's pass mark is not judged from them.

**What ran.** The run was started three times from the frozen plan. The first start (23 November 2026) stopped because the reader of the Dutch file refused the real file. The second start stopped because the reader of the Korean file refused its file. Each reader was corrected to the file's layout (a layout look at the first line, column shapes and record counts of the two files and the labels of one column of each file, taken outside the loader's log), each correction changed only how a file is read, and each is recorded in the plan's list of changes (section 14 of the plan). The third start ran to the end, at commit a14654659234650660bc88285530ad1b2bf6be77, and ended at 06:03 UTC on 24 November. All nine units were built in both settings; none was refused at the truth build.

**Looks taken after the plan was frozen.** Besides the layout look above, a scan of the three held-out files by the corrected readers (counts and pass or fail only), reads of the run's stored outputs while it went, five runs of a corrected interval step apart from the run, and a read of stored outputs for the pass marks and tables were all made after the freeze. Each is an item of `docs/prior-looks.md` (the items from 51 on follow the freeze). Four of the five apart runs are covered by a recorded approval; the fifth, which read the stored output of a setting whose own interval step had passed, to check that the correction changes no number, is not. The layout look and the scan read held-out files outside the loader and its log; they showed layout and counts, never an outcome, and came before any result. Before the freeze, the program that split the Norwegian file by site parsed the file holding the two held-out Norwegian sites, wrote the held-out rows to a file of their own and printed header names, file sizes, row counts and hashes (item 13). The record shows no opening of a held-out file by the later reads of stored outputs, and none by the apart runs. What was chosen after reading stored outputs was the correction of the interval step, the wording of working summaries, the rule for the combined figures of section 9 and the trials of section 15.

Table files: `steps.md`, `units.md`, `outcomes.md` (all tables of the run in one file), `wp16-manifest.json` (the command, commits and hashes of the run).

The four failed steps, copied from `steps.md` (lines 20, 21, 24, 25):

| step | status | reason |
|---|---|---|
| intervals-lgbm_tail | failed | the run ended with exit status 1 |
| intervals-hazard_tuned_inputs | failed | the run ended with exit status 1 |
| intervals-lgbm_tail-forecast-energy | failed | the run ended with exit status 1 |
| intervals-lgbm_tail-held | failed | the run ended with exit status 1 |

The other rows of `steps.md` read "passed": the four truth builds, the learning step of each model and setting, the two sensitivity interval steps (rho 0.1 and 0.2), the ledger, the comparison, the corner search, the two cold-start steps, the drift guard, O3 and the two O2 steps. `units.md` lists nine units, each with window "built" and no reason.

## 2. How to read the tables

- **Unit.** A car park site, or one regime of a site, or a portfolio of sites. Nine held-out units: the Dutch office car park before March 2023 (`asr-before-march-2023`) and from March 2023 (`asr-from-march-2023`, when its managed-charging controller began); two Norwegian sites (`OSL_T`, `TRO_R`); five Korean portfolios of slow chargers (`korea-home`, `korea-lodging`, `korea-work`, `korea-public-parking`, `korea-visits`). The Korean file has no charging-rate column, so a rate of 7 kW is assumed. No model was fitted on any held-out unit.
- **Box and offer.** At a fixed decision time each morning the car park sells a box: for each of up to eight later hours (the service hours) a range of total charging power in kW, which the cars' combined charging must meet whatever request inside the range arrives. The offer is the total width of the ranges in kWh.
- **Counting day, test day.** A counting day is a day with at least 3 cars plugged in at the decision time. 100 counting days of a unit, drawn at random with a fixed seed (half of them when a unit has fewer than 200), are its calibration days; the rest are its counting test days. An offer is given per counting test day, and a day with no offer counts as 0.
- **Hindsight ceiling.** The largest box that any rule could keep if every car's arrival, departure, rate and energy and every hour's request were known in advance. It is solved exactly. The best-causal ceiling is the largest box a rule can keep that learns each request only when it arrives, with the cars known.
- **Scale, calibration.** The certified box is multiplied by a scale between 0 and 1, chosen on the calibration days so that two losses stay within limits. Scale 0 means the method offers nothing (abstains) for that unit.
- **Losses and limits.** The slot loss of a day is the share of service hours from the first failed hour to the end of the window (limit 0.1 on average, so availability of at least 90%). The driver loss of a day is the share of the cars' protected energy not delivered at departure (limit 0.05, called the shortfall). A limit holds in expectation over the random choice of calibration days, for days taken as exchangeable (days that can be treated as drawn at random in the same way), by conformal risk control (a method that sets the scale from past days so that the average loss stays within the limit; statement T6 below); it is not promised for every unit or every day.
- **Tiers.** Tier A (driver-protected) uses the smaller scale of the two losses and is the tier that is judged. Tier B uses the slot loss alone; its driver loss is reported. The tiers are never pooled.
- **Rungs.** The fixed plan runs the certificate's own policy. The one-way re-solve solves the remaining hours again each hour as cars leave (charging only). The two-way rung also lets cars discharge, up to a 5 kWh budget per car.
- **Configuration names.** `main-k0.5-d0.02-empirical` reads: level (the letter k, kappa) 0.5 of the departure forecast (its median), driver-risk dial 0.02 (it limits the expected shortfall of a car's needed energy, over lower departure levels, to that share of the energy the car needs; `none` means no such limit), forecast departures (`oracle` means true departures). In the main setting only departures are forecast; each car's rate, room (energy it can still store) and needed energy are taken as known.
- **Models.** `lgbm_tail` is the main model (boosted quantile trees). `hazard_tuned_inputs` is a neural model that forecasts departures hour by hour. `earlier_gbm` and `earlier_population` are two forecasts re-implemented from an earlier analysis.
- **Rho.** The share of each car's energy that may stay undelivered. It is 0 in the main setting; 0.1 and 0.2 are sensitivity settings.
- **Main and held setting.** In the main setting each car is taken to charge uncontrolled from arrival at its nominal rate until the decision time; in the held setting no charging happens before the decision time. The held setting is run in full for the main model only. They are never pooled.
- **Values.** V_hind is the hindsight value (the ceiling). V_caus is the best value any real-time rule can keep, V_aff the best value of a simple (affine) rule, V_diag the best value of a rule that treats each hour apart, V_cert the value of the certificate. The plan states V_diag <= V_aff <= V_caus <= V_hind.
- **Statements of the plan.** T1: the exact ceiling formula. T2: the certificate that a box can be kept under the forecast sets; it is a proof only on the event E' (every forecast bound is on the safe side). T3: hourly re-solving is the best real-time rule. T4: equalities between the values, a hand proof for the hourly product, tested for the block product. T5: each later rung never fails before the one before it. T6: the guarantee above. T9: the ledger of section 7. T10: the drift guard of section 10.
- **"No figure" and "no interval".** A cell that reads "no figure: ..." gives the reason a figure is missing, never a zero. "no interval" says the run holds no interval for that cell. A unit with no figure adds 0 to a sum and counts as not passing.

## 3. Headline offer (O1)

**Failures and conditions first.**
- The main model's headline has no interval in the run, because its interval step failed (section 1).
- Three of the nine units offer anything. Six units calibrate to scale 0 and offer 0: `OSL_T` and the five Korean units (`o4-points.md`, lines 7, 9 to 13).
- The figure rests on one site. Nearly all of the main model's 130.153 kWh per counting test day comes from the Dutch office car park in its two periods (51.0323 and 78.4819 in `o1-units.md`, lines 5 and 6).
- The cells for the offer per car per day, the share of the hindsight ceiling and the share of the best-causal ceiling read "no figure" for the sum, because a unit with an offer of 0 or a missing share makes the ratio undefined. At four Korean units the share is missing for the reason in section 1.
- At the one pre-set setting tried for each, the offer is 0 when each car's rate, room and needed energy are forecast instead of taken as known, when departures are seen only at the start of the next hour, and without hourly re-solving (the fixed plan) (rows below). These zeros are results of those settings, not proofs that nothing is possible.
- Without the driver-risk dial (`main-k0.5-none-empirical`, one-way re-solve, tier A) the offer is 0 at all nine units (read from the results file of the run made apart, `20261124T000606Z-intervals-lgbm_tail-apart`).
- The figure of about 36 to 37% below is of the hindsight ceiling of one site. It is not a claim about fleets in general.

**Headline and settings.** Table file: `o1-summary.md` (lines 5, 6, 9, 13, 14, 15). Columns shown: run, X (the sum over the nine units of each unit's offer per counting test day, kWh), the total offered kWh over the total counting test days, and D (the offer that keeps every request, kWh per counting test day).

| run | X: sum of the units' offers (kWh per counting test day) | total offered kWh over total counting test days | D: every-request offer (kWh per counting test day) |
|---|---|---|---|
| lgbm_tail (main model), one-way re-solve | 130.153 (no interval: failed: intervals-lgbm_tail (the run ended with exit status 1)) | 16.1075 (no interval: failed: intervals-lgbm_tail (the run ended with exit status 1)) | 0 |
| hazard_tuned_inputs, one-way re-solve [made apart at a later commit, after the run's output was seen; fills no slot of plan section 11] | 142.263 [79.4322, 164.523] | 17.2915 [9.4507, 20.2056] | 0 |
| lgbm_tail, fixed plan | 0 (no interval: failed: intervals-lgbm_tail (the run ended with exit status 1)) | 0 (no interval: failed: intervals-lgbm_tail (the run ended with exit status 1)) | |
| lgbm_tail, departures seen at the next hour (beside reading) | 0 (no interval: failed: intervals-lgbm_tail (the run ended with exit status 1)) | 0 (no interval: failed: intervals-lgbm_tail (the run ended with exit status 1)) | |
| lgbm_tail, forecast energy at the headline's own kappa, reading and dial | 0 (no interval: failed: intervals-lgbm_tail-forecast-energy (the run ended with exit status 1)) | 0 (no interval: failed: intervals-lgbm_tail-forecast-energy (the run ended with exit status 1)) | |
| lgbm_tail, held setting | 160.776 (no interval: failed: intervals-lgbm_tail-held (the run ended with exit status 1)) | 19.7226 (no interval: failed: intervals-lgbm_tail-held (the run ended with exit status 1)) | |

In the same file the columns for the offer per car per day, the share of the hindsight ceiling and the share of the best-causal ceiling read "no figure: a unit's offer or per-car is 0 or missing", "no figure: a unit's offer or share of the ceiling is 0 or missing" and "no figure: a unit's offer or share of the best-causal ceiling is 0 or missing" for the main model's row (line 5). The neural model's row (line 6) gives 1.81333 [1.0044, 2.0941], 0.350491 [0.196485, 0.400057] and 0.404807 [0.22743, 0.461801] there. Each unit of the neural row uses its own pre-set dial (0.02 or 0.03).

**Interval made apart for the main model's headline** (read from the run's results file; not in a table file). Computed after held-out output had been seen, from the run's stored outputs at a later commit in which the interval step's code differs; it fills no slot of the plan and is not a pass. 95% bootstrap over days (repeated random redraws of the days; 2,000 resamples, seed 20261122; in each resample a unit's calibration and test days are redrawn together and the scale is recomputed): 130.153 kWh has [110.105, 140.741] and 16.108 kWh per counting test day has [13.819, 17.299]. The two forms belong together. The plan's pass mark for the headline is not met: X has no interval in the run and no unit passes the check of section 6 under reading A (none is judged; reading B is in section 6); the made-apart interval is not used for this mark.

**Per unit.** Table file: `o1-units.md` (lines 5 to 13, main model, tier A, configuration `main-k0.5-d0.02-empirical`). Every cell of the offer, offer per car, share-of-ceiling and best-causal columns in that file also reads "(no interval: day-intervals.json holds no interval for this cell (no such entry, or none written))"; the note is left out of the table below for width. An empty cell is empty in the file.

| unit | offer (kWh per counting test day) | offer per car per day (kWh) | share of the hindsight ceiling | share of the best-causal ceiling | counting test days | offering test days | failing test days | scale |
|---|---|---|---|---|---|---|---|---|
| asr-before-march-2023 | 51.0323 | 4.23383 | 0.359085 | 0.382298 | 187 | 186 | 16 | 0.889844 |
| asr-from-march-2023 | 78.4819 | 4.14794 | 0.373754 | 0.403801 | 290 | 289 | 43 | 0.882812 |
| OSL_T | 0 | 0 | 0 | 0 | 388 | 94 | | 0 |
| TRO_R | 0.638434 | 0.0529283 | 0.0463223 | 0.108227 | 241 | 121 | 52 | 0.672656 |
| korea-home | 0 | 0 | | 0 | 259 | 26 | | 0 |
| korea-lodging | 0 | 0 | | 0 | 263 | 260 | | 0 |
| korea-work | 0 | 0 | | 0 | 197 | 1 | | 0 |
| korea-public-parking | 0 | 0 | | 0 | 159 | 0 | | 0 |
| korea-visits | 0 | 0 | 0 | 0 | 31 | 0 | | 0 |

At the Dutch office car park the calibrated offer is about 36 to 37% of its hindsight ceiling (0.359085 and 0.373754). At the other units the largest share is 0.0463223 (`TRO_R`); at four Korean units the share is missing and at two (`OSL_T`, `korea-visits`) it is 0, and six units offer nothing.

## 4. Comparison methods and other forecasts (O1, O10)

**Failures and conditions first.**
- None of these figures has an interval (the plan gives none for them).
- The comparison step calibrates every method on a coarser grid of 10 fixed scales, without bisection, so its own re-solve (117.569) is not the headline (130.153) and the two are not mixed. The headline searches for the scale by bisection (repeatedly halving the range that holds the largest safe scale) to a precision of 0.001.
- The comparison step's own re-solve (117.569) is above the two standard controllers the plan names (ASA 43.8749 and LLF 102.375) but not above every method: the sizing rule ALSO-X gives 117.949, slightly above that re-solve in the same step. "Beats every baseline" is not supported.
- A change in per cent against the fixed-plan models (three simpler departure forecasts: Kaplan-Meier, pooled LightGBM and Cox, each a standard statistical or machine-learning method) has no figure, because their offer is 0.
- The neural model's headline (142.263) uses a dial of 0.03 at workplaces where the main model uses 0.02. With both at 0.02 the neural model gives 104.83 kWh per counting test day (read from the run's results file, not from a table file). An advantage of the neural model over the main model is not established: its made-apart interval is wider than the main model's and contains it.

**Comparison methods.** Table file: `o1-comparison.md` (lines 5 to 15). The methods are given the same forecasts, requests, calibration and loss. ASA is the adaptive scheduling algorithm (Lee et al., 2021) as a linear programme and `asa-es` adds its equal-sharing term. LLF is least-laxity-first (charge the car with the least slack first); `smoothed-llf` smooths the laxity; `edf` charges in order of believed departure; `max-entropy-feedback` is built as LLF guarded by the feasibility of the believed future, so its figures equal LLF's except at one unit with true departures; `diagonal-split` and `fixed-plan` do not re-solve; `also-x` and `cvar` are two sizing rules that allow a small chance of failure (chance-constrained) and replace the slot loss's scale.

| method | sum of the units' offers (kWh per counting test day) | units with a figure |
|---|---|---|
| re-solve (the comparison run's own) | 117.569 (no interval) | 9 of 9 |
| fixed-plan | 0 (no interval) | 9 of 9 |
| edf | 0 (no interval) | 9 of 9 |
| LLF (B) | 102.375 (no interval) | 9 of 9 |
| smoothed-llf | 93.4848 (no interval) | 9 of 9 |
| ASA (A) | 43.8749 (no interval) | 9 of 9 |
| asa-es | 70.5448 (no interval) | 9 of 9 |
| diagonal-split | 0 (no interval) | 9 of 9 |
| max-entropy-feedback | 102.375 (no interval) | 9 of 9 |
| also-x | 117.949 (no interval) | 9 of 9 |
| cvar | 76.4697 (no interval) | 9 of 9 |

**Other forecasts through the full method.** Table file: `o10-model-comparison.md` (lines 5 to 14; `o10-model-comparison.png` is its figure). Each model against each baseline on the same rung on both sides; the change reads as the model's X over the baseline's X.

| model | against | rung | model's X (kWh per counting test day) | baseline's X (kWh per counting test day) | change of the offer (%) |
|---|---|---|---|---|---|
| lgbm_tail | earlier_gbm | one-way | 130.153 | 94.4326 | 37.826 |
| lgbm_tail | earlier_population | one-way | 130.153 | 21.4532 | 506.683 |
| lgbm_tail | Kaplan-Meier | fixed | 0 | 0 | no figure: the baseline's offer is 0 |
| lgbm_tail | pooled LightGBM | fixed | 0 | 0 | no figure: the baseline's offer is 0 |
| lgbm_tail | Cox | fixed | 0 | 0 | no figure: the baseline's offer is 0 |
| hazard_tuned_inputs | earlier_gbm | one-way | 142.263 | 94.4326 | 50.6501 |
| hazard_tuned_inputs | earlier_population | one-way | 142.263 | 21.4532 | 563.132 |
| hazard_tuned_inputs | Kaplan-Meier | fixed | 0 | 0 | no figure: the baseline's offer is 0 |
| hazard_tuned_inputs | pooled LightGBM | fixed | 0 | 0 | no figure: the baseline's offer is 0 |
| hazard_tuned_inputs | Cox | fixed | 0 | 0 | no figure: the baseline's offer is 0 |

The earlier forecasts' rows are in `o1-summary.md` (lines 7 and 8): `earlier_gbm` 94.4326 and `earlier_population` 21.4532 with no interval; the fixed-plan rows for Kaplan-Meier, pooled LightGBM and Cox (lines 10 to 12) read 0.

## 5. Tolerated shortfall rho 0.1 and 0.2 (information only)

**Failures and conditions first.** These rows are information only: the plan's pass mark for the sensitivity settings is not passed after a run whose first mark fails. Their intervals are the run's own interval steps (which passed), not made apart. Each row is the main model's run at that rho, at its own level and dials, and is never pooled with another rho. At rho 0.1 and 0.2 the sums below are above the main setting's; a larger rho is a different promise to drivers, since each car's protected energy is smaller.

Table file: `o1-rho.md` (lines 5 to 7). Columns shown: X, the total offered kWh over the total counting test days, and the share of the hindsight ceiling.

| run | X: sum of the units' offers (kWh per counting test day) | total offered kWh over total counting test days | Z: share of the hindsight ceiling |
|---|---|---|---|
| lgbm_tail (main model), one-way re-solve | 130.153 (no interval: failed: intervals-lgbm_tail (the run ended with exit status 1)) | 16.1075 (no interval: failed: intervals-lgbm_tail (the run ended with exit status 1)) | no figure: a unit's offer or share of the ceiling is 0 or missing |
| lgbm_tail, rho 0.1 (WP15 sensitivity) [information only] | 149.607 [135.234, 163.44] | 18.5856 [16.8967, 20.2197] | 0.334265 [0.307399, 0.360281] |
| lgbm_tail, rho 0.2 (WP15 sensitivity) [information only] | 176.314 [158.539, 192.28] | 21.738 [19.7115, 23.6085] | 0.363243 [0.330054, 0.390013] |

## 6. The guarantee (O4)

**Failures first.** The check of the guarantee judges no unit and passes none, for the main model or for the neural model (0 of 9 pass; 0 judged under reading A). The check asks, per unit, that the 95% interval of availability reaches 90% or more and the interval of the shortfall reaches 5% or less. Six units are at scale 0 (no offer, no loss); reading A counts them as not judged, a reading fixed after the run began, and reading B counts them as passing at zero loss (table below). The other three have no interval, because their interval steps failed. At `TRO_R` the point value of the slot loss is above its limit. The availability and shortfall intervals are not produced (every row of `o4-availability-shortfall.md` reads "no figure: failed: intervals-..."). The guarantee holds on average over the random choice of calibration days, so one unit above a limit is not a counterexample to it; it is a missed pass mark.

Made apart afterwards (run `20261124T000606Z-intervals-lgbm_tail-apart`, after held-out output was seen; fills no slot of the plan; information only; read from that run's results file), availability and shortfall at the headline configuration are 0.946 [0.914, 0.971] and 0.025 [0.019, 0.031] at `asr-before-march-2023`, 0.901 [0.871, 0.929] and 0.028 [0.024, 0.032] at `asr-from-march-2023`, and 0.854 [0.817, 0.891] and 0.042 [0.025, 0.065] at `TRO_R`.

**The pass mark read two ways.** Table file: `o4-pass-both-ways.md` (lines 5 and 6). Reading A leaves a unit at scale 0 outside the count ("not judged"); it was fixed after the run began and before any result was read. Reading B judges it and counts it as passing at zero loss, since a day that does not offer has no loss. The plan covers a unit that abstains as a whole, and no held-out unit does, so both readings are printed. A unit with no figure counts as not passing in both; a unit with no interval for another reason is not judged in both.

| model | units | A: units passing | A: units judged | units at scale 0 (not judged in A) | units with no interval (not judged) | units not passing, judged | B: units passing (scale 0 counted as passing) | B: units judged |
|---|---|---|---|---|---|---|---|---|
| lgbm_tail | 9 | 0 | 0 | 6 | 3 | 0 | 6 | 6 |
| hazard_tuned_inputs | 9 | 0 | 0 | 6 | 3 | 0 | 6 | 6 |

**Point values per unit.** Table file: `o4-points.md` (lines 5 to 22). They are the run's own point values at each unit's headline configuration (tier A, one-way rung), with no interval and no verdict. Rows for `lgbm_tail` at `asr-before-march-2023`, `asr-from-march-2023` and `TRO_R`, and for `hazard_tuned_inputs` at the same units:

| model | unit | offer (kWh per counting test day) | mean slot loss (offering test days) | slot loss limit | mean driver loss (offering test days) | driver loss limit | scale | offering test days |
|---|---|---|---|---|---|---|---|---|
| lgbm_tail | asr-before-march-2023 | 51.0323 | 0.0537634 | 0.1 | 0.0245256 | 0.05 | 0.889844 | 186 |
| lgbm_tail | asr-from-march-2023 | 78.4819 | 0.0991926 | 0.1 | 0.0279562 | 0.05 | 0.882812 | 289 |
| lgbm_tail | TRO_R | 0.638434 | 0.145661 | 0.1 | 0.0423923 | 0.05 | 0.672656 | 121 |
| hazard_tuned_inputs | asr-before-march-2023 | 61.9979 | 0.0648649 | 0.1 | 0.0308867 | 0.05 | 0.889062 | 185 |
| hazard_tuned_inputs | asr-from-march-2023 | 79.6906 | 0.0542099 | 0.1 | 0.0314242 | 0.05 | 0.745313 | 289 |
| hazard_tuned_inputs | TRO_R | 0.574384 | 0.136719 | 0.1 | 0.0359046 | 0.05 | 0.640625 | 128 |

For every other unit (both models) the offer is 0, the scale is 0 and the loss cells read "not judged: scale 0 (no offer, so no loss)". At `TRO_R` the main model's mean slot loss is 0.145661, which is 14.57% of service hours lost against the 10% limit (read from the run's results file for the percentage; the table gives 0.145661).

**Tier B at two points.** Table file: `o4-tier-b-two-points.md` (lines 5 to 22 for the main model). The full curve of the offer against the driver loss over the dial was not computed for the held-out units; the file holds two points, no dial (`none`) and the chosen dial. Rows for the main model at the three units with an offer:

| unit | dial (driver-risk level) | tier B offer (kWh per counting test day) | tier B mean driver loss (offering test days) |
|---|---|---|---|
| asr-before-march-2023 | 0.02 | 51.0323 | 0.0245256 |
| asr-before-march-2023 | none | 133.383 | 0.113698 |
| asr-from-march-2023 | 0.02 | 78.4819 | 0.0279562 |
| asr-from-march-2023 | none | 192.38 | 0.13331 |
| TRO_R | 0.02 | 0.638434 | 0.0423923 |
| TRO_R | none | 0 | |

At the two Dutch units, without the dial the offers are larger, but the driver losses (0.113698 and 0.13331) are above the 0.05 limit, so those offers are not driver-protected. Every unit not in the table is 0 at both points. The neural model's rows are in the file (lines 23 to 49).

**Availability and shortfall intervals.** Table file: `o4-availability-shortfall.md` (every cell reads "no figure: failed: ..."), with `o4-availability-shortfall-lgbm_tail.png` and `o4-availability-shortfall-hazard_tuned_inputs.png` (one per model).

**Where the certificate is a proof (event E').** Table file: `eprime-days.md` (lines 5 to 13 for the main model; the neural model's rows follow). The certificate of T2 is a proof on the event E', in which every steered car's forecast bounds are on the safe side (steered cars are the cars of the day with a forecast and a non-empty forecast set, whose charging the offer controls). Columns shown: counting test days, certified test days (days on which the certified box has a value above 0), days in E', and steered cars on certified test days and in their own event.

| unit | counting test days | certified test days | days in E' | share of certified test days in E' | steered cars on certified test days | steered cars in E'_i | share of steered cars in E'_i |
|---|---|---|---|---|---|---|---|
| asr-before-march-2023 | 187 | 186 | 1 | 0.00537634 | 2243 | 747 | 0.333036 |
| asr-from-march-2023 | 290 | 289 | 0 | 0 | 5462 | 1542 | 0.282314 |
| OSL_T | 388 | 94 | 0 | 0 | 936 | 344 | 0.367521 |
| TRO_R | 241 | 121 | 1 | 0.00826446 | 1797 | 732 | 0.407346 |
| korea-home | 259 | 26 | 0 | 0 | 213 | 39 | 0.183099 |
| korea-lodging | 263 | 260 | 0 | 0 | 2533 | 182 | 0.0718516 |
| korea-work | 197 | 1 | 0 | 0 | 3 | 0 | 0 |
| korea-public-parking | 159 | 0 | 0 | no figure: certified_test_days is 0, so the run writes no share | 0 | 0 | no figure: steered_cars_on_certified_test_days is 0, so the run writes no share |
| korea-visits | 31 | 0 | 0 | no figure: certified_test_days is 0, so the run writes no share | 0 | 0 | no figure: steered_cars_on_certified_test_days is 0, so the run writes no share |

E' holds on 1 of 186 certified test days at `asr-before-march-2023` and on 0 of 289 at `asr-from-march-2023`. At these units the offers' loss limits therefore rest on the calibration, not on the certificate. (This table is of held-out units and is shown here, not under the development figures of section 14.)

## 7. Where the offer is lost (O6)

**Conditions first.** The ledger (statement T9) splits the gap between the hindsight ceiling and the offer into terms that add to the ceiling, each as a share of it, on a seeded sample of up to 100 test days per unit (31 at `korea-visits`). The terms are: the offer itself; R, the price of real-time decisions (hindsight value minus best real-time value); A, the price of affine rules (best real-time value minus best affine value); U, the price of unknown cars (affine value minus the certificate's value); K, the price of calibration (certificate minus offer). F, the format loss (the ceiling minus the largest box in the simpler set of LP (24), section 9), lies outside the sum. U and K are signed: a negative U means the certificate, computed from forecast departures, is above what the true cars allow; calibration then scales it down, and where the offer is scaled to 0 the whole certified value falls in K, which can then exceed 1. This is the case at `korea-lodging` (U -0.757607, K 1.39754, offer 0, terms still summing to 1). The small negative values of A (of the order of 1e-17) are rounding. For the block product, calibrated offers were cut, so the offer, F, U and K are not computed for it.

Table file: `o6-ledger.md` (lines 5, 7, 9, 11, 13, 15, 17, 19, 21 for the hourly product), tier A, one-way.

| unit | days | offer | R | A | U | K | F (format loss) |
|---|---|---|---|---|---|---|---|
| asr-before-march-2023 | 100 | 0.370027 | 0.0571389 | -1.80703e-17 | 0.527028 | 0.0458067 | 0.147579 |
| asr-from-march-2023 | 100 | 0.368889 | 0.0767553 | -6.59356e-19 | 0.505388 | 0.0489676 | 0.157444 |
| OSL_T | 100 | 0 | 0.432614 | 5.31159e-17 | 0.507363 | 0.0600228 | 0.434838 |
| TRO_R | 100 | 0.0476152 | 0.58146 | 2.10025e-17 | 0.347753 | 0.0231716 | 0.60229 |
| korea-home | 100 | 0 | 0.777095 | 1.07436e-16 | 0.213463 | 0.00944159 | 0.799844 |
| korea-lodging | 100 | 0 | 0.360071 | 2.53761e-18 | -0.757607 | 1.39754 | 0.548546 |
| korea-work | 100 | 0 | 0.8306 | -1.12811e-17 | 0.1694 | 0 | 0.930021 |
| korea-public-parking | 100 | 0 | 0.837329 | 1.07225e-17 | 0.162671 | 0 | 0.85995 |
| korea-visits | 31 | 0 | 0.76304 | 0 | 0.23696 | 0 | 0.76304 |

The price of the block (hourly minus block hindsight value, as a share of the ceiling) is in the block rows of the same file (lines 6, 8, 10, ... 22): 9.68989e-18, 0.000267049, 0.250002, 0.236659, 0.506035, 0.121566, 0.72498, 0.745823 and 0.52608, in the unit order of the table. The block rows' R, A and V_aff - V_diag are in the file as well.

## 8. Exact-ceiling check on held-out days (O2)

**Conditions first.** The check is the statement that, for the block product, the best real-time value equals the hindsight value on every day that meets the statement's conditions (one-way charging, known cars, no car leaving before the last service hour, and every car able to take all its protected energy at full rate after the window and before it leaves). Few days meet them (87 of 2,846 counting days in the main setting, 31 in the held setting), and four Korean units have no such day, so "no day to check" is not a pass. `korea-work` has one such day in each setting and `OSL_T` one in the held setting, so their "passed" rests on one day. The statement is tested for the block product and not proved. The two settings are never pooled. The check passed in both settings as a step of a run whose own pass mark is not met.

Table files: `o2-block-product.md` (lines 5 to 22) and `o2-day-counts.md` (lines 5 to 24). In the first table X is the mean of V_caus/V_hind over every day of the unit in per cent, Q the median, P the ratio on days with a departure inside the window, and K the number of days on which the affine value equals the real-time value.

Main setting:

| unit | M days | X: mean of V_caus/V_hind (%) | Q: median of V_caus/V_hind (%) | P: on days with an in-window departure (%) | K: days with V_aff = V_caus | pass |
|---|---|---|---|---|---|---|
| asr-before-march-2023 | 287 | 93.3706 | 96.4188 | 91.9432 | 287 | passed |
| asr-from-march-2023 | 390 | 90.6092 | 93.0085 | 89.8347 | 390 | passed |
| OSL_T | 488 | 85.7466 | 100 | 85.5966 | 488 | passed |
| TRO_R | 341 | 53.1188 | 54.1912 | 50.1706 | 341 | passed |
| korea-home | 359 | 49.9429 | 48.5703 | 49.9429 | 359 | no day to check |
| korea-lodging | 363 | 64.6747 | 75 | 64.6747 | 363 | no day to check |
| korea-work | 297 | 48.4958 | 50 | 48.3705 | 297 | passed |
| korea-public-parking | 259 | 46.7778 | 50 | 46.7778 | 259 | no day to check |
| korea-visits | 62 | 50 | 50 | 50 | 62 | no day to check |

Held setting:

| unit | M days | X: mean of V_caus/V_hind (%) | Q: median of V_caus/V_hind (%) | P: on days with an in-window departure (%) | K: days with V_aff = V_caus | pass |
|---|---|---|---|---|---|---|
| asr-before-march-2023 | 287 | 93.4168 | 95.3667 | 91.4673 | 287 | passed |
| asr-from-march-2023 | 390 | 91.1395 | 92.843 | 90.6251 | 390 | passed |
| OSL_T | 488 | 55.7895 | 55.0989 | 54.8969 | 488 | passed |
| TRO_R | 341 | 54.352 | 52.4461 | 49.4732 | 341 | passed |
| korea-home | 359 | 58.2338 | 62.5 | 58.2338 | 359 | no day to check |
| korea-lodging | 363 | 87.1921 | 100 | 87.1921 | 363 | no day to check |
| korea-work | 297 | 59.5261 | 82.2065 | 57.7056 | 293 | passed |
| korea-public-parking | 259 | 51.4062 | 53.4759 | 51.4062 | 259 | no day to check |
| korea-visits | 62 | 50 | 50 | 50 | 62 | no day to check |

Days meeting the statement's conditions (`o2-day-counts.md`): main setting, `asr-before-march-2023` 36, `asr-from-march-2023` 6, `OSL_T` 12, `TRO_R` 32, the four Korean units 0 except `korea-work` 1, and 87 over all units named (of 2846 counting days); held setting 20, 2, 1, 7, `korea-work` 1, the rest 0, and 31 over all units named. On every one of them the column "of them with V_hind different from V_caus" reads 0, and none was missing a value.

Over every day of a unit, not only the days meeting the conditions, X is 53.1188 at `TRO_R` and between 46.7778 and 64.6747 at the five Korean units in the main setting; the statement covers only the days counted above.

## 9. Published inner approximations against the ceiling (O3)

**Conditions first.** O3 is measured on true sessions with no forecast, on sampled days, so it says how much of the ceiling a published simpler set reaches, not what a forecast offer achieves. None of these methods could run in advance of the day. LP (24) is a linear programme (an optimisation problem) of Al Taha et al. (2025) that gives a simpler set of charging plans, the general split is their per-car split, and LP (27), another linear programme of the same paper, has many optimal solutions whose sets hold boxes of very different sizes, so its largest and smallest box are estimated from inside by a stated search, never claimed exact (the table gives the largest box found, a proved upper bound beside it, and the smallest found). Sample: 200 days per unit for LP (24), the general split and the ceilings; 100 for LP (27), whose run time is capped at about two hours per unit; 62 at `korea-visits`. All asked LP (27) days were reached within the cap (`o3-days.md`, lines 5 to 13: 0 days not reached). The percentages below are on the days on which every method has a value (the "same days"); at several units fewer than 100 days have every value (97, 97, 87 and 90 for the hourly product at `OSL_T`, `TRO_R`, `korea-home` and `korea-lodging`), because a method has no value on the other days (a missing value is never read as 0; the column "missing, by outcome" of `o3-ceiling-shares.md` counts them). The combined figures at the end were set after the results were seen, by the plan's rule for O1 carried over (sum over units of each unit's mean value over the sum of each unit's mean ceiling); the plan states no pooling rule for O3, so they fill no slot and replace no per-unit value, and a unit with a large ceiling weighs more.

Table files: `o3-same-days.md` (per unit, every method on the same days), `o3-ceiling-shares.md` and `o3-ceiling-shares.png` (per unit, each method on its own days; not to be compared across methods), `o3-days.md`, `o3-combined-units.md` (the inputs of the combined table) and `o3-combined.md`. Hourly product, share of the ceiling as a ratio of means in per cent, from `o3-same-days.md` (the block product's rows are in the same file):

| unit | days with every value | lp24 | general_split | lp27_smallest_found | lp27_largest_found | lp27_largest_bound |
|---|---|---|---|---|---|---|
| asr-before-march-2023 | 100 | 85.449 | 97.121 | 78.7473 | 91.7018 | 94.8444 |
| asr-from-march-2023 | 100 | 84.641 | 96.422 | 80.8802 | 92.0581 | 94.979 |
| OSL_T | 97 | 62.3023 | 62.8937 | 79.7392 | 96.9089 | 99.586 |
| TRO_R | 97 | 39.2433 | 42.5718 | 51.3037 | 91.4269 | 98.2197 |
| korea-home | 87 | 23.0413 | 24.993 | 30.7396 | 91.3116 | 100 |
| korea-lodging | 90 | 46.7202 | 64.867 | 33.7011 | 97.9095 | 99.1829 |
| korea-work | 100 | 4.60204 | 7.94209 | 26.555 | 96.8474 | 100 |
| korea-public-parking | 100 | 14.4177 | 16.2406 | 22.7575 | 86.5467 | 100 |
| korea-visits | 62 | 14.1246 | 14.1246 | 69.739 | 100 | 100 |

The same file also holds `general_split_no_hour0_gain` (the general split without the gain from the decision hour's free power), the mean and the median of the per-day shares, and the block product.

Combined over the nine units, `o3-combined.md` (lines 5 to 16):

| product | value | units summed | days summed | sum of unit mean values (kWh) | sum of unit mean ceilings (kWh) | combined share of the ceiling (%) |
|---|---|---|---|---|---|---|
| hourly | lp24 | 9 | 833 | 314.166 | 401.386 | 78.2702 |
| hourly | general_split | 9 | 833 | 358.136 | 401.386 | 89.2249 |
| hourly | general_split_no_hour0_gain | 9 | 833 | 346.496 | 401.386 | 86.3249 |
| hourly | lp27_smallest_found | 9 | 833 | 300.414 | 401.386 | 74.8442 |
| hourly | lp27_largest_found | 9 | 833 | 369.678 | 401.386 | 92.1004 |
| hourly | lp27_largest_bound | 9 | 833 | 383.308 | 401.386 | 95.4959 |
| block | lp24 | 9 | 840 | 308.968 | 383.093 | 80.6507 |
| block | general_split | 9 | 840 | 358.281 | 383.093 | 93.5231 |
| block | general_split_no_hour0_gain | 9 | 840 | 346.644 | 383.093 | 90.4855 |
| block | lp27_smallest_found | 9 | 840 | 278.295 | 383.093 | 72.6441 |
| block | lp27_largest_found | 9 | 840 | 331.486 | 383.093 | 86.5287 |
| block | lp27_largest_bound | 9 | 840 | 361.853 | 383.093 | 94.4554 |

## 10. Properties that must hold (O5) and the drift guard

**Failures first.** No property check failed (every "breaks" cell below is 0). The drift guard's end value is a failure of a kind: its mean slot loss at the end is above its 10% level, which its theorem does not promise at that point. The drift guard rows are information only (the sensitivity work package's mark is not passed), and they cover one Norwegian unit only.

**Properties.** Table file: `o5-properties.md` (lines 5 to 14). A break is a bug and would be reported as one.

| property | checked | breaks | largest residual or detail | holds |
|---|---|---|---|---|
| T9 identity: offer + R + A + U + K = ceiling, every day and every mean | 103104 | 0 | 5.68434e-14 | yes |
| the run's offers reproduced from the day rows | | 0 | | yes |
| the run's ceiling sums reproduced | | 0 | | yes |
| the sampled days are the run's days | | 0 | | yes |
| R + A, R, A and the other terms not below 0 | | 0 | | yes |
| the one-way ceilings of O8 reproduced | | 0 | | yes |
| offer order: fixed <= one-way <= two-way on every test day | | 0 | | yes |
| sampled days whose sets could not be built | | 0 | | yes |
| the ledger's own verdict | | | | yes |
| T5 (a) and (b): no rung fails before its predecessor; each car ends with at least the predecessor's energy | 367560 | 0 | breaks (a): 0; breaks (b): 0 | yes |

**Offer order.** `o5-offer-ratio.md` (lines 5 to 13): at all nine units both shares are 1 (fixed <= one-way and one-way <= two-way on every counting test day) with 0 days breaking.

**Drift guard.** The guard moves the scale day by day in date order by a step size (statement T10; its average loss is bounded on any sequence of days). Step sizes were fixed from the training units (`o5-step-sizes.md`, lines 5 to 8):

| tier | site type | step size |
|---|---|---|
| A | other | 3 |
| A | workplace | 0.5 |
| B | other | 50 |
| B | workplace | 0.5 |

Chronological run at `OSL_T`, the one unit the plan names, from the table file `o5-drift.md` (lines 5 and 6): the first 50 counting days calibrate and the guard runs on the rest. Columns shown:

| unit | tier | start | step size | steps taken | slot loss, mean at the end | slot loss, T10 bound at the end | slot loss level | driver loss, mean at the end | driver loss level | steps over the bound | mean offer (kWh per counting day) | driver loss, T10 bound at the end | information only |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OSL_T | A | zero | 3 | 106 | 0.102594 | 0.108491 | 0.1 | 0.0459312 | 0.05 | 0 | 0.0413228 | 0.0589623 | yes |
| OSL_T | B | zero | 50 | 106 | 0.102594 | 0.108491 | 0.1 | 0.0502132 | not held down (reported) | 0 | 0.0651303 | not held down (reported) | yes |

The mean slot loss ends at 0.102594, above the 0.1 level and under the theorem's bound of 0.108491; no step has the running mean above the bound, which checks the code and not the data. The offer is small (0.0413228 kWh per counting day in tier A).

## 11. Worst-case request paths (O7)

**Failures first.** The offers elsewhere in this page are scored on 8 fixed request paths a day. A search of the extreme request paths inside each offered box (every corner path, with and without true departures) finds breaks. For the project's own re-solve with the learned forecast at least one breaking path is found on 27 of 50 searched days at `asr-before-march-2023`, 40 of 50 at `asr-from-march-2023` and 34 of 50 at `TRO_R` (101 of 150 searched days in all, read from the run's results file; with true departures 0 of 250; the two counts are on different days and scales, so they are not a matched comparison). Every share but the fixed plan's is a lower bound, because for any other method a request path inside the box can be worse than every corner. Methods that offer on a unit are searched; a unit with no offer reads "no offer". Per unit only; nothing is pooled.

Table file: `o7-corner-units.md` (lines 5 to 13 and 104 to 112 for the re-solve; the other ten methods are in the rest of the file, 198 rows in all). Columns shown for the re-solve:

| variant | method | unit | scale | searched days | days with a result | days breaking | share of days breaking (%) | status |
|---|---|---|---|---|---|---|---|---|
| without oracle departures | re-solve | asr-before-march-2023 | 0.8 | 50 | 50 | 27 | 54 | searched |
| without oracle departures | re-solve | asr-from-march-2023 | 0.8 | 50 | 50 | 40 | 80 | searched |
| without oracle departures | re-solve | OSL_T | 0 | 50 | | | | no offer |
| without oracle departures | re-solve | TRO_R | 0.6 | 50 | 50 | 34 | 68 | searched |
| without oracle departures | re-solve | korea-home | 0 | 26 | | | | no offer |
| without oracle departures | re-solve | korea-lodging | 0 | 50 | | | | no offer |
| without oracle departures | re-solve | korea-work | 0 | 1 | | | | no offer |
| without oracle departures | re-solve | korea-public-parking | 0 | 0 | | | | no offer |
| without oracle departures | re-solve | korea-visits | 0 | 0 | | | | no offer |
| with oracle departures | re-solve | asr-before-march-2023 | 1 | 50 | 50 | 0 | 0 | searched |
| with oracle departures | re-solve | asr-from-march-2023 | 1 | 50 | 50 | 0 | 0 | searched |
| with oracle departures | re-solve | OSL_T | 0 | 50 | | | | no offer |
| with oracle departures | re-solve | TRO_R | 1 | 50 | 50 | 0 | 0 | searched |
| with oracle departures | re-solve | korea-home | 1 | 50 | 50 | 0 | 0 | searched |
| with oracle departures | re-solve | korea-lodging | 1 | 50 | 50 | 0 | 0 | searched |
| with oracle departures | re-solve | korea-work | 0 | 4 | | | | no offer |
| with oracle departures | re-solve | korea-public-parking | 0 | 4 | | | | no offer |
| with oracle departures | re-solve | korea-visits | 0 | 1 | | | | no offer |

Without true departures the comparison methods break on a similar share of their days where they offer: LLF 50% and 70% and ASA 56% and 72% at the two Dutch units (lines 32, 33, 50, 51). With true departures, where the re-solve breaks on no searched day, LLF and ASA break on more days than the re-solve: LLF 10%, 32% and 40% at the two Dutch units and `TRO_R` (lines 131, 132, 134) and ASA 22%, 20%, 70% and 82% at the two Dutch units, `TRO_R` and `korea-home` (lines 149 to 153). The fixed plan breaks on no searched day with true departures (lines 113 to 118: 0 at every unit that offers). Without true departures the fixed plan offers nothing at any unit.

## 12. Two-way charging (O8)

**Failures and conditions first.** Two-way charging adds nothing to the offer here: the two-way offer equals the one-way offer at the three units with an offer (ratio 1) and both are 0 at the other six, although the exact two-way ceiling is larger. The discharge in kWh per car-day (column D) was not run, so no figure is given. Two-way charging is measured at a 5 kWh budget per car only; budgets of 2 and 10 kWh are not measured.

Table file: `o8-two-way.md` (lines 5 to 13). Columns shown: X, the two-way offers over the one-way offers (an empty cell where there is no one-way offer); V, the exact two-way ceilings over the one-way ceilings; and the two-way offer as a share of the two-way ceiling.

| unit | test days | X: two-way offers over one-way offers | V: two-way ceilings over one-way ceilings (B = 5 kWh) | two-way offer as a share of the two-way ceiling | test days with a one-way ceiling of 0 |
|---|---|---|---|---|---|
| asr-before-march-2023 | 187 | 1 | 1.97357 | 0.181947 | 1 |
| asr-from-march-2023 | 290 | 1 | 1.92044 | 0.194618 | 0 |
| OSL_T | 388 | | 17.2716 | 0 | 229 |
| TRO_R | 241 | 1 | 6.36697 | 0.0072754 | 60 |
| korea-home | 259 | | 1.7254 | 0 | 16 |
| korea-lodging | 263 | | 1.68902 | 0 | 22 |
| korea-work | 197 | | 1.55382 | 0 | 92 |
| korea-public-parking | 159 | | 1.45905 | 0 | 61 |
| korea-visits | 31 | | 2.76335 | 0 | 22 |

## 13. Cold start (O9)

**Failures first.** With 0, 7, 14, 28 or 56 days of local history in time order, the main model offers nothing at any of the seven other units up to 56 days, nor at either Dutch unit up to 28 days. It offers at the two Dutch units at 56 days only. At the first of them the availability at 56 days (0.881159) is below the 90% target. Where the scale is 0 the file names one of two reasons: "too few offering calibration days" or "the bound fails at every grid scale". Pooled LightGBM (`lgbm`) is the reference model.

Table file: `o9-cold-start.md` (lines 5 to 9 and 10 to 14 for `asr-before-march-2023`, 15 to 19 and 20 to 24 for `asr-from-march-2023`; `o9-cold-start.png` is its figure). Rows with a scale above 0:

| model | unit | k (local days) | offering local days | scale | offer (kWh per counting scored day) | A_k: availability | Y_k: share of the hindsight ceiling | share of the best-causal ceiling | offering scored days | failing scored days |
|---|---|---|---|---|---|---|---|---|---|---|
| lgbm_tail | asr-before-march-2023 | 56 | 54 | 1 | 44.6442 | 0.881159 | 0.289428 | 0.30759 | 230 | 44 |
| lgbm | asr-before-march-2023 | 28 | 27 | 0.3 | 10.799 | 0.998551 | 0.0700095 | 0.0744027 | 230 | 1 |
| lgbm | asr-before-march-2023 | 56 | 54 | 1 | 42.387 | 0.895652 | 0.274795 | 0.292038 | 230 | 44 |
| lgbm_tail | asr-from-march-2023 | 56 | 56 | 0.6 | 39.351 | 0.95509 | 0.183966 | 0.199519 | 334 | 20 |
| lgbm | asr-from-march-2023 | 56 | 56 | 0.7 | 45.2559 | 0.938124 | 0.211571 | 0.229458 | 334 | 27 |

Every other row of the file (every other unit and every other k) has scale 0, offer 0 and a reason in the last column. For example `lgbm_tail` at `asr-before-march-2023` reads "too few offering calibration days" at k of 0, 7 and 14 and "the bound fails at every grid scale" at k of 28.

## 14. Development figures behind the settings

**Conditions first.** These are figures of development data (sites used to build the method and fix its settings), read from two finished development runs and copied as written. They are not figures of the held-out run, and none may be set beside a held-out figure. Every setting fixed before the plan was frozen was fixed after development data had been seen. One bears on the headline: the plan's first reading of the gate (a development go or no-go test of the offer, with the thresholds in the table below) used empirical departure quantiles and was not met, and the model read at the gate was then changed to `lgbm_tail`, which the plan names the main model; the change is recorded in the plan with its date.

**The checks on development days.** Table file: `dev-days-checks.md` (lines 5 to 94; run at commit 0f62fc089033b29b378aa2c3068fdc65f43282e0, exit 0, file hash in the table's last line). Setting of the run: 100 sampled days (seed 2026), level 0.1, energy bounds known, a car with no usable certificate left unsteered. It is not the setup of the headline forecast. Key rows:

| scope | quantity | figure |
|---|---|---|
| order check, every day | verdict | passed |
| order check, every day | days covered | 585 |
| T4 log, block product, every day | verdict | passed |
| T4 log, block product, every day | days meeting T4's hypotheses | 148 |
| T4 log, block product, every day | meets T4's hypotheses: days with the tree below hindsight | 0 |
| T4 log, block product, every day | other days, no in-window departure: days with the tree below hindsight | 86 |
| T4 log, block product, every day | other days, in-window departure: days with the tree below hindsight | 249 |
| T4 log, block product, every day | other days, no in-window departure: mean gap hindsight minus tree, kWh | 6.27059 |
| T4 log, block product, every day | other days, in-window departure: mean gap hindsight minus tree, kWh | 7.26539 |
| T4 log, block product, every day | days with the affine value below the tree (all three groups) | 0 |
| T2's check, sampled days | verdict | passed |
| T2's check, sampled days | sampled days certified | 98 |
| T2's check, sampled days | days in E' | 46 |
| T2's check, sampled days | share of certified days in E' | 0.469388 |
| T2's check, sampled days | forecast rows checked | 9275 |
| T2's check, sampled days | closed form against replay, mismatches | 0 |
| T2's check, sampled days | cars in E'_i checked | 490 |
| T2's check, sampled days | cars in E'_i breaking a true row | 0 |
| T2's check, sampled days | cars outside E'_i breaking a true row (counted, not judged) | 76 |
| T2's check, sampled days | corner paths with a request missed | 0 |
| two-way corner tests, sampled days | verdict | passed |
| two-way corner tests, sampled days | days on which both products' trees matched | 100 |

Read as the file gives them: where the block product's statement T4 does not apply, the best real-time value falls below the hindsight value on 86 of 98 days with no in-window departure and on 249 of 339 days with one; where it applies the gap is 0 on all 148 days.

**The dial choice.** Table file: `dev-dial-choice.md` (run of the main model at level 0.3, 200 seeded splits per unit, eight dials from 0 to 0.1, tiers A and B, one-way rung, commit 0f62fc089033b29b378aa2c3068fdc65f43282e0, exit 0). Offer in kWh per counting test day over the splits, tier A, at the dial that gave the largest mean offer, and the gate reading at dial 0.02 (the offer against the mean hindsight ceiling with the decision hour free); the file also holds the per-split spread, the losses and the best-causal ceiling. The gate reading is the file's own; its verdict is not in the table file and is not stated here. The plan's thresholds, in the file, are 0.1 for the offer's share of the ceiling and 1.5 for the oracle-departure offer over the offer.

| unit | counting days | dial with the largest mean offer (tier A) | its mean offer | gate reading at dial 0.02: mean offer over all splits | mean ceiling (decision hour free) | oracle offer over mean offer |
|---|---|---|---|---|---|---|
| norway-development-pooled | 242 | 0.02 | 2.2378 | 2.2378 | 11.5492 | 4.50557 |
| nrel-before-may-2019 | 9 | 0 | 0 | 0 | 22.9785 | no figure: not a number: the mean offer is 0 |
| nrel-from-may-2019 | 334 | 0.02 | 17.0777 | 17.0777 | 64.693 | 3.29557 |

(From `dev-dial-choice.md`, lines 12, 13, 33, 36, 37, 211, 214, 217, 374, 377, 378, 552, 555, 558, 715, 718, 719, 893, 896, 899.) At the development unit before May 2019 the largest mean offer over the dials is 0.

**E' on held-out units.** The table `eprime-days.md` is of held-out units and is shown in section 6.

## 15. Trials made after the held-out run

**These are not results of the plan.** Every trial below was made AFTER the held-out results were seen, on development units (the sites used to build the method and choose settings); the trials' guards refuse held-out unit names, but in trial B the option `--tables`, which points to a folder of tables, is not guarded, and in the mix, no-encoder and both energy-bound trials (E and I) the options `--tables` and `--truth-run` are reported in the output, not refused (so is the option `--first-run` of trial I, which names the run folder of trial E); in the local trial, whose options include these two, the results record the settings file's name and SHA-256 and the run matrix's, so a run with an option shows it. The list names the trials whose account states this. Every trial's figures come from one run, except trial B's, which come from two runs, one per level (0.05 and 0.1); trials A and C share one run. Four starts ended at once with exit status 2 and no figure (two of the command that ran trials A and C, one of the mix trial's and one of the no-encoder trial's); they are not counted as runs. No trial changes the main model, the plan, a pass mark or any held-out figure. No held-out figure appears in this part, and no trial figure is to be set beside, added to or subtracted from one of sections 1 to 14: the trials use other sites, other days and other splits. A trial that finds no gain tried one setting or a few; it does not show that no setting helps. A "split" is one seeded random draw of calibration and test days, and "200 seeded splits" are drawn from the same days, so they are not 200 independent samples. The trials' own summary files give the figures. Units are named by role: the development workplace unit from May 2019 (334 counting days), the pooled Norwegian development sites (242 counting days) and the development workplace unit before May 2019 (9 counting days).

Nine trials were made. Four tried to turn zero offers into offers (A to D), two tried more cautious energy bounds (E and I), and three tried the departure forecast itself (mix, local, and the neural model without its set encoder, the part of the neural model that applies attention over the cars of a site-day, so that a car's encoding depends on the other cars; without it, a car's encoding depends on its own inputs only). Figures are mean kWh per counting test day, tier A (driver-protected), unless the row says otherwise.

**A. Energy forecast at a more cautious departure level (level 0.2, dial 0.02, one-way re-solve).** One run (trials A and C share one command).

| unit | energy forecast instead of known | known energy, same level and dial |
|---|---|---|
| Norway pooled | abstains: 15 of 100 calibration days offer, the driver bound needs 19 | scale 0.333, 0.737 (38 of 100 calibration days offer) |
| workplace before May 2019 | abstains (1 of 4 calibration days offer) | abstains (4 calibration days) |
| workplace from May 2019 | no scale is safe: 82 of 100 calibration days offer, the bound fails at every scale tried | scale 0.739, 15.258 |

Reading: forecasting the cars' energy keeps the offer at 0 at every unit at this level, where known energy offers at two. One setting of the energy forecast was tried.

**B. No driver-risk dial (levels 0.05 and 0.1), compared with dial 0.02.** 200 seeded splits.

| level | unit | no dial | with dial 0.02 |
|---|---|---|---|
| 0.05 | workplace before May 2019 | 0.00 | 0.00 |
| 0.05 | workplace from May 2019 | 0.67 (52% of splits offer) | 0.66 (55%) |
| 0.05 | Norway pooled | 0.00 | 0.00 |
| 0.1 | workplace before May 2019 | 0.00 | 0.00 |
| 0.1 | workplace from May 2019 | 1.80 (32% of splits offer) | 7.22 (98%) |
| 0.1 | Norway pooled | 0.01 (1% of splits offer) | 0.02 (3%) |

Tier B (information, not the deciding tier) at level 0.1 for Norway pooled: 0.76 without the dial against 0.58 with it (44% and 60% of splits offer). Reading: without the dial the offer is above the dial's in one tier-A cell (0.67 against 0.66), equal at 0.00 in three cells and below the dial's in two (1.80 against 7.22; 0.01 against 0.02); it does not turn the zeros into offers at the levels tried.

**C. No hourly re-solving (the fixed plan), level 0.1, dial 0.02.** One run (the command shared with A).

| unit | fixed plan | one-way re-solve, known energy, level 0.1, dial 0.02 |
|---|---|---|
| Norway pooled | abstains: 17 of 100 calibration days offer | abstains: 17 of 100 offer |
| workplace before May 2019 | abstains (2 of 4) | abstains (2 of 4) |
| workplace from May 2019 | no scale is safe: 92 of 100 offer, the slot bound fails at every scale tried | scale 0.594, 8.179 |

Reading: without re-solving nothing is offered where re-solving offers at one unit. The figures of A and C come from one run, and those of B from 200 splits. The same unit at level 0.1 and dial 0.02 reads 8.179 in C and 7.22 in B; the two come from different commands (`learn_run` for C, `dial_choice` for B), and no cause of the difference is given here.

**D. One scale calibrated on five Norwegian development sites together** (level 0.5, dial 0.02, tier A, 200 seeded splits), against one scale per site.

| site (counting days) | own scale | pooled scale |
|---|---|---|
| OSL_1 (48) | 0.00 | 4.78 (driver loss over its limit in 12% of splits) |
| BER (47) | 0.00 | 0.22 |
| KRO (53) | 0.85 | 4.71 (slot loss over its limit in 32% of splits) |
| OSL_2 (2) | 0.00 | 0.62 (1 test day per split) |
| BAR_2 (92) | 0.03 | 1.71 |

Over the pool the scale was above 0 in 200 of 200 splits (mean scale 0.988), the mean offer over the five sites' test days was 2.67 (tier A; 2.71 in tier B) and the mean test slot loss was 0.051, over its limit in 0% of the splits. Reading: this is the only one of the four that raises the offer. Its guarantee holds on average over the pool and not for each site, which is a different guarantee from the plan's per-unit one (statement T6); whether it is taken further is a later decision. Two of the five sites are over a loss limit in some splits. No held-out unit was tried.

**E. More cautious energy bounds, first trial** (the cars' charging rate and room put at level 0.05, the energy a car needs unchanged, departure level 0.2, main model, 200 seeded splits, each arm at its own dial). Bars fixed before the run: 8% at the workplace unit from May 2019 and 30% at the pooled Norwegian unit.

| unit | cautious bounds, tier A | the usual (empirical) bounds, tier A | known energy, tier A |
|---|---|---|---|
| workplace from May 2019 | 0.00 (no measurable difference) | 0.00 | 14.47 |
| Norway pooled | 0.00 (no measurable difference) | 0.00 | 1.92 |
| workplace before May 2019 | 0.00 (not judged: no bar) | 0.00 | 0.00 |

Tier B (information, not judged): the workplace unit from May 2019 gives 0.83 for the cautious arm against 0.01 for the usual bounds (known energy 25.13), and Norway pooled 0.00 (known energy 5.26). Reading: one setting was judged (the rate and the room at level 0.05); this trial left the needed-energy bound unchanged (trial I changes it), so it does not show that no setting of the energy bounds helps. In tier A at the workplace unit from May 2019 the cautious arm's calibration finds no safe scale in 199 or 200 of 200 splits at every dial, and the one split that offers has a driver loss of 0.098 against the limit 0.05.

**F. Average of the two learned forecasts** (the tree model's and the neural model's departure forecasts; 200 seeded splits). Noise bars fixed in committed files before the run: 8% and 30%.

| unit | mix | main model alone | reading |
|---|---|---|---|
| workplace from May 2019 | 18.29 (+3.3%, bar 8%) | 17.70 | no measurable difference |
| Norway pooled | 2.33 (-6.3%, bar 30%) | 2.48 | no measurable difference |
| workplace before May 2019 | 0 | 0 | not judged |

Each arm is at its own dial; at the main model's dial the mix is lower (15.84). An extra line, the mix at level 0.4, gives 18.86 (+6.5%, bar 8%) at the workplace unit and -0.6% at Norway pooled, again no measurable difference.

**G. The main model refitted on a unit's own first 56 days** (two weightings: unweighted and equal-mass; 200 seeded splits, every arm scored on the days after the first 56). Bars as in F.

| unit | refit, unweighted | refit, equal-mass | not refitted | reading |
|---|---|---|---|---|
| workplace from May 2019 | 17.66 (+4.5%, bar 8%) | 12.44 (-26.4%) | 16.90 | unweighted: no measurable difference; equal-mass: shows a loss |
| Norway pooled | 2.62 (-3.8%, bar 30%) | 2.80 (+3.0%) | 2.72 | no measurable difference |

The workplace unit before May 2019 is left out (too few days). The summary's own line is "the refit hurts" (one arm shows a loss and none a gain). The "not refitted" figure differs from F's main-model figure because G scores only the days after the first 56.

**H. The neural model fitted without its set encoder** (the part that lets a car's encoding depend on the other cars of its site-day, as defined above; three seeds per neural model, 200 seeded splits, each model at its own dial; units: the workplace unit from May 2019 and Norway pooled). The reading rule and bars (8% and 30%) were fixed before the run; a gain needs the share above the bar and every seed figure of the no-encoder model above every seed figure of the comparison. The bars were set in advance as an analogy to an earlier development look at level 0.3, not as this trial's own measured noise.

The trial shows that removing the set encoder gives no measurable difference against the same network with it (workplace unit +3.6%, bar 8%; Norway pooled +8.2%, bar 30%); against `hazard_tuned_inputs` the shares are +12.9% (workplace unit) and +20.7% (Norway pooled), also no measurable difference. In tier B (information only), at Norway pooled the no-encoder model is below the main model (5.42 against 6.54) and below `hazard_tuned_inputs` (5.75). It does not show that the neural model is better than the main model: one workplace unit carries the gain against the main model (+18.2%, bar 8%), the Norwegian unit's difference (+21.1%) is inside its bar of 30%, each neural figure rests on three seeds, the 200 splits are drawn from the same development days and are not 200 independent samples, and the trial was one run. It says nothing about the held-out units.

Offers (mean of three seeds, least to greatest in brackets) at the workplace unit from May 2019 and at Norway pooled: main model 17.70 and 2.48; the same neural network with the encoder 20.19 (19.68 to 21.12) and 2.78 (2.36 to 3.11); the neural model the project reports (`hazard_tuned_inputs`) 18.54 (17.74 to 19.82) and 2.49 (2.02 to 3.04); the no-encoder model 20.92 (19.50 to 22.29) and 3.01 (2.13 to 3.46).

**I. More cautious energy bounds, second trial** (departure level 0.2, main model, 200 seeded splits, each arm at its own dial; the same three units as in E; bars as in E, 8% at the workplace unit from May 2019 and 30% at the pooled Norwegian unit). Five arms: known energy; the usual (empirical) bounds; trial E's setting re-run (the rate and the room at level 0.05, the needed energy as in the usual bounds); the energy a car needs alone at its 0.95 upper quantile (information only, read by no rule); and all three energy bounds at level 0.05 (the judged arm). One comparison is judged: all three at level 0.05 against the usual bounds, tier A. The settings were fixed before the run, which was made once. Check against trial E: the three arms that trial E also ran, known energy, the usual bounds and trial E's setting, gave tier A mean offers exactly equal to trial E's (the output's line reads "First run compared: equal", where the first run is trial E's); the check reported no fault.

| unit | known energy, tier A | usual bounds | trial E's setting | needed energy alone (information) | all three at level 0.05 | all three minus usual | reading |
|---|---|---|---|---|---|---|---|
| workplace from May 2019 | 14.47 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 (standard error 0.01); more in 7, same in 193, less in 0 of 200 splits; 0.1% of the known-energy offer, bar 8% | no measurable difference |
| Norway pooled | 1.92 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00; same in 200 splits; 0.0%, bar 30% | no measurable difference |
| workplace before May 2019 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | not judged (the unit has no bar) | not judged |

The fixed rule's one-line reading is "no measurable difference; a mixed result is reported as mixed." At the workplace unit from May 2019 the seven splits that offer under the judged arm have a mean test driver loss of 0.083, against the driver limit of 0.05, and a mean test slot loss of 0.072; each mean is over the splits that offer, not over all 200.

Tier B (information, not judged): at the workplace unit from May 2019 the offers are 25.13 for known energy, 0.01 for the usual bounds, 0.83 for trial E's setting, 0.12 for the needed energy alone and 0.39 for all three at level 0.05 (0.38 above the usual bounds, standard error 0.02; more in 172, same in 26, less in 2 of 200 splits; 1.5% of the known-energy offer). At Norway pooled every arm but known energy (5.26) gives 0.00.

Reading: on development data, at these two units, departure level 0.2 and the main model, putting all three energy bounds at level 0.05 does not bring back the tier A offer that the usual bounds lose against known energy (14.47 and 1.92 kWh per day lost; 0.1% and 0.0% of it recovered). One setting was judged in this trial, so it does not show that the bounds are equal, or that no setting of the energy bounds recovers the offer. In tier B at the workplace unit all three at level 0.05 give less (0.39) than trial E's setting alone (0.83). Across E and I, two settings were judged: the rate and the room at level 0.05, and all three bounds at level 0.05; the needed energy alone at its 0.95 upper quantile was information only. This trial did not try: levels between, other departure levels, another model, energy rows in the certificate (a plan change). It says nothing about held-out units or the Dutch and Korean sites.

## 16. Limits

**Not measured or cut.** A step for dated events by a language model and two foundation-model forecasters were cut before the plan was frozen. The block product's calibrated offers were cut (only O2 and the ledger's block rows remain). Two-way charging at budgets of 2 and 10 kWh and the discharge given back are not measured. A second rate rule (charging evenly before the decision time, a site cap) is not run. The held setting is run in full for the main model only. The tier B curve over the dial is not computed for held-out units. Intervals were planned for the main model and the neural model only, and the run's own interval steps for their headlines failed (the rho 0.1 and 0.2 intervals of the main model passed). A contextual sizing rule and conformal training were tried on development data and none was kept: a sizing had to beat the fixed multiplier and its shuffled controls, and did not. Four further published methods are not run. The earlier analysis is not rebuilt as a comparison method, and beating it is not claimed.

**What the data do not show.** One Dutch office car park carries the result. The claim supported is that at one Dutch workplace car park the calibrated offer is about 36 to 37% of the hindsight ceiling; it is not that the method works on most fleets and not a statement about which kinds of fleet can sell reserve. The nine held-out units differ in size and kind, and per-unit points say nothing about other units. Regime effects are not split beyond the two Dutch periods. Clocks and charging rates are assumed for some data sets (the Korean rate of 7 kW, one development data set's clock). The offer is measured on logged sessions with an assumed uncontrolled-charging truth; the charging that would have happened under the offered box, drivers' response and a market are not observed. At three Korean units (public parking, visits and workplaces) even perfect knowledge of departures gives an offer on only 3 of 100, 0 of 31 and 4 of 100 calibration days (read from the run's results file) under the assumed charger power and window, below the 9 offering days (slot loss) and 19 (driver loss) the calibration needs, so a better departure forecast cannot by itself make them offer; whether a forecast that overstates how long cars stay could still certify some value there, at a cost to drivers, was not examined.

**What the run was.** One logged run from the frozen plan (its third start) with two corrections to file readers, whose stored outputs were read while it went; every failed step is named beside any figure read from steps that passed, and it is never presented as a passed run. The favourable figures carry the conditions of section 3: one site gives nearly all of the main model's sum, six of nine units offer nothing, the energy is taken as known, and no unit passes the guarantee check (under reading A none is judged). The zeros (energy forecast, no dial, fixed plan) are results of one pre-set setting each, not proofs that nothing is possible; section 15 gives the trials made afterwards and what they do not show.

**Limits of the theory.** T2 states a certificate for a lower-triangular affine rule (each hour's charging may depend linearly on the requests already seen), not a causal optimum, and its proof is a hand-checked sketch. T4 for the free shape is a hand proof, and what is claimed for it is what the entry "T4 of Section 4: a correction of the plan's sentence" of the plan's list of changes (section 14) calls "this sentence, the short form of the hand proof", which this page does not restate; the entry records a decision of 24 November 2026, made after held-out output was seen; the proof was written by hand and no project run is part of it; a separate review of the proof, by hand, read "holds with gaps", and its four gaps were all in that short form and are corrected in the entry's text; the machine checks of the proof's claims are the 30 tests of `tests/wp5/test_t4_hand_proof.py` as at commit 20ae92d135da72a34fbae5b1f7a2720ee9b00113, a commit made after the tag `frozen-plan`; the tests are not in the code at that tag, and they run on made-up days and hand-made days; the short form does not cover fixed shapes, forecast sets, several decision hours or the code, whose equalities are tested only, and the conjecture that the affine and real-time values are equal for fixed shapes is open. T5's second statement is checked, with a proof sketch under stated conditions. T6 holds for exchangeable offering days and bounds the expected loss over calibration draws, not a day and not one unit's realised mean; its driver bound is a sum of per-car worst cases and is conservative. The driver-risk rows are a model of risk, not a guarantee.

## 17. Rebuilding

The one command that writes every table file named above (outputs O1 to O10 and the development figures), run from the repository with the records of the held-out run in place (the file that records each step of the run, and the summaries written by the steps that passed; the command opens no held-out data file). It ends with exit status 1 by its own rule, because four steps of the held-out run failed, so the run's record is not complete:

```
python -m fleetreserve.wp16_run --root <repository> --out <output folder> --dev-days-run <run folder> --dial-choice-run <run folder>
```

The last two options each take the folder of a development run. The development figures are copied from those two runs: the checks on development days (`python -m fleetreserve.dev_day_checks --t2-sample-size 100 --energy-bounds known --uncertifiable-car leave-unsteered --kappa 0.1 --departure-floor as-planned --workers 4`) and the dial choice (`python -m fleetreserve.dial_choice --model lgbm_tail --matrix gate/matrix.json --splits 200 --workers 6`), both recorded at commit 0f62fc089033b29b378aa2c3068fdc65f43282e0 with exit status 0. The held-out files are placed in `data/heldout/` as downloaded, except the two held-out Norwegian sites, whose rows a tested split script writes there from the Norway file (see `docs/README.md`); the held-out loader opens them, at the frozen tag and at the logged parsing-fix commits, and the looks recorded in `docs/prior-looks.md` are the only other reads. The held-out run itself is the one drive recorded in `wp16-manifest.json`; it is not repeated by the command above.

The checks: `bash check.sh` (the checks that guard every change; exit 0 passed, 1 failed, 2 could not run) and `bash passmark.sh wp14` (the pass mark of the held-out run, which returned status 1 for the run described above). The table files are written as `.csv` and `.md`, and some also as `.png`.
