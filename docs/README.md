# Fleet reserve: an hourly offer from charging electric cars

A car park holds many electric cars that are plugged in at once. Each morning, at a fixed decision time, the car park can offer a **box**: for each of a few later hours (the service hours, at most 8), a range of total charging power in kW. In each service hour a request inside that range, known at the start of the hour, must be met by the combined charging of the cars. Each car should still get the energy it needs before it leaves; how far cars fall short is measured and limited, as below. The size of the offer is the total width of the ranges, in kWh.

This project measures how large such an offer can be, with stated guarantees, on public charging data.

## What is built
- **Hindsight ceiling.** For each day, the largest box that could have been kept had each car's departure time, charging rate and energy need, and each hour's request, been known in advance. It is solved exactly.
- **Offer.** A box made at the decision time and kept by per-car charging rules that use only what is known by the hour in which each acts. It is calibrated: a scale factor on the box is chosen on calibration days of the unit so that, per unit (a car park site, a regime of the a.s.r. or NREL data, or a Korean portfolio), the expected mean availability of service hours on offering days is at least 90% (hours from the first missed request on count as lost) and the expected mean share of protected energy (the energy each car needs, less the allowed fraction rho, 0 in the main setting) left undelivered at departure is at most 5%, the expectation taken over the random choice of calibration days, with days assumed exchangeable (conformal risk control). This holds in the driver-protected tier; a second tier calibrates on the first limit alone.
- **Departure models.** Learned forecasts of when each plugged-in car leaves. Their bounds set how much each car gives.
- **Checks of the stated theorems** on the data, with comparisons against published charging controllers, the cost of deciding in real time, simple (affine) rules, unknown cars, and two-way charging.

## Results

The held-out run used data that no loader opened until the plan was frozen (the looks are listed in `docs/prior-looks.md`; the main ones are described in `docs/results.md`, section 1), in nine units (sites, site groups or periods of one site). An offer is the summed width of the promised power ranges (kWh). A counting test day is a day outside the calibration days (random days of a unit used to set the scale) with at least three cars plugged in at decision time.

**What failed or is missing**
- Pass mark (the test the run had to meet) not met: 26 of the run's 30 steps passed; the four that failed compute 95% intervals.
- The run judged no unit against the guarantee (service hours kept at least 90%, undelivered driver energy at most 5%): six units offer nothing (scale, the shrinking factor, is 0) and are counted as not judged, a reading chosen after the run began; the other three have no interval. Made apart afterwards, after held-out output was seen, information only: for the Dutch office car park, both intervals before March 2023 lie wholly inside the limits, and from March 2023 the availability interval includes its limit; for TRO_R (a Norwegian site) it lies wholly below it, and TRO_R's shortfall interval contains the 5% limit.
- At TRO_R the run's average loss of service hours, over its 121 test days with an offer, was 14.57% against the 10% limit: a missed pass mark, not a counterexample, since the limit is promised on average over random calibration draws.

**Headline figures** (main model: a tree forecast of departures, offer re-solved each hour)
- 130.153 kWh per counting test day, summed over the nine units' offers; 16.1075 kWh per counting test day, all units' days pooled. Made apart, 95% intervals are 110.105 to 140.741 for the first figure and 13.819 to 17.299 for the second; the headline's pass mark is not judged from them.
- 51.03 and 78.48 kWh per counting test day of the 130.153 come from one Dutch office car park in its two periods: 35.9% and 37.4% of its hindsight ceiling (the largest offer possible knowing every arrival and departure in advance).
- Each car's charging rate, room (energy a car can still take) and energy need are taken as known. At the one pre-set setting tried for each, the offer is 0 when they are forecast, in the driver-protected tier (undelivered driver energy limited) without the driver-risk dial (a limit on the expected shortfall of a car's needed energy if it leaves earlier than forecast), and without hourly re-solving.

**Trials after the held-out run**
Nine trials used development units only; six tried to turn zero offers into offers, three tried other departure forecasts. They are not results of the plan. Of the six, only pooling one scale across five development sites raised the driver-protected offer: 2.67 kWh per counting test day over the pool, above each site's own-scale offer. In the other five, the changed arm was never more than 0.01 above its comparison in that tier. Pooling's guarantee is over the pool, not per site; two sites broke a limit in some splits; no held-out unit was tried. Both energy-bound trials judged one setting. The second made all three energy bounds more cautious and showed no measurable difference in the driver-protected tier; it does not show that no setting helps. One neural variant, without its set encoder (one part of the neural model), showed a gain over the main model at one of two development units by a rule fixed beforehand; against the same network with the encoder it showed no measurable difference; it does not show the neural model is better and says nothing about held-out units.

The full account, with every table, is in `docs/results.md`.

## Plan and prior looks
`docs/plan.md` is the specification. It is frozen at the git tag `frozen-plan`; after the tag, changes are listed only in its section 14. `docs/prior-looks.md` records every look at the data before the tag, and the looks taken after it.

Data roles: training ACN and Norway; development NREL and small Norwegian sites; held-out (kept for the single final run and opened by its loader, at the tag and at any logged parsing fix, apart from the looks recorded in `docs/prior-looks.md`) a.s.r., two Norwegian sites and Korean portfolios. A pass mark is the condition that finishes a work package (plan section 13). The results of that run are in the section Results above and in `docs/results.md`.

This repository holds snapshots of a local project; commit identifiers, run names and numbered decisions and readings named in the documents refer to that project's local history and records, which are not included.

## Known text errors at the tag

These lines of the frozen code are wrong in wording only; no number, row or outcome depends on them, and they are not corrected in this snapshot.
- `tools/learn_logged_check.py` (its description and the label "hazard model (main)" that it prints) calls the hazard model the plan's main model and its variant an open choice; plan section 7 names `lgbm_tail` the main model and runs `hazard_tuned_inputs` beside it.

## Data
Each set's source and licence are in the table of plan section 3 (sources: ACN-Data, data.nlr.gov, Zenodo, 4TU, figshare). `tools/data_sources.csv` lists the files that `tools/fetch_data.py` downloads, with role and URL, and size and MD5 for all but the development file `dataSet_Oct2021.csv`. No charging data file is in this repository (`data/` is ignored by git). `data-hashes.json` lists each file's size, SHA-256 and role. `tools/fetch_data.py` downloads the listed files and checks them; `tools/fetch_acn.py` downloads ACN office001 sessions with an access token that the script reads itself. Held-out files are saved in `data/heldout/` as downloaded, except the two held-out Norwegian sites, whose rows a tested split script wrote there from the Norway file, printing hashes and counts but no row (plan section 3); the held-out loader opens them, at the tag and at the logged parsing-fix commits; the looks recorded in `docs/prior-looks.md` are the only other reads.

## Layout
- `src/fleetreserve/`: the method (loaders, truth sets, solvers, offer, calibration, learning, comparisons).
- `tests/`: the tests of the code; `tests/slow.txt` lists the slow ones, which `check.sh` leaves out.
- `docs/`: `plan.md` (the specification), `prior-looks.md` (looks at data before the tag, and the looks taken after it), `results.md` (the results of the held-out run and of the later trials), `README.md` (this page).
- `passmarks/`, `passmark.sh`: the pass mark list of each work package that has a command, and the command that runs one.
- `check.sh`: the one command that checks the code.
- `tools/`: check scripts, data download and hash tools, repository rules.
- `gate/`: run matrices, including those of the held-out run and its rehearsal.
- `learn/`: settings and input tables of the departure models and drift guard.
- `run-lists/`: the run list of the held-out run and its rehearsal.
- `trials-after-sealed-run/`: the settings of the trials made after the held-out results were seen, with their commands (`README.md`).
- `data-hashes.json`: fingerprints of the data files.
- `pyproject.toml`, `uv.lock`, `.python-version`: packages, locked versions, Python version (3.12).
- `.gitignore`: keeps data, run output and keys out of git.

## Running the checks
Install `uv`; Python 3.12 comes from `.python-version`.
- `bash check.sh`: the checks that guard every change. Exit 0 passed, 1 a check failed, 2 a check could not run; only 0 is a pass.
- `bash tools/check-slow.sh`: the slow tests, with the same exit codes.
- `bash passmark.sh wp<N>`: whether work package N is finished, for example `bash passmark.sh wp1`. Same exit codes. Some items need the data.
