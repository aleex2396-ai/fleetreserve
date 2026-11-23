# Fleet reserve: an hourly offer from charging electric cars

A car park holds many electric cars that are plugged in at once. Each morning, at a fixed decision time, the car park can offer a **box**: for each of a few later hours (the service hours, at most 8), a range of total charging power in kW. In each service hour a request inside that range, known at the start of the hour, must be met by the combined charging of the cars. Each car should still get the energy it needs before it leaves; how far cars fall short is measured and limited, as below. The size of the offer is the total width of the ranges, in kWh.

This project measures how large such an offer can be, with stated guarantees, on public charging data.

## What is built
- **Hindsight ceiling.** For each day, the largest box that could have been kept had each car's departure time, charging rate and energy need, and each hour's request, been known in advance. It is solved exactly.
- **Offer.** A box made at the decision time and kept by per-car charging rules that use only what is known by the hour in which each acts. It is calibrated: a scale factor on the box is chosen on calibration days of the unit so that, per unit (a car park site, a regime of the a.s.r. or NREL data, or a Korean portfolio), the expected mean availability of service hours on offering days is at least 90% (hours from the first missed request on count as lost) and the expected mean share of protected energy (the energy each car needs, less the allowed fraction rho, 0 in the main setting) left undelivered at departure is at most 5%, the expectation taken over the random choice of calibration days, with days assumed exchangeable (conformal risk control). This holds in the driver-protected tier; a second tier calibrates on the first limit alone.
- **Departure models.** Learned forecasts of when each plugged-in car leaves. Their bounds set how much each car gives.
- **Checks of the stated theorems** on the data, with comparisons against published charging controllers, the cost of deciding in real time, simple (affine) rules, unknown cars, and two-way charging.

## Plan and prior looks
`docs/plan.md` is the specification. It is frozen at the git tag `frozen-plan`; after the tag, changes are listed only in its section 14. `docs/prior-looks.md` records every look at the data before the tag.

Data roles: training ACN and Norway; development NREL and small Norwegian sites; held-out (kept sealed for the single final run, opened only by its loader, at the tag and at any logged parsing fix) a.s.r., two Norwegian sites and Korean portfolios. A pass mark is the condition that finishes a work package (plan section 13). Results are not yet in; they are added after the one run on the held-out data.

This repository holds snapshots of a local project; commit identifiers, run names and numbered decisions and readings named in the documents refer to that project's local history and records, which are not included.

## Known text errors at the tag

These lines of the frozen code are wrong in wording only; no number, row or outcome depends on them, and they are corrected after the held-out run.
- `tools/learn_logged_check.py` (its description and the label "hazard model (main)" that it prints) calls the hazard model the plan's main model and its variant an open choice; plan section 7 names `lgbm_tail` the main model and runs `hazard_tuned_inputs` beside it.

## Data
Each set's source and licence are in the table of plan section 3 (sources: ACN-Data, data.nlr.gov, Zenodo, 4TU, figshare). `tools/data_sources.csv` lists the files that `tools/fetch_data.py` downloads, with role and URL, and size and MD5 for all but the development file `dataSet_Oct2021.csv`. No charging data file is in this repository (`data/` is ignored by git). `data-hashes.json` lists each file's size, SHA-256 and role. `tools/fetch_data.py` downloads the listed files and checks them; `tools/fetch_acn.py` downloads ACN office001 sessions with an access token that the script reads itself. Held-out files are saved in `data/heldout/` as downloaded, except the two held-out Norwegian sites, whose rows a tested split script wrote there from the Norway file, printing hashes and counts but no row (plan section 3); only the held-out loader opens them, at the tag.

## Layout
- `src/fleetreserve/`: the method (loaders, truth sets, solvers, offer, calibration, learning, comparisons).
- `tests/`: the tests of the code; `tests/slow.txt` lists the slow ones, which `check.sh` leaves out.
- `docs/`: `plan.md` (the specification), `prior-looks.md` (looks at data before the tag), `README.md` (this page).
- `passmarks/`, `passmark.sh`: the pass mark list of each work package that has a command, and the command that runs one.
- `check.sh`: the one command that checks the code.
- `tools/`: check scripts, data download and hash tools, repository rules.
- `gate/`: run matrices, including those of the held-out run and its rehearsal.
- `learn/`: settings and input tables of the departure models and drift guard.
- `run-lists/`: the run list of the held-out run and its rehearsal.
- `data-hashes.json`: fingerprints of the data files.
- `pyproject.toml`, `uv.lock`, `.python-version`: packages, locked versions, Python version (3.12).
- `.gitignore`: keeps data, run output and keys out of git.

## Running the checks
Install `uv`; Python 3.12 comes from `.python-version`.
- `bash check.sh`: the checks that guard every change. Exit 0 passed, 1 a check failed, 2 a check could not run; only 0 is a pass.
- `bash tools/check-slow.sh`: the slow tests, with the same exit codes.
- `bash passmark.sh wp<N>`: whether work package N is finished, for example `bash passmark.sh wp1`. Same exit codes. Some items need the data.
