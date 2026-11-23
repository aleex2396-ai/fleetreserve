# The pass mark of work package 14 (plan section 13, WP14: "Pass: one logged run plus logged
# parsing fixes, the scoring runs made again at a fix commit belonging to that one run; logs
# hashed"), sourced by passmark.sh. One line, changed only together with tests/test_passmark.py.
# tools/sealed_run_check.py checks the three marks and states what it reads; it opens nothing in
# the sealed folder. Run it in the project's own folder, at the tag or at the logged parsing fix
# the gate of the run from the tag passes at, with a clean tree, after the sealed-run drive, under
# the run recorder: `python -m fleetreserve.run_record --name wp14-passmark -- bash passmark.sh
# wp14`. The check writes wp14-passmark.json into the run folder (each mark's status and findings,
# the SHA-256 of every input) and could not run without one; the line passes the recorder's
# FLEETRESERVE_RUN_DIR on, since `check` runs each command with PATH and HOME alone.
check "one logged run plus logged parsing fixes, the scoring runs made again at a fix commit belonging to that one run; logs hashed: exactly one complete sealed-run drive of the committed run list at the checked-out commit (the tag's or a logged parsing fix the gate accepts), every opening inside a step, every earlier drive that ran a step an operational stop with no opening or stopped at a logged layout refusal, every run's manifest and log.txt hashed as logged; results in wp14-passmark.json (plan 13 WP14)" env FLEETRESERVE_RUN_DIR="${FLEETRESERVE_RUN_DIR-}" uv run --frozen python tools/sealed_run_check.py
