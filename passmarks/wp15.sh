# The pass mark of work package 15 (plan section 13, WP15: "Sensitivities: ρ=0.1, 0.2 for the main
# model only; chronological calibration (T10, at OSL_T); regimes (no split beyond the two a.s.r.
# units). Pass: all reported."), sourced by passmark.sh. One line, changed only together with
# tests/test_passmark.py, which names it.
# tools/wp15_check.py reads the plan's line through the readings W1 to W3 that the committed run
# list run-lists/run-list.json states in its about (not plan text), finds the entries they name in
# that list, and reads their runs from the run from the tag: the one complete sealed-run drive of
# the list at the checked-out commit, read as the WP14 check reads it (tools/sealed_run_check.py),
# never from the rehearsal. It opens nothing in the sealed folder. With no complete drive it prints
# the incomplete run first, then the parts readable from the steps that passed as lines begun
# "information:", and exits 1, never 0. Run it in the project's own folder, at the tag or at the
# logged parsing fix the gate of the run from the tag passes at, with a clean tree, after the
# sealed-run drive.
# A part whose run ended and failed, or whose results lack a needed figure, fails the item; so does
# a step that timed out in a drive that ran to its end: the drive is then incomplete, the item
# exits 1 and names the timeout (the part is lost). No sealed-run drive log, only rehearsal logs, a
# stale or broken-off drive, a missing entry, a run that was stopped, or one that does not verify:
# the item could not run and is never a pass.
check "sensitivities all reported, from the run from the tag by the readings W1 to W3 that the committed run list states in its about: rho 0.1 and 0.2 for the main model, its learn and interval entries at each rho holding the offer and both ceilings; the chronological calibration, the drift guard's entry on OSL_T alone holding its per-day losses and offers; the regimes, the two a.s.r. units scored as units of their own by the main model's learn entry (plan 13 WP15)" uv run --frozen python tools/wp15_check.py
