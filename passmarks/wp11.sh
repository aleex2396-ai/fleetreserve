# The pass mark of work package 11 (plan section 13, WP11: "Corner search; run matrix frozen.
# Pass: closed-form worst case reproduced on 20 days; run matrix frozen (one check item ...)"),
# sourced by passmark.sh. One line per clause, in the plan's order; a line is changed only
# together with tests/test_passmark.py, which names both.
# tools/corner_closed_form_check.py reads the newest recorded corner search (label
# wp11-corner-search, `python -m fleetreserve.run_record --name wp11-corner-search --long
# --max-seconds <n> -- uv run --frozen --group learn python -m fleetreserve.corner_run --model
# <model> --dial <dial> --matrix gate/matrix.json --days <n> --workers <n>`, after a truth build
# at the same commit) and needs its manifest to verify, on the committed gate/matrix.json, with a
# verified main truth build, each recorded at the checked-out commit or read from an earlier
# commit by the rule of fleetreserve.run_evidence. A pair whose closed form differs from the
# corner worst case fails the item; no run, a stale run, fewer than 20 reproduced days or a run
# that cannot decide: the item could not run and is never a pass.
check "fixed plan's closed-form worst case reproduced on 20 days in the recorded corner search (plan 13 WP11)" uv run --frozen python tools/corner_closed_form_check.py
# tools/freeze_check.py matrix (its module text) reads, at HEAD in a clean tree, the committed run
# list run-lists/run-list.json and every matrix it names by --matrix (gate/sealed-matrix/,
# gate/rehearsal-matrix/), and the kappa-choice results each matrix records, verified by their
# runs' manifests. A named file not committed, changed or hidden, a faulty run list, a matrix that
# fails sealed_matrix.check, whose source has another SHA-256 or that differs from what
# sealed_matrix writes from it, or run-list dials other than the main model's: failed. No run
# list, no matrix written yet, a source run missing or an unclean tree: could not run.
check "run matrix frozen: the committed run list run-lists/run-list.json and every matrix it names, committed and unchanged at HEAD, each what sealed_matrix writes from its verified kappa-choice results, the run list's dials the main model's (plan 13 WP11)" uv run --frozen python tools/freeze_check.py matrix
