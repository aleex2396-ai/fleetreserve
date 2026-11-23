# The pass mark of work package 9 (plan section 13, WP9: "Calibration, conformal training, T7
# test, η. Pass: loss 0 at abstain; development loss within its 200-split spread; T8's check (if
# failed, the owner may cut 3: conformal training ...); conformal training kept only if it raises
# the development offer"), sourced by passmark.sh. One item per clause, in the plan's order; the
# items are changed only together with tests/test_passmark.py, which names them. Each runs
# tools/wp9_check.py <item>, whose module text gives the readings W1 to W4f and the runs read:
# - abstain: the newest gate run (label wp6-gate, python -m fleetreserve.gate_run --matrix
#   gate/matrix.json); an abstaining entry with an offer or a loss that is not 0 fails it.
# - spread: lgbm_tail's dial-choice run (label wp6-dial-choice-lgbm_tail) and its comparison run at
#   each gate unit's tier A dial (labels wp10-compare-lgbm_tail-eps<dial>, python -m
#   fleetreserve.compare_run), read from the split table: per gate unit, the fixed plan's mean test
#   loss over the 200 development splits that offer, slot (both tiers) and driver (tier A), above
#   the level plus two standard errors fails it.
# - t8: the newest run of each soft-scale check listed in passmarks/wp9-t8-runs.json (labels
#   wp9-soft-scale-check-<...>, python -m fleetreserve.soft_scale_check), combined over distinct
#   days; a failed deciding comparison fails it.
# - conformal: a decision, not a pass of the work: "kept" or "not kept" per arm from the newest
#   sizing comparison (label wp9-size-resolve-whatif-ct-lgbm_tail_rotation, python -m
#   fleetreserve.size_resolve_whatif with --development-units, the gate units scored with sizings
#   fitted on every training unit only); exit 0 when both arms are decided. Only the owner cuts.
# Every run must verify, on the committed gate/matrix.json, with a verified main truth build, each
# recorded at the checked-out commit or read from an earlier commit by the rule of
# fleetreserve.run_evidence (item t8 also compares passmarks/wp9-t8-runs.json). No run, a stale
# run or one that cannot decide: the item could not run and is never a pass.
check "loss 0 at abstain in both tiers, in the recorded gate run (plan 13 WP9)" uv run --frozen python tools/wp9_check.py abstain
check "development loss within its 200-split spread: the mean test loss over the 200 development splits at most the level plus two standard errors, slot loss in both tiers and driver loss in tier A, per gate unit, in the recorded comparison runs (plan 13 WP9)" uv run --frozen python tools/wp9_check.py spread
check "T8's check in the recorded soft-scale checks: a passed comparison on 100 days in every direction, no failed comparison, an MPS batch timed, tracking logged (plan 13 WP9)" uv run --frozen python tools/wp9_check.py t8
check "conformal training kept or not kept, decided from the recorded sizing comparison on the development units (plan 13 WP9: kept only if it raises the development offer)" uv run --frozen python tools/wp9_check.py conformal
