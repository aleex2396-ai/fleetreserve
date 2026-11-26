"""Outcome O9: the cold-start curves (plan section 11, O9).

"O9 Cold start, k=0, 7, 14, 28, 56 local days in time order: availability A_k, offer share Y_k,
against pooled LightGBM."

The steps `cold-start-workplaces` and `cold-start-others` (`fleetreserve.cold_start`) write
`cold-start.json`: per model, unit, k and tier, `figures.<model>.<unit>.<k>.<tier>` with the
offering local days, the scale, the offer per counting scored day, `availability` (A_k: 1 minus the
mean slot loss over the offering scored days), `offer_share.<ceiling>.share_of_ceiling` (Y_k) and
the failing scored days. The models are `models.cold_started` (`lgbm_tail`) and
`models.reference` (pooled LightGBM, `lgbm`).

Readings (alternatives in brackets):
  S1. Tier A, the one-way rung, at the run's one dial; Y_k is the share of the hindsight ceiling
      (`run.the_ceiling`), the best-causal share beside it. [Alternative: tier B as well.]
  S2. One row per model, unit and k, as the file has it; no figure over units is made (the scored
      days differ by unit), and none is written by an entry of the run list.
  S3. A unit scored by neither step (the truth build gave it no window) has its reason in one row;
      a model and k that was not scored for a unit says "not scored" with the run's failure list.
"""

from fleetreserve.wp16_cells import Missing, Table
from fleetreserve.wp16_drive import Missed, Source

STEPS = ("cold-start-workplaces", "cold-start-others")
RESULTS = "cold-start.json"
TIER = "A"
COLUMNS = (
    "model",
    "unit",
    "k (local days)",
    "offering local days",
    "scale",
    "offer (kWh per counting scored day)",
    "A_k: availability",
    "Y_k: share of the hindsight ceiling",
    "share of the best-causal ceiling",
    "offering scored days",
    "failing scored days",
    "why scale 0",
)


def _share(found: dict, name):
    one = (
        found.get("offer_share", {}).get(name, {})
        if isinstance(found.get("offer_share"), dict)
        else {}
    )
    return one.get("share_of_ceiling") if isinstance(one, dict) else None


def build(source: Source, units: list, absent: dict) -> tuple[list[Table], list[dict]]:
    table = Table(
        "o9-cold-start", "O9: availability and offer share by days of local history", COLUMNS
    )
    seen: set = set()
    curves: dict = {}
    for label in STEPS:
        try:
            document = source.json(label, RESULTS)
        except Missed as problem:
            table.add("all models", f"units of {label}", "", Missing(problem.text), *([""] * 8))
            continue
        run = document.get("run", {}) if isinstance(document.get("run"), dict) else {}
        models = document.get("models", {}) if isinstance(document.get("models"), dict) else {}
        ceilings = run.get("ceilings") if isinstance(run.get("ceilings"), list) else []
        names = (ceilings + [None, None])[:2]
        found = document.get("units") if isinstance(document.get("units"), dict) else {}
        for unit in found:
            seen.add(unit)
            for model in (models.get("cold_started"), models.get("reference")):
                by_k = document.get("figures", {}).get(model, {}).get(unit, {})
                for k in run.get("local_days", []):
                    one = by_k.get(str(k), {}).get(TIER) if isinstance(by_k, dict) else None
                    if not isinstance(one, dict):
                        status, failed = found[unit].get("status"), document.get("failed")
                        why = f"not scored: {status}; failed: {failed}"
                        table.add(model, unit, k, Missing(why), *([""] * 8))
                        continue
                    table.add(
                        model,
                        unit,
                        k,
                        one.get("offering_local_days"),
                        one.get("scale"),
                        one.get("offer_kwh_per_counting_scored_day"),
                        one.get("availability"),
                        _share(one, names[0]),
                        _share(one, names[1]),
                        one.get("offering_scored_days"),
                        one.get("failing_scored_days"),
                        one.get("scale_zero_reason") or "",
                    )
                    curves.setdefault((model, unit), []).append(
                        (k, one.get("availability"), _share(one, names[0]))
                    )
    for unit in units:
        if unit not in seen:
            why = absent.get(unit, "no cold-start step holds it")
            table.add("all models", unit, "", Missing(f"no figure: {why}"), *([""] * 8))
    table.notes.append("Readings S1 to S3 of the module text of fleetreserve.wp16_o9.")
    figures = [{"kind": "o9", "curves": [(m, u, pts) for (m, u), pts in curves.items()]}]
    return [table], figures
