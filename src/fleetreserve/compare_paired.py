"""Paired differences between the comparison methods over the seeded splits of a unit (plan
section 6: the seeded development splits; section 10: the comparison methods).

Input: the per-split rows of `compare_splits` (one per unit, variant, tier, method and split; only
the fields `unit`, `variant`, `tier`, `method`, `seed` and `offer_kwh_per_counting_test_day` are
read). Pure: no files, no randomness.

One pair (method, reference) of one unit, variant and tier. A split of the run (`seeds`) is
paired when both methods have a row for it with a finite offer. A split with no row, or with an
empty or non-finite offer, for either method is left out of that pair and counted
(`splits_left_out_value_missing`); it is never read as 0. On each paired split `s` the difference
is `d_s` = the method's offer minus the reference's offer on that same split (kWh per counting
test day). Over the paired splits:

- `mean_difference_kwh_per_test_day`: the mean of `d_s`;
- `difference_p5_kwh_per_test_day`, `difference_p95_kwh_per_test_day`: the 5th and 95th
  percentiles of `d_s` (`numpy.quantile`, linear interpolation between order statistics);
- `share_of_paired_splits_above_0`, `_equal_0`, `_below_0`: the shares of paired splits with
  `d_s` above, exactly equal to and below 0;
- `method_mean_offer_kwh_per_test_day`, `reference_mean_offer_kwh_per_test_day`: the mean offers
  of the two methods over the paired splits;
- `mean_difference_share_of_reference_mean_offer`: the mean difference over the reference's mean
  offer on the paired splits; None when that mean offer is 0 (the summary writes "no offer");
- `verdict` (`VERDICTS`, a fixed rule of this module that no figure of a run sets or moves):
  "ahead" when the 5th percentile is above 0, "behind" when the 95th percentile is below 0, "not
  separated" otherwise.

With no paired split every figure and the verdict are None. With very few paired splits the
percentiles, and so the verdict, rest on very few differences: with one paired split both
percentiles are that one difference, so the verdict is "ahead" or "behind" whenever it is not 0.

What the percentiles are and are not. The splits of a unit share the same days: they differ only
in which days calibrate and which are scored. The 5th to 95th percentile range is the spread of
the difference over those draws at this unit, under this model and dial. It is not a confidence
interval for new days, for other units or for other sites, and the splits are not independent
samples of anything beyond this unit's days.

The pairs (`pairs`), in the order of the run's methods. "against_re_solve": every other method
against `re-solve` as reference, when `re-solve` is in the run (else none). "other_pairs": every
two methods other than `re-solve`, the one listed earlier as reference.

`paired(rows, units, variants, tiers, methods, seeds)` gives unit -> variant -> tier ->
{"against_re_solve": [...], "other_pairs": [...]}, each entry the method, the reference and the
figures above. A row listed twice (the same unit, variant, tier, method and seed) is refused with
`ValueError`.
"""

import math

import numpy as np

from fleetreserve.compare_methods import RE_SOLVE

OFFER = "offer_kwh_per_counting_test_day"
QUANTILES = (0.05, 0.95)
VERDICTS = ("ahead", "behind", "not separated")
TABLES = ("against_re_solve", "other_pairs")
FIGURES = (
    "mean_difference_kwh_per_test_day",
    "difference_p5_kwh_per_test_day",
    "difference_p95_kwh_per_test_day",
    "share_of_paired_splits_above_0",
    "share_of_paired_splits_equal_0",
    "share_of_paired_splits_below_0",
    "method_mean_offer_kwh_per_test_day",
    "reference_mean_offer_kwh_per_test_day",
    "mean_difference_share_of_reference_mean_offer",
    "verdict",
)


def verdict(p5: float, p95: float) -> str:
    """The verdict word of the module text."""
    if p5 > 0:
        return VERDICTS[0]
    if p95 < 0:
        return VERDICTS[1]
    return VERDICTS[2]


def _finite(value) -> bool:
    return value is not None and math.isfinite(value)


def pair_statistics(method: dict, reference: dict, seeds) -> dict:
    """The figures of one pair (module text). `method`, `reference`: seed -> offer (None, or no
    entry, for a missing value)."""
    paired = [s for s in seeds if _finite(method.get(s)) and _finite(reference.get(s))]
    found = {
        "splits_paired": len(paired),
        "splits_left_out_value_missing": len(seeds) - len(paired),
        **dict.fromkeys(FIGURES),
    }
    if not paired:
        return found
    ours = np.array([method[s] for s in paired], dtype=float)
    theirs = np.array([reference[s] for s in paired], dtype=float)
    differences = ours - theirs
    p5, p95 = (float(q) for q in np.quantile(differences, QUANTILES))
    mean = float(differences.mean())
    reference_mean = float(theirs.mean())
    n = len(paired)
    found.update(
        {
            "mean_difference_kwh_per_test_day": mean,
            "difference_p5_kwh_per_test_day": p5,
            "difference_p95_kwh_per_test_day": p95,
            "share_of_paired_splits_above_0": int((differences > 0).sum()) / n,
            "share_of_paired_splits_equal_0": int((differences == 0).sum()) / n,
            "share_of_paired_splits_below_0": int((differences < 0).sum()) / n,
            "method_mean_offer_kwh_per_test_day": float(ours.mean()),
            "reference_mean_offer_kwh_per_test_day": reference_mean,
            "mean_difference_share_of_reference_mean_offer": (
                None if reference_mean == 0 else mean / reference_mean
            ),
            "verdict": verdict(p5, p95),
        }
    )
    return found


def pairs(methods) -> tuple[list, list]:
    """(the pairs against `re-solve`, the other pairs), each (method, reference) (module text)."""
    methods = list(methods)
    against = [(m, RE_SOLVE) for m in methods if m != RE_SOLVE] if RE_SOLVE in methods else []
    others = [m for m in methods if m != RE_SOLVE]
    other = [(b, a) for i, a in enumerate(others) for b in others[i + 1 :]]
    return against, other


def paired(rows, units, variants, tiers, methods, seeds) -> dict:
    """unit -> variant -> tier -> table -> the figures of each pair (module text)."""
    seeds = tuple(seeds)
    offers: dict = {}
    for row in rows:
        key = (row["unit"], row["variant"], row["tier"], row["method"])
        found = offers.setdefault(key, {})
        if row["seed"] in found:
            raise ValueError(f"the split row {key} seed {row['seed']} is listed twice")
        found[row["seed"]] = row[OFFER]
    against, other = pairs(methods)
    out: dict = {}
    for unit in units:
        for variant in variants:
            for tier in tiers:

                def table(chosen, unit=unit, variant=variant, tier=tier):
                    return [
                        {
                            "method": m,
                            "reference": r,
                            **pair_statistics(
                                offers.get((unit, variant, tier, m), {}),
                                offers.get((unit, variant, tier, r), {}),
                                seeds,
                            ),
                        }
                        for m, r in chosen
                    ]

                out.setdefault(unit, {}).setdefault(variant, {})[tier] = {
                    TABLES[0]: table(against),
                    TABLES[1]: table(other),
                }
    return out
