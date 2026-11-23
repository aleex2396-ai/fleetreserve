"""The arithmetic of the ledger of plan T9 (section 5) and outcome O6 (section 11), per day and in
means over days.

T9: V_hind - beta V_cert = (V_hind - V_caus) + (V_caus - V_aff) + (V_aff - V_cert) +
(V_cert - beta V_cert), the prices of real-time decisions, affine rules, unknown cars and
calibration. The letters of O6, per day:

- the ceiling, V_hind;
- R = V_hind - V_caus (real-time decisions), A = V_caus - V_aff (affine rules), and their sum
  R + A = V_hind - V_aff;
- U = V_aff - V_cert (unknown cars);
- K = (1 - beta) V_cert (calibration), with beta the calibrated scale;
- the offer, beta V_cert, as the scoring run recorded it for the day (not recomputed here).

So offer + R + A + U + K = ceiling on every day, and the same holds for means over one set of
days. The identity is checked, not assumed: the offer is the run's recorded number and K is
computed from the run's recorded scale and certified value, so a day whose offer is not
beta V_cert breaks it (`identity_residual`). F (the format loss, plan section 1: the ceiling
minus the largest box of the product in LP (24)'s set) is outside the sum and is not formed here.

`day_terms` gives every term of one day; a term whose inputs are missing (V_aff or V_caus not
computed, or a solve that did not end optimal) is None, never 0. `means` averages the terms over
one set of days and gives each as a share of the mean ceiling of the same days (the ratio of the
means, so the shares of offer, R, A, U and K add up to 1 where they are all given). The block
product's terms (`block_terms`) have no offer in the gate, so only R, A, V_aff - V_diag and the
price of the block (hourly V_hind minus block V_hind) are formed for it.

Tolerances. An identity holds when its residual is at most `IDENTITY_TOLERANCE` times
max(1, |ceiling|). A term the plan's order makes non-negative (R, A, R + A, V_aff - V_diag, F,
the price of the block) breaks the order when it is below `-ORDER_TOLERANCE` times
max(1, |ceiling|); smaller negative values are left as measured, never clipped.
"""

from collections.abc import Sequence

IDENTITY_TOLERANCE = 1e-9
ORDER_TOLERANCE = 1e-6
JOINT = ("offer", "r_plus_a", "u", "k")  # the sum on every day: R and A together
SPLIT = ("offer", "r", "a", "u", "k")  # the sum on a day whose tree value is known
NOT_COMPUTED_BLOCK = "not computed: no block offer in the gate"


def calibration_term(beta: float, certified: float) -> float:
    """K = (1 - beta) V_cert."""
    return (1.0 - beta) * certified


def _difference(high, low):
    return None if high is None or low is None else high - low


def day_terms(
    ceiling: float,
    best_causal: float | None,
    certified: float,
    beta: float,
    offer: float,
    tree: float | None = None,
) -> dict:
    """Every term of one day (module text). `best_causal` is V_aff, `tree` V_caus (None when the
    day's tree was not solved)."""
    return {
        "ceiling": ceiling,
        "best_causal": best_causal,
        "tree": tree,
        "certified": certified,
        "beta": beta,
        "offer": offer,
        "r_plus_a": _difference(ceiling, best_causal),
        "r": _difference(ceiling, tree),
        "a": _difference(tree, best_causal),
        "u": _difference(best_causal, certified),
        "k": calibration_term(beta, certified),
    }


def identity_residual(terms: dict, names: Sequence[str]) -> float | None:
    """The sum of the terms `names` minus the ceiling, or None when a term is missing."""
    parts = [terms[name] for name in names]
    if any(part is None for part in parts) or terms["ceiling"] is None:
        return None
    return sum(parts) - terms["ceiling"]


def within(value: float, ceiling: float, tolerance: float) -> bool:
    """Whether `value` is at most `tolerance` times max(1, |ceiling|) in size."""
    return abs(value) <= tolerance * max(1.0, abs(ceiling))


def below_order(value: float | None, ceiling: float) -> bool:
    """Whether a term the plan's order makes non-negative breaks that order (module text)."""
    return value is not None and value < -ORDER_TOLERANCE * max(1.0, abs(ceiling))


def _mean(values) -> float | None:
    values = list(values)
    return float(sum(values) / len(values)) if values else None


def means(days: Sequence[dict], names: Sequence[str], ceiling: str = "ceiling") -> dict:
    """The mean of each term `names` and of the ceiling over `days` (every day must hold all of
    them), and each term's share of the mean ceiling (None when that mean is not above 0)."""
    for day in days:
        if any(day[name] is None for name in (ceiling, *names)):
            raise ValueError("every day of a mean must hold every term of it")
    mean_ceiling = _mean(day[ceiling] for day in days)
    found = {name: _mean(day[name] for day in days) for name in names}
    share = {
        name: None
        if found[name] is None or mean_ceiling is None or mean_ceiling <= 0
        else found[name] / mean_ceiling
        for name in names
    }
    return {
        "days": len(days),
        "mean_ceiling_kwh": mean_ceiling,
        "mean_kwh": found,
        "share_of_ceiling": share,
    }


def identity_of_means(block: dict, names: Sequence[str]) -> float | None:
    """The sum of the mean terms `names` of a `means` result minus its mean ceiling."""
    if block["days"] == 0:
        return None
    return sum(block["mean_kwh"][name] for name in names) - block["mean_ceiling_kwh"]


def block_terms(
    hourly_ceiling: float | None,
    block_ceiling: float | None,
    block_tree: float | None,
    block_affine: float | None,
    block_diagonal: float | None,
) -> dict:
    """The block product's terms of one day: R, A, V_aff - V_diag and the price of the block."""
    return {
        "hourly_ceiling": hourly_ceiling,
        "ceiling": block_ceiling,
        "r": _difference(block_ceiling, block_tree),
        "a": _difference(block_tree, block_affine),
        "affine_minus_diagonal": _difference(block_affine, block_diagonal),
        "price_of_the_block": _difference(hourly_ceiling, block_ceiling),
        "u": NOT_COMPUTED_BLOCK,
        "k": NOT_COMPUTED_BLOCK,
    }
