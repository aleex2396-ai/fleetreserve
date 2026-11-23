"""The seeded sample of development counting days for the block-product comparison (plan 13 WP2).

Plan 13 WP2: "a block-product sample: tree and affine against hindsight on 50 development days".
The plan does not say how the 50 days are spread over the development units, nor how they are
drawn. Two readings, marked as such:

- Spread (reading): proportional to each unit's number of counting days, at least one day per unit,
  the total exactly the sample size. The shares are rounded by the largest remainder; a tie goes to
  the unit whose name sorts first. The alternative is an equal number per unit.
- Draw (reading): within a unit, the days with the smallest ranks of `day_split.split_days` (a
  SHA-256 hash of the unit name, the seed and the day's text) are drawn, so the draw depends on
  the unit, the seed and the set of days, and on nothing else (not on the order the days are
  given in). A day is named by its site and date (`day_text`), since a pooled unit holds several
  sites on one date.

A unit with fewer counting days than its share contributes all of them and the rest goes to the
other units; fewer counting days than the sample size in total is a refusal (`ValueError`).
"""

from collections.abc import Mapping, Sequence

from fleetreserve.day_split import split_days

SAMPLE_SIZE = 50
DEFAULT_SEED = 2026


def day_text(site: str, date: str) -> str:
    """The text a day is hashed and listed by."""
    return f"{site} {date}"


def allocate(counts: Mapping[str, int], size: int = SAMPLE_SIZE) -> dict[str, int]:
    """Days per unit: proportional to `counts`, at least 1 and at most the unit's count, summing
    to `size`. Raises `ValueError` when the units hold fewer than `size` days or a unit has none."""
    if size < len(counts) or not counts:
        raise ValueError("the sample must hold at least one day per unit")
    if any(n < 1 for n in counts.values()):
        raise ValueError("a unit with no counting day cannot give a sample day")
    total = sum(counts.values())
    if total < size:
        raise ValueError(f"{total} counting days in all, fewer than the sample size {size}")
    quota = {unit: size * n / total for unit, n in counts.items()}
    share = {unit: min(max(1, int(quota[unit])), counts[unit]) for unit in counts}
    names = sorted(counts)
    # The minimum of one day per unit can push the sum over the size: take from the most over.
    while sum(share.values()) > size:
        unit = max(
            (u for u in names if share[u] > 1), key=lambda u: (share[u] - quota[u], -names.index(u))
        )
        share[unit] -= 1
    while sum(share.values()) < size:
        unit = max(
            (u for u in names if share[u] < counts[u]),
            key=lambda u: (quota[u] - share[u], -names.index(u)),
        )
        share[unit] += 1
    return share


def draw(unit: str, seed: int, days: Sequence[str], count: int) -> tuple[str, ...]:
    """`count` of a unit's `days` (texts), in day order, drawn by the seeded hash ranking."""
    ordered = tuple(sorted(days))
    if count >= len(ordered):
        return ordered
    return split_days(unit, seed, ordered, n_calibration=count, min_days=count + 1).calibration


def sample_days(
    days: Mapping[str, Sequence[str]], seed: int = DEFAULT_SEED, size: int = SAMPLE_SIZE
) -> dict[str, tuple[str, ...]]:
    """The sampled day texts of each unit; `days[unit]` lists the unit's counting days (texts)."""
    shares = allocate({unit: len(listed) for unit, listed in days.items()}, size)
    return {unit: draw(unit, seed, days[unit], shares[unit]) for unit in sorted(days)}
