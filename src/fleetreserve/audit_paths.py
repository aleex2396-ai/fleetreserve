"""The audited set of a day (plan section 1): 8 fixed request paths.

Rows, each of `n` entries of -1 or +1 (`zeta_t` for service hour `t`):

1. all -1 (every request at the low end of the box);
2. all +1;
3. alternating, starting at +1;
4. to 8. five vertex paths drawn at random, seeded from the day's unit and date, so the same day
   always gets the same paths. They are drawn independently and may repeat each other or the first
   three (with `n` service hours there are only `2^n` vertex paths).

So for a short window the 8 rows are fewer distinct paths: at most 2 for `n = 1` (rows 2 and 3
coincide), at most 4 for `n = 2`, and for `n = 3` often fewer than 8. The set is kept as plan
section 1 states it, repeats included; a caller counting paths counts rows.

The seed is the first 8 bytes of the SHA-256 of `"<unit>|<date in ISO form>"`, read as an
unsigned integer, given to numpy's default generator.
"""

import datetime
import hashlib

import numpy as np

N_SEEDED = 5


def day_seed(unit: str, date: datetime.date) -> int:
    digest = hashlib.sha256(f"{unit}|{date.isoformat()}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def audited_paths(unit: str, date: datetime.date, n_service: int) -> np.ndarray:
    """`(8, n_service)`: the day's audited paths in the order of the module text."""
    if n_service < 1:
        raise ValueError(f"n_service must be at least 1, got {n_service}")
    alternating = np.where(np.arange(n_service) % 2 == 0, 1.0, -1.0)
    rng = np.random.default_rng(day_seed(unit, date))
    seeded = np.where(rng.integers(0, 2, size=(N_SEEDED, n_service)) == 1, 1.0, -1.0)
    fixed = np.stack([-np.ones(n_service), np.ones(n_service), alternating])
    return np.vstack([fixed, seeded])
