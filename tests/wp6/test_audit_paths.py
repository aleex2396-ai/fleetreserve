"""Plan section 1: the audited set, 8 fixed paths a day.

1. Rows 1 to 3 are all -1, all +1 and the alternating path starting at +1; rows 4 to 8 are vertex
   paths (every entry -1 or +1).
2. The same unit and date always give the same paths; across the 30 days of a month at one unit,
   with 6 service hours, the seeded rows differ between days (`MIN_DISTINCT` distinct sets of five).
3. The seed is the first 8 bytes of SHA-256 of "<unit>|<date>": checked by hand for one day.
"""

import datetime
import hashlib

import numpy as np

from fleetreserve.audit_paths import audited_paths, day_seed

DATE = datetime.date(2019, 6, 3)
MIN_DISTINCT = 30


def test_the_eight_paths():
    paths = audited_paths("site-a", DATE, 5)
    assert paths.shape == (8, 5)
    np.testing.assert_array_equal(paths[0], -np.ones(5))
    np.testing.assert_array_equal(paths[1], np.ones(5))
    np.testing.assert_array_equal(paths[2], [1, -1, 1, -1, 1])
    assert set(np.unique(paths)) == {-1.0, 1.0}


def test_same_day_same_paths_and_days_differ():
    again = audited_paths("site-a", DATE, 6)
    np.testing.assert_array_equal(audited_paths("site-a", DATE, 6), again)
    month = [audited_paths("site-a", DATE.replace(day=d), 6)[3:].tobytes() for d in range(1, 31)]
    assert len(set(month)) >= MIN_DISTINCT
    assert audited_paths("site-b", DATE, 6)[3:].tobytes() != again[3:].tobytes()


def test_seed_by_hand():
    digest = hashlib.sha256(b"site-a|2019-06-03").digest()
    assert day_seed("site-a", DATE) == int.from_bytes(digest[:8], "big")
