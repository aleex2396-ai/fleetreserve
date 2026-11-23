"""The holiday flag of the learned models' features (`learn_features.holiday_flags`), which needs
the `holidays` package of the learn group: New Year's Day is a public holiday in Norway and in
the United States (California), the day after is not, Norway's Constitution Day is one in
Norway only, and a site with no country given is refused.
"""

import pandas as pd
import pytest

from fleetreserve.learn_features import holiday_flags

COUNTRIES = {"NO-site": ("NO",), "US-site": ("US", "CA")}


def test_public_holidays_by_site():
    table = pd.DataFrame(
        {
            "site": ["NO-site", "NO-site", "US-site", "US-site", "NO-site", "US-site"],
            "date": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-01",
                "2020-01-02",
                "2019-05-17",
                "2019-05-17",
            ],
        }
    )
    flagged = holiday_flags(table, COUNTRIES)
    assert flagged["public_holiday"].tolist() == [True, False, True, False, True, False]
    assert list(flagged.columns) == ["site", "date", "public_holiday"]


def test_a_site_without_a_country_is_refused():
    table = pd.DataFrame({"site": ["other"], "date": ["2020-01-01"]})
    with pytest.raises(ValueError, match="no country"):
        holiday_flags(table, COUNTRIES)
