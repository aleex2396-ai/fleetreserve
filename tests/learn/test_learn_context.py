"""The two context columns of the learned models (`learn_features.context_columns`), which need
the `holidays` package of the learn group, and the committed site mappings.

1. `public_holiday` is true on a known public holiday and false the day after, for a site in the
   United States (California and Colorado), in Norway, in the Netherlands and in South Korea, by
   the committed mapping; a holiday of one country is none in another.
2. `site_type` follows plan section 2 for every mapped site: the ACN sites, NREL and a.s.r. are
   workplaces (1), every Norwegian site and every Korean portfolio the other type (0).
3. The two mappings cover the same sites: every training site, the sites of the development
   units, the two Norwegian sites kept for the final run, the a.s.r. site and the five Korean
   portfolios under the site names `load_korea` writes (`load_korea.UNITS`), each in South Korea;
   the data-set name `korea` is no site.
4. The columns depend on `site` and `date` only: a frame with nothing else gives them, and any
   other column of the feature table, changed, changes neither.
5. A site in neither mapping is refused, as is a site with a type but no country.
6. They are not columns of `build_table`, and not in the older column lists.
"""

import numpy as np
import pandas as pd
import pytest
from test_learn_features import _history_unit

from fleetreserve import learn_features as lf
from fleetreserve import load_acn, load_asr, load_korea, load_norway, load_nrel
from fleetreserve.truth_build import TRAIN_UNITS

HOLIDAYS = [  # site, a public holiday there, the day after
    ("caltech", "2019-07-04", "2019-07-05"),
    ("nrel", "2019-11-28", "2019-11-29"),
    ("OSL_S", "2019-05-17", "2019-05-18"),
    ("asr", "2023-04-27", "2023-04-28"),
    *((site, "2021-08-15", "2021-08-17") for site in load_korea.UNITS),
]


@pytest.mark.parametrize("site, holiday, after", HOLIDAYS)
def test_a_known_public_holiday_is_flagged_and_the_day_after_is_not(site, holiday, after):
    frame = pd.DataFrame({"site": [site, site], "date": [holiday, after]})
    assert lf.context_columns(frame)["public_holiday"].tolist() == [True, False]


def test_south_korea_day_after_liberation_day_2019_is_no_holiday():
    frame = pd.DataFrame({"site": ["korea-work"] * 2, "date": ["2019-08-15", "2019-08-16"]})
    assert lf.context_columns(frame)["public_holiday"].tolist() == [True, False]


def test_a_holiday_of_one_country_is_none_in_another():
    dates = {"2019-07-04": "caltech", "2019-05-17": "OSL_S", "2023-04-27": "asr"}
    for date, home in dates.items():
        sites = ["caltech", "nrel", "OSL_S", "asr", "korea-home"]
        frame = pd.DataFrame({"site": sites, "date": [date] * len(sites)})
        flags = dict(zip(sites, lf.context_columns(frame)["public_holiday"], strict=True))
        assert flags[home]
        assert flags["nrel"] == (home == "caltech")  # one country, another state
        assert sum(flags.values()) == (2 if home == "caltech" else 1)


def test_the_subdivision_is_applied():
    # the day after Thanksgiving 2019 is a holiday of California only, the first Monday of
    # October 2021 of Colorado only, in the pinned package's calendars
    frame = pd.DataFrame(
        {"site": ["caltech", "nrel"] * 2, "date": ["2019-11-29"] * 2 + ["2021-10-04"] * 2}
    )
    assert lf.context_columns(frame)["public_holiday"].tolist() == [True, False, False, True]


def test_the_site_type_follows_plan_section_2_for_every_mapped_site():
    workplaces = {*load_acn.SITES, load_nrel.SITE, load_asr.SITE}
    assert workplaces == {"caltech", "jpl", "office001", "nrel", "asr"}
    for site in load_korea.UNITS:
        assert lf.SITE_TYPES[site] == lf.OTHER_SITE
    for site, kind in lf.SITE_TYPES.items():
        assert kind == (lf.WORKPLACE if site in workplaces else lf.OTHER_SITE), site
    assert (lf.WORKPLACE, lf.OTHER_SITE) == (1, 0)
    frame = pd.DataFrame({"site": list(lf.SITE_TYPES), "date": "2019-03-05"})
    assert lf.context_columns(frame)["site_type"].tolist() == list(lf.SITE_TYPES.values())
    assert not lf.context_columns(frame)["public_holiday"].any()


def test_the_mappings_cover_the_same_sites_of_every_role():
    norway = {*load_norway.SITES["train"], *load_norway.SITES["development"], "OSL_T", "TRO_R"}
    korea = {"korea-home", "korea-lodging", "korea-work", "korea-public-parking", "korea-visits"}
    assert set(load_korea.UNITS) == korea
    expected = {*load_acn.SITES, load_nrel.SITE, *norway, "asr", *korea}
    assert set(lf.SITE_TYPES) == set(lf.SITE_COUNTRIES) == expected
    assert set(TRAIN_UNITS) <= expected  # every training unit is a site of its own name
    assert len(norway) == 12 == len(lf.NORWAY_SITES)
    for site in norway:
        assert lf.SITE_COUNTRIES[site] == ("NO",)
    for site in load_acn.SITES:
        assert lf.SITE_COUNTRIES[site] == ("US", "CA")
    assert lf.SITE_COUNTRIES["nrel"] == ("US", "CO")
    assert lf.SITE_COUNTRIES["asr"] == ("NL",)
    for site in korea:
        assert lf.SITE_COUNTRIES[site] == ("KR",)
    assert "korea" not in lf.SITE_TYPES and "korea" not in lf.SITE_COUNTRIES


def test_the_columns_depend_on_site_and_date_only():
    cars = _history_unit()
    cars["site"] = "jpl"
    table = lf.build_table(cars, rho=0.0)
    base = lf.context_columns(table)
    assert list(base.columns) == list(lf.CONTEXT_COLUMNS) == ["public_holiday", "site_type"]
    assert base.index.equals(table.index)
    assert base["public_holiday"].any() and not base["public_holiday"].all()  # 1 January 2020
    pd.testing.assert_frame_equal(base, lf.context_columns(table[["site", "date"]]))
    changed = table.copy()
    for column in table.columns:
        if column not in ("site", "date"):
            changed[column] = changed[column].iloc[::-1].to_numpy()
    pd.testing.assert_frame_equal(base, lf.context_columns(changed))
    with_them = lf.with_context(table)
    assert list(with_them.columns) == [*table.columns, *lf.CONTEXT_COLUMNS]
    pd.testing.assert_frame_equal(with_them[list(lf.CONTEXT_COLUMNS)], base)
    # values already in the frame are set anew, never read
    wrong = table.assign(public_holiday=True, site_type=7)
    pd.testing.assert_frame_equal(lf.with_context(wrong)[list(lf.CONTEXT_COLUMNS)], base)
    # a changed site or date does change them
    moved = table.assign(site="OSL_S")
    assert (lf.context_columns(moved)["site_type"] == 0).all()
    assert (base["site_type"] == 1).all()


def test_a_site_in_neither_mapping_is_refused():
    frame = pd.DataFrame({"site": ["somewhere"], "date": ["2019-07-04"]})
    with pytest.raises(ValueError, match="no site type"):
        lf.context_columns(frame)
    with pytest.raises(ValueError, match="no country"):
        lf.context_columns(frame, types={"somewhere": 1})
    given = lf.context_columns(frame, countries={"somewhere": ("US",)}, types={"somewhere": 1})
    assert given.to_numpy().tolist() == [[True, 1]]


def test_an_empty_frame_gives_no_rows():
    frame = pd.DataFrame({"site": pd.Series(dtype=str), "date": pd.Series(dtype=str)})
    assert len(lf.context_columns(frame)) == 0


def test_the_columns_are_not_in_the_feature_table_or_the_older_lists():
    table = lf.build_table(_history_unit(), rho=0.0)
    older = (*lf.FEATURE_COLUMNS, *lf.TAIL_HISTORY_COLUMNS, *lf.DECLARED_COLUMNS)
    for name in lf.CONTEXT_COLUMNS:
        assert name not in table.columns
        assert name not in older
    assert np.array_equal(lf.FEATURE_COLUMNS[:3], ("weekday", "month", "day_of_year"))
