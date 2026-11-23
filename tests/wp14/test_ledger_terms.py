"""The arithmetic of the ledger (`ledger_terms`), with numbers worked by hand.

1. One day: ceiling 3, V_caus 2.5, V_aff 2, V_cert 1.6, scale 0.75, offer 1.2 (= 0.75 x 1.6).
   R = 3 - 2.5 = 0.5, A = 2.5 - 2 = 0.5, R + A = 1, U = 2 - 1.6 = 0.4, K = 0.25 x 1.6 = 0.4,
   and 1.2 + 0.5 + 0.5 + 0.4 + 0.4 = 3: both identities have residual 0.
2. An offer that is not the scale times V_cert (1.3) leaves a residual of 0.1 in both identities.
3. A missing V_caus leaves R and A missing (None, never 0) and the split identity open, while
   R + A and the joint identity stay; a missing V_aff leaves A, R + A and U missing.
4. A scale above 1 gives a negative K, as measured: scale 1.25, V_cert 1.6, K = -0.4.
5. Means over two days with ceilings 3 and 1: the mean ceiling 2, each mean term and its share of
   the mean ceiling (the ratio of the means), the shares of the offer and the terms adding up to 1,
   the identity of the means 0; a share is None when the mean ceiling is 0; a day lacking a term
   is refused.
6. The block product: hourly ceiling 3, block ceiling 2, block V_caus 1.5, V_aff 1, V_diag 0.5:
   R 0.5, A 0.5, V_aff - V_diag 0.5, the price of the block 3 - 2 = 1, U and K not computed.
7. The tolerances: a residual of 1e-10 on a ceiling of 3 is within the identity's; 4e-9 is not;
   a term of -1e-7 keeps the order, -1e-5 breaks it.
"""

import pytest

from fleetreserve import ledger_terms as lt


def test_one_day_worked_by_hand():
    terms = lt.day_terms(3.0, 2.0, 1.6, 0.75, 1.2, tree=2.5)
    assert terms["r"] == pytest.approx(0.5)
    assert terms["a"] == pytest.approx(0.5)
    assert terms["r_plus_a"] == pytest.approx(1.0)
    assert terms["u"] == pytest.approx(0.4)
    assert terms["k"] == pytest.approx(0.4)
    assert lt.identity_residual(terms, lt.JOINT) == pytest.approx(0.0, abs=1e-12)
    assert lt.identity_residual(terms, lt.SPLIT) == pytest.approx(0.0, abs=1e-12)


def test_an_offer_that_is_not_the_scale_times_the_certified_value_breaks_the_identity():
    terms = lt.day_terms(3.0, 2.0, 1.6, 0.75, 1.3, tree=2.5)
    assert lt.identity_residual(terms, lt.JOINT) == pytest.approx(0.1)
    assert lt.identity_residual(terms, lt.SPLIT) == pytest.approx(0.1)
    assert not lt.within(lt.identity_residual(terms, lt.JOINT), 3.0, lt.IDENTITY_TOLERANCE)


def test_a_missing_value_leaves_its_terms_missing_never_zero():
    no_tree = lt.day_terms(3.0, 2.0, 1.6, 0.75, 1.2)
    assert no_tree["r"] is None and no_tree["a"] is None
    assert no_tree["r_plus_a"] == pytest.approx(1.0)
    assert lt.identity_residual(no_tree, lt.SPLIT) is None
    assert lt.identity_residual(no_tree, lt.JOINT) == pytest.approx(0.0, abs=1e-12)
    no_affine = lt.day_terms(3.0, None, 1.6, 0.75, 1.2, tree=2.5)
    assert no_affine["a"] is None and no_affine["r_plus_a"] is None and no_affine["u"] is None
    assert no_affine["r"] == pytest.approx(0.5)
    assert lt.identity_residual(no_affine, lt.JOINT) is None


def test_a_scale_above_one_gives_a_negative_calibration_term():
    terms = lt.day_terms(3.0, 2.0, 1.6, 1.25, 2.0, tree=2.5)
    assert terms["k"] == pytest.approx(-0.4)
    assert lt.identity_residual(terms, lt.SPLIT) == pytest.approx(0.0, abs=1e-12)


def test_means_and_shares_of_the_ceiling():
    days = [
        lt.day_terms(3.0, 2.0, 1.6, 0.75, 1.2, tree=2.5),
        lt.day_terms(1.0, 1.0, 0.8, 0.75, 0.6, tree=1.0),
    ]
    block = lt.means(days, lt.SPLIT)
    assert block["days"] == 2
    assert block["mean_ceiling_kwh"] == pytest.approx(2.0)
    expected = {"offer": 0.9, "r": 0.25, "a": 0.25, "u": 0.3, "k": 0.3}
    for name, value in expected.items():
        assert block["mean_kwh"][name] == pytest.approx(value)
        assert block["share_of_ceiling"][name] == pytest.approx(value / 2.0)
    assert sum(block["share_of_ceiling"].values()) == pytest.approx(1.0)
    assert lt.identity_of_means(block, lt.SPLIT) == pytest.approx(0.0, abs=1e-12)
    empty = lt.means([], lt.SPLIT)
    assert empty["days"] == 0 and lt.identity_of_means(empty, lt.SPLIT) is None
    zero = lt.means([lt.day_terms(0.0, 0.0, 0.0, 0.5, 0.0, tree=0.0)], lt.SPLIT)
    assert zero["share_of_ceiling"]["offer"] is None
    with pytest.raises(ValueError):
        lt.means([lt.day_terms(3.0, 2.0, 1.6, 0.75, 1.2)], lt.SPLIT)


def test_the_block_product_worked_by_hand():
    terms = lt.block_terms(3.0, 2.0, 1.5, 1.0, 0.5)
    assert terms["r"] == pytest.approx(0.5)
    assert terms["a"] == pytest.approx(0.5)
    assert terms["affine_minus_diagonal"] == pytest.approx(0.5)
    assert terms["price_of_the_block"] == pytest.approx(1.0)
    assert terms["u"] == terms["k"] == lt.NOT_COMPUTED_BLOCK
    missing = lt.block_terms(3.0, 2.0, None, 1.0, None)
    assert missing["r"] is None and missing["a"] is None
    assert missing["affine_minus_diagonal"] is None


def test_the_tolerances():
    assert lt.within(1e-10, 3.0, lt.IDENTITY_TOLERANCE)
    assert not lt.within(4e-9, 3.0, lt.IDENTITY_TOLERANCE)
    assert not lt.below_order(-1e-7, 3.0)
    assert lt.below_order(-1e-5, 3.0)
    assert not lt.below_order(None, 3.0)
