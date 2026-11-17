import pytest

from gbpricing.shaping import shape_factors_by_year, smooth_shape_factors, stability


def test_one_factor_per_year_month_daytype_slot(hist):
    sf = shape_factors_by_year(hist)
    assert len(sf) == 2 * 12 * 2 * 48


def test_ratios_average_to_one_within_each_month(hist):
    d = hist.assign(r=hist["price_gbp_mwh"] / hist.groupby("month")["price_gbp_mwh"].transform("mean"))
    assert d.groupby("month")["r"].mean().to_numpy() == pytest.approx(1.0)


def test_smoothed_shape_is_complete(shape):
    assert len(shape) == 12 * 2 * 48 and shape["shape_factor"].notna().all()


def test_synthetic_shapes_are_stable_across_years(hist):
    st = stability(shape_factors_by_year(hist))
    assert (st["mean_pair_corr"] > 0.8).all()


def test_additive_method_runs(hist):
    sf = shape_factors_by_year(hist, method="additive")
    assert sf["shape_factor"].abs().max() < 200