import pytest

from gbpricing.eda import daily_shape_by_season, monthly_base_peak, negative_price_stats, volatility_by_slot


def test_monthly_base_is_mean_and_peak_above_base(hist):
    t = monthly_base_peak(hist)
    jan = hist[hist["month"] == "2023-01-01"]["price_gbp_mwh"].mean()
    assert t.loc["2023-01-01", "base_gbp_mwh"] == pytest.approx(jan)
    assert (t["peak_base_ratio"] > 1).mean() > 0.8


def test_negative_stats_count_matches(hist):
    n = negative_price_stats(hist)
    assert n["negative_periods"].sum() == int((hist["price_gbp_mwh"] < 0).sum())


def test_shape_and_volatility_have_48_slots(hist):
    assert len(daily_shape_by_season(hist)) == 48
    assert len(volatility_by_slot(hist)) == 48