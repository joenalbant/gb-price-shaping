import pandas as pd

from gbpricing.calendar import settlement_index
from gbpricing.features import add_time_features
from gbpricing.holidays import parse_bank_holidays

HOLIDAYS = ["2024-05-06", "2024-12-25"]


def feat(start, end, **kw):
    return add_time_features(settlement_index(start, end), HOLIDAYS, **kw)


def test_parse_bank_holidays():
    payload = {"england-and-wales": {"events": [{"date": "2024-12-25"}, {"date": "2024-05-06"}]}}
    out = parse_bank_holidays(payload)
    assert list(out) == [pd.Timestamp("2024-05-06"), pd.Timestamp("2024-12-25")]


def test_slots_are_local_half_hours_in_summer():
    f = feat("2024-06-03", "2024-06-03")  # Monday, BST
    assert f["slot"].tolist() == list(range(48))
    assert f.loc[f["settlement_period"] == 1, "local_start"].iat[0].hour == 0


def test_peak_is_weekday_0700_to_1900_local():
    f = feat("2024-06-03", "2024-06-09")  # Mon-Sun
    weekday = f[f["weekday"] == 0]
    assert weekday["is_peak"].sum() == 24  # 12 hours x 2
    assert weekday.loc[weekday["is_peak"], "slot"].min() == 14  # 07:00
    assert weekday.loc[weekday["is_peak"], "slot"].max() == 37  # 18:30
    assert not f.loc[f["weekday"] >= 5, "is_peak"].any()


def test_bank_holiday_is_weekend_day_type():
    f = feat("2024-05-06", "2024-05-06")
    assert (f["day_type"] == "weekend").all()
    assert f["is_peak"].sum() == 24  # included in peak by default
    f2 = feat("2024-05-06", "2024-05-06", peak_includes_bank_holidays=False)
    assert f2["is_peak"].sum() == 0


def test_clock_change_days_keep_slots_consistent():
    spring = feat("2024-03-31", "2024-03-31")
    autumn = feat("2024-10-27", "2024-10-27")
    assert set(range(48)) - set(spring["slot"]) == {2, 3}  # 01:00-02:00 skipped
    assert autumn["slot"].value_counts()[2] == 2            # 01:00-01:30 twice
    assert spring["is_peak"].sum() == 0 and autumn["is_peak"].sum() == 0  # Sundays