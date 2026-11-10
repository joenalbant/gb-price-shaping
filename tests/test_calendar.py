import datetime as dt

import pandas as pd
import pytest

from gbpricing.calendar import (
    periods_in_day,
    settlement_index,
    sp_to_utc,
    utc_to_sp,
)

SPRING = dt.date(2024, 3, 31)  # clocks go forward
AUTUMN = dt.date(2024, 10, 27)  # clocks go back


@pytest.mark.parametrize(
    "day, expected",
    [(SPRING, 46), (AUTUMN, 50), (dt.date(2024, 6, 1), 48), (dt.date(2024, 1, 15), 48),
     (dt.date(2025, 3, 30), 46), (dt.date(2025, 10, 26), 50)],
)
def test_periods_in_day(day, expected):
    assert periods_in_day(day) == expected


def test_sp1_in_winter_is_midnight_utc():
    assert sp_to_utc("2024-01-15", 1) == pd.Timestamp("2024-01-15 00:00", tz="UTC")


def test_sp1_in_summer_is_2300_utc_previous_day():
    assert sp_to_utc("2024-06-01", 1) == pd.Timestamp("2024-05-31 23:00", tz="UTC")


def test_spring_day_skips_local_0100_to_0200():
    # SP3 starts 01:00 UTC = 02:00 BST: the local 01:00-02:00 hour never happens
    assert sp_to_utc(SPRING, 3) == pd.Timestamp("2024-03-31 01:00", tz="UTC")
    assert sp_to_utc(SPRING, 3).tz_convert("Europe/London").hour == 2


def test_autumn_day_repeats_local_0100_to_0200():
    local_hours = [sp_to_utc(AUTUMN, sp).tz_convert("Europe/London").hour for sp in range(1, 51)]
    assert local_hours.count(1) == 4  # four half-hours start at local 01:xx


def test_last_period_ends_at_next_day_start():
    for day in (SPRING, AUTUMN, dt.date(2024, 6, 1)):
        n = periods_in_day(day)
        assert sp_to_utc(day, n) + pd.Timedelta(minutes=30) == sp_to_utc(day + dt.timedelta(days=1), 1)


def test_invalid_period_raises():
    with pytest.raises(ValueError):
        sp_to_utc(SPRING, 47)
    with pytest.raises(ValueError):
        sp_to_utc("2024-06-01", 0)


def test_round_trip_over_two_years():
    idx = settlement_index("2023-01-01", "2024-12-31")
    for ts, d, sp in idx.sample(2000, random_state=0).itertuples(index=False):
        assert utc_to_sp(ts) == (d.date(), sp)


def test_settlement_index_is_continuous_and_unique():
    idx = settlement_index("2024-01-01", "2024-12-31")
    assert idx["utc_start"].is_unique
    assert (idx["utc_start"].diff().dropna() == pd.Timedelta(minutes=30)).all()
    # 2024 is a leap year: 366 days * 48, -2 in spring, +2 in autumn
    assert len(idx) == 366 * 48