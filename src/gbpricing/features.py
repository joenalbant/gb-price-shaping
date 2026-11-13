"""Time features used by every later step: month, day type, local slot, peak flag.

Two different ideas are kept apart on purpose:

* ``day_type`` is about how prices *behave*: bank holidays look like weekends,
  so they are grouped with weekends when estimating shapes.
* ``is_peak`` is the *product* definition: GB peak = weekdays 07:00-19:00 local
  time. Whether bank holidays are inside the peak product is a market
  convention; it is a parameter here (default: included, like other weekdays).
  State the convention you use in the README.

``slot`` is the local half-hour of the day (0 = 00:00-00:30 ... 47 = 23:30-24:00).
Shapes are keyed on slot rather than settlement period because settlement
periods shift by one hour on clock-change days (SP3 is 01:00 local on a normal
day but 02:00 local on the spring day). On the 50-period autumn day two
periods share slots 2 and 3; on the 46-period spring day slots 2 and 3 are absent.
"""
from __future__ import annotations

import pandas as pd

from .calendar import LONDON

PEAK_START_HOUR = 7
PEAK_END_HOUR = 19


def add_time_features(
    df: pd.DataFrame,
    bank_holidays=(),
    peak_includes_bank_holidays: bool = True,
) -> pd.DataFrame:
    """Return a copy of a half-hourly table with time feature columns added.

    Needs columns ``utc_start`` and ``settlement_date``. Adds:
    local_start, year, month (first day of the delivery month), weekday (0=Mon),
    is_bank_holiday, day_type ('weekday' or 'weekend'), slot (0-47), is_peak.
    """
    out = df.copy()
    holidays = pd.DatetimeIndex(pd.to_datetime(list(bank_holidays))).normalize()
    local = pd.to_datetime(out["utc_start"], utc=True).dt.tz_convert(LONDON)
    sd = pd.to_datetime(out["settlement_date"])

    out["local_start"] = local
    out["year"] = sd.dt.year
    out["month"] = sd.dt.to_period("M").dt.to_timestamp()
    out["weekday"] = sd.dt.day_of_week
    out["is_bank_holiday"] = sd.isin(holidays)
    weekend = out["weekday"] >= 5
    out["day_type"] = (weekend | out["is_bank_holiday"]).map({True: "weekend", False: "weekday"})
    out["slot"] = local.dt.hour * 2 + local.dt.minute // 30

    in_hours = (local.dt.hour >= PEAK_START_HOUR) & (local.dt.hour < PEAK_END_HOUR)
    peak_day = ~weekend if peak_includes_bank_holidays else ~(weekend | out["is_bank_holiday"])
    out["is_peak"] = in_hours & peak_day
    return out