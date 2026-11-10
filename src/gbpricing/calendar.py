"""GB settlement calendar.

A GB settlement day runs from local (Europe/London) midnight to the next local
midnight and is split into 30-minute settlement periods (SPs):

* normal day                  -> 48 periods
* spring clock change (March) -> 46 periods (01:00 GMT jumps to 02:00 BST)
* autumn clock change (Oct)   -> 50 periods (01:00-02:00 local happens twice)

During BST, SP1 starts at 23:00 UTC on the previous calendar day.
All conversions go through UTC so that there is never an ambiguous timestamp.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

LONDON = "Europe/London"
PERIOD = pd.Timedelta(minutes=30)


def _to_date(d) -> dt.date:
    return pd.Timestamp(d).date()


def day_start_utc(settlement_date) -> pd.Timestamp:
    """UTC instant at which the settlement day starts (local midnight)."""
    local_midnight = pd.Timestamp(_to_date(settlement_date)).tz_localize(LONDON)
    return local_midnight.tz_convert("UTC")


def periods_in_day(settlement_date) -> int:
    """Number of settlement periods in a day: 46, 48 or 50."""
    d = _to_date(settlement_date)
    start = day_start_utc(d)
    end = day_start_utc(d + dt.timedelta(days=1))
    return int((end - start) / PERIOD)


def sp_to_utc(settlement_date, settlement_period: int) -> pd.Timestamp:
    """UTC start time of a settlement period."""
    n = periods_in_day(settlement_date)
    if not 1 <= settlement_period <= n:
        raise ValueError(
            f"{settlement_date} has {n} periods; got settlement_period={settlement_period}"
        )
    return day_start_utc(settlement_date) + (settlement_period - 1) * PERIOD


def utc_to_sp(ts) -> tuple[dt.date, int]:
    """(settlement_date, settlement_period) for a UTC period start time."""
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    if ts != ts.floor("30min"):
        raise ValueError(f"{ts} is not on a 30-minute boundary")
    d = ts.tz_convert(LONDON).date()
    sp = int((ts - day_start_utc(d)) / PERIOD) + 1
    return d, sp


def settlement_index(start_date, end_date) -> pd.DataFrame:
    """Every settlement period between two dates (inclusive), one row each.

    Columns: utc_start, settlement_date, settlement_period.
    This is the 'expected' grid that downloaded data is checked against.
    """
    rows = []
    for d in pd.date_range(start_date, end_date, freq="D").date:
        start = day_start_utc(d)
        for sp in range(1, periods_in_day(d) + 1):
            rows.append((start + (sp - 1) * PERIOD, d, sp))
    df = pd.DataFrame(rows, columns=["utc_start", "settlement_date", "settlement_period"])
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    return df