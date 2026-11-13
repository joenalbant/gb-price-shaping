"""Step 2: exploratory analysis. Each function returns a table; notebooks draw the charts.

All functions expect a table that has been through ``add_time_features`` and has a
``price_gbp_mwh`` column. Missing prices (NaN) are ignored by pandas averages.
"""
from __future__ import annotations

import pandas as pd

SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
           6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}


def monthly_base_peak(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly baseload, peak and off-peak averages (£/MWh) and the peak/base ratio."""
    g = df.groupby("month")
    out = pd.DataFrame({
        "base_gbp_mwh": g["price_gbp_mwh"].mean(),
        "peak_gbp_mwh": df[df["is_peak"]].groupby("month")["price_gbp_mwh"].mean(),
        "offpeak_gbp_mwh": df[~df["is_peak"]].groupby("month")["price_gbp_mwh"].mean(),
    })
    out["peak_base_ratio"] = out["peak_gbp_mwh"] / out["base_gbp_mwh"]
    return out


def daily_shape_by_season(df: pd.DataFrame) -> pd.DataFrame:
    """Average price per local half-hour slot, one column per season and day type."""
    season = df["month"].dt.month.map(SEASONS)
    return (
        df.assign(season=season)
        .pivot_table(index="slot", columns=["season", "day_type"], values="price_gbp_mwh", aggfunc="mean")
    )


def negative_price_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per month: number and share of periods with a negative price, and the lowest price."""
    valid = df.dropna(subset=["price_gbp_mwh"])
    g = valid.groupby("month")["price_gbp_mwh"]
    return pd.DataFrame({
        "negative_periods": g.apply(lambda s: int((s < 0).sum())),
        "negative_share": g.apply(lambda s: (s < 0).mean()),
        "min_gbp_mwh": g.min(),
    })


def volatility_by_slot(df: pd.DataFrame) -> pd.DataFrame:
    """Price volatility per slot.

    ``std_gbp_mwh`` is the spread of prices around their monthly level, so seasonal
    moves don't count as volatility. ``std_change_gbp_mwh`` is the spread of the
    day-on-day change in the same slot, closer to what a trader means by volatility.
    Absolute £/MWh is used (not log returns) because prices can be zero or negative.
    """
    d = df.dropna(subset=["price_gbp_mwh"]).copy()
    d["deviation"] = d["price_gbp_mwh"] - d.groupby("month")["price_gbp_mwh"].transform("mean")
    d = d.sort_values("utc_start")
    d["day_change"] = d.groupby("slot")["price_gbp_mwh"].diff()
    g = d.groupby("slot")
    return pd.DataFrame({
        "std_gbp_mwh": g["deviation"].std(),
        "std_change_gbp_mwh": g["day_change"].std(),
    })