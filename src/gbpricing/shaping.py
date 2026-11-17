"""Step 3: shape factors.

A shape factor says how expensive one half-hour is relative to its month:

    ratio method:    factor = average price in (month, day type, slot) / month's baseload
    additive method: offset = average price in (month, day type, slot) - month's baseload

Factors are estimated per historical year, checked for stability across years,
then averaged and smoothed into one set keyed by
(month of year 1-12, day type, slot 0-47) that Step 4 applies to forward prices.

The ratio method is the market norm but breaks down when a month's baseload is
close to zero (dividing by a small number). The additive method doesn't, at the
cost of not scaling with the price level. Both are kept so you can compare them.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

KEYS = ["month_of_year", "day_type", "slot"]


def shape_factors_by_year(df: pd.DataFrame, method: str = "ratio", min_base: float = 10.0) -> pd.DataFrame:
    """Shape factor for every (year, month of year, day type, slot).

    ``df`` must have been through ``add_time_features`` and contain ``price_gbp_mwh``.
    Months whose baseload is below ``min_base`` £/MWh are dropped for the ratio
    method (with a warning) because their ratios are unstable.
    """
    d = df.dropna(subset=["price_gbp_mwh"]).copy()
    d["month_of_year"] = d["month"].dt.month
    d["base"] = d.groupby("month")["price_gbp_mwh"].transform("mean")
    if method == "ratio":
        low = d.loc[d["base"] < min_base, "month"].unique()
        if len(low):
            warnings.warn(f"dropping {len(low)} month(s) with baseload < {min_base}: ratios unstable")
            d = d[d["base"] >= min_base]
        d["value"] = d["price_gbp_mwh"] / d["base"]
    elif method == "additive":
        d["value"] = d["price_gbp_mwh"] - d["base"]
    else:
        raise ValueError("method must be 'ratio' or 'additive'")
    return (
        d.groupby(["year"] + KEYS)["value"].mean()
        .rename("shape_factor").reset_index()
    )


def stability(sf_by_year: pd.DataFrame) -> pd.DataFrame:
    """How similar each (month, day type) shape is across years.

    Returns, per (month_of_year, day_type): the average correlation between each
    pair of years' 48-slot shapes (1 = identical shape), and the average spread
    (standard deviation) of a single factor across years.
    """
    rows = []
    for (m, dt_), g in sf_by_year.groupby(["month_of_year", "day_type"]):
        wide = g.pivot(index="slot", columns="year", values="shape_factor")
        corr = wide.corr().to_numpy()
        pairs = corr[np.triu_indices_from(corr, k=1)]
        rows.append({
            "month_of_year": m, "day_type": dt_, "years": wide.shape[1],
            "mean_pair_corr": float(np.nanmean(pairs)) if len(pairs) else np.nan,
            "mean_std_across_years": float(wide.std(axis=1).mean()),
        })
    return pd.DataFrame(rows)


def smooth_shape_factors(sf_by_year: pd.DataFrame, window: int = 3, years=None) -> pd.DataFrame:
    """Average across years, then smooth across neighbouring slots.

    The smoothing is a centred rolling mean over ``window`` slots that wraps
    around midnight (slot 47 is next to slot 0). ``years`` optionally restricts
    which years are averaged, e.g. to leave out 2022.
    """
    d = sf_by_year if years is None else sf_by_year[sf_by_year["year"].isin(years)]
    avg = d.groupby(KEYS)["shape_factor"].mean()
    full = pd.MultiIndex.from_product(
        [range(1, 13), ["weekday", "weekend"], range(48)], names=KEYS
    )
    avg = avg.reindex(full)
    out = []
    for (m, dt_), s in avg.groupby(level=[0, 1]):
        v = s.droplevel([0, 1]).interpolate(limit_direction="both")
        if window > 1:
            pad = window // 2
            ext = pd.concat([v.iloc[-pad:], v, v.iloc[:pad]])
            v = ext.rolling(window, center=True).mean().iloc[pad:-pad]
        out.append(pd.DataFrame({"month_of_year": m, "day_type": dt_, "slot": range(48), "shape_factor": v.to_numpy()}))
    return pd.concat(out, ignore_index=True)