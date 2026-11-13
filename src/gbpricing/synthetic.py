"""Synthetic half-hourly prices for tests and dry runs. NOT market data.

Shaped like the clean MID table (same columns) so every step can be run end to
end without a download. Prices = seasonal monthly level x daily shape + noise,
with a midday dip in summer and occasional negative prices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar import LONDON, settlement_index


def make_synthetic_mid(start="2023-01-01", end="2025-12-31", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = settlement_index(start, end)
    local = df["utc_start"].dt.tz_convert(LONDON)
    hour = (local.dt.hour + local.dt.minute / 60).to_numpy()
    doy = local.dt.day_of_year.to_numpy()
    weekend = (df["settlement_date"].dt.day_of_week >= 5).to_numpy()

    winter = np.cos(2 * np.pi * (doy - 15) / 365)                      # 1 mid-Jan, -1 mid-Jul
    level = 80 + 20 * winter                                           # monthly-ish level
    evening = 0.35 * np.exp(-((hour - 17.5) / 1.5) ** 2) * (1 + 0.5 * winter)
    morning = 0.15 * np.exp(-((hour - 8) / 1.5) ** 2)
    night = -0.25 * np.exp(-((hour - 3.5) / 2.5) ** 2)
    solar_dip = -0.35 * np.exp(-((hour - 13) / 2.0) ** 2) * (1 - winter) / 2
    shape = 1 + evening + morning + night + solar_dip - 0.12 * weekend
    noise = rng.normal(0, 0.12, len(df)) * level
    price = level * shape + noise
    price = np.where(rng.random(len(df)) < 0.004, -rng.uniform(1, 30, len(df)), price)

    df["price_gbp_mwh"] = price.round(2)
    df["volume_mwh"] = rng.uniform(500, 2500, len(df)).round(0)
    df["is_missing"] = False
    return df