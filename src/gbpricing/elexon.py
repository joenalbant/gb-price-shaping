"""Download and clean half-hourly Market Index Data (MID) from the Elexon Insights API.

Endpoint: https://data.elexon.co.uk/bmrs/api/v1/balancing/pricing/market-index
No API key is needed. Requests are split into short chunks to stay within the
API's range limits.

MID is reported by two providers: APXMIDP (EPEX, the liquid one) and N2EXMIDP
(often zero volume). We keep the raw data for both and build the clean series
from APXMIDP.
"""
from __future__ import annotations

import time

import pandas as pd
import requests

from .calendar import settlement_index, sp_to_utc

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"
MID_ENDPOINT = f"{BASE_URL}/balancing/pricing/market-index"
PROVIDERS = ("APXMIDP", "N2EXMIDP")


# --------------------------------------------------------------------------- download
def _get(url: str, params: dict, retries: int = 4, timeout: int = 60) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def fetch_mid_raw(start, end, chunk_days: int = 6, pause: float = 0.2) -> pd.DataFrame:
    """Raw MID rows for both providers between two dates (inclusive)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    frames = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + pd.Timedelta(days=chunk_days - 1), end)
        params = {
            # Use the UTC window that covers the whole settlement days.
            "from": sp_to_utc(chunk_start, 1).strftime("%Y-%m-%dT%H:%MZ"),
            "to": (sp_to_utc(chunk_end + pd.Timedelta(days=1), 1)).strftime("%Y-%m-%dT%H:%MZ"),
            "dataProviders": list(PROVIDERS),
            "format": "json",
        }
        payload = _get(MID_ENDPOINT, params)
        data = payload.get("data", payload) if isinstance(payload, dict) else payload
        frames.append(pd.DataFrame(data))
        print(f"  {chunk_start.date()} -> {chunk_end.date()}: {len(frames[-1])} rows")
        chunk_start = chunk_end + pd.Timedelta(days=1)
        time.sleep(pause)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if raw.empty:
        return raw
    # The UTC window can catch a period either side; keep only requested settlement days.
    sd = pd.to_datetime(raw["settlementDate"])
    return raw.loc[(sd >= start.normalize()) & (sd <= end.normalize())].reset_index(drop=True)


# --------------------------------------------------------------------------- cleaning
def clean_mid(raw: pd.DataFrame, provider: str = "APXMIDP") -> pd.DataFrame:
    """One row per settlement period for one provider, on a complete calendar grid.

    Output columns:
        utc_start, settlement_date, settlement_period, price_gbp_mwh, volume_mwh, is_missing

    * The timestamp is rebuilt from (settlement_date, settlement_period), then
      cross-checked against the API's startTime if present.
    * Periods with zero traded volume carry no price information, so their
      price is set to NaN and flagged rather than silently kept as 0.
    * Periods absent from the API are added as NaN rows, so gaps are visible.
    """
    df = raw.loc[raw["dataProvider"] == provider].copy()
    df["settlement_date"] = pd.to_datetime(df["settlementDate"])
    df["settlement_period"] = df["settlementPeriod"].astype(int)
    df = df.drop_duplicates(["settlement_date", "settlement_period"], keep="last")

    df["utc_start"] = [
        sp_to_utc(d, sp) for d, sp in zip(df["settlement_date"], df["settlement_period"])
    ]
    if "startTime" in df:
        api_ts = pd.to_datetime(df["startTime"], utc=True)
        bad = df.loc[api_ts != df["utc_start"]]
        if len(bad):
            raise ValueError(f"{len(bad)} rows where API startTime disagrees with calendar:\n{bad.head()}")

    df["price_gbp_mwh"] = df["price"].astype(float)
    df["volume_mwh"] = df["volume"].astype(float)
    df.loc[df["volume_mwh"] <= 0, "price_gbp_mwh"] = float("nan")

    grid = settlement_index(df["settlement_date"].min(), df["settlement_date"].max())
    out = grid.merge(
        df[["settlement_date", "settlement_period", "price_gbp_mwh", "volume_mwh"]],
        on=["settlement_date", "settlement_period"],
        how="left",
    )
    out["is_missing"] = out["price_gbp_mwh"].isna()
    return out


def validate(df: pd.DataFrame) -> None:
    """Raise if the clean series breaks a calendar rule."""
    if df["utc_start"].duplicated().any():
        raise ValueError("duplicate UTC timestamps")
    steps = df["utc_start"].diff().dropna().unique()
    if len(steps) != 1 or steps[0] != pd.Timedelta(minutes=30):
        raise ValueError(f"UTC index is not a continuous 30-minute grid: steps={steps}")
    counts = df.groupby("settlement_date")["settlement_period"].agg(["count", "max"])
    if not counts["count"].isin([46, 48, 50]).all() or (counts["count"] != counts["max"]).any():
        raise ValueError(f"bad period counts:\n{counts[~counts['count'].isin([46, 48, 50])]}")


def summary(df: pd.DataFrame) -> pd.Series:
    counts = df.groupby("settlement_date").size()
    return pd.Series(
        {
            "first_date": df["settlement_date"].min().date(),
            "last_date": df["settlement_date"].max().date(),
            "periods": len(df),
            "missing_periods": int(df["is_missing"].sum()),
            "days_46": int((counts == 46).sum()),
            "days_50": int((counts == 50).sum()),
            "negative_price_periods": int((df["price_gbp_mwh"] < 0).sum()),
            "mean_price_gbp_mwh": round(df["price_gbp_mwh"].mean(), 2),
        }
    )