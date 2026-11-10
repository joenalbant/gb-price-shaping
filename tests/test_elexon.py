"""Cleaning tests on a synthetic API response (no network needed)."""
import numpy as np
import pandas as pd
import pytest

from gbpricing.calendar import settlement_index
from gbpricing.elexon import clean_mid, validate


def fake_raw(start="2024-10-26", end="2024-10-28"):
    grid = settlement_index(start, end)
    rows = []
    for ts, d, sp in grid.itertuples(index=False):
        for prov, vol in (("APXMIDP", 500.0), ("N2EXMIDP", 0.0)):
            rows.append({
                "startTime": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "dataProvider": prov,
                "settlementDate": d.strftime("%Y-%m-%d"),
                "settlementPeriod": sp,
                "price": 80.0 + sp,
                "volume": vol,
            })
    return pd.DataFrame(rows)


def test_clean_handles_autumn_clock_change():
    out = clean_mid(fake_raw())
    validate(out)
    counts = out.groupby(out["settlement_date"].dt.date).size()
    assert list(counts) == [48, 50, 48]
    assert not out["is_missing"].any()


def test_missing_period_is_flagged_not_dropped():
    raw = fake_raw()
    raw = raw[~((raw["settlementDate"] == "2024-10-27") & (raw["settlementPeriod"] == 49))]
    out = clean_mid(raw)
    validate(out)
    assert out["is_missing"].sum() == 1


def test_zero_volume_price_becomes_nan():
    raw = fake_raw()
    raw.loc[(raw["dataProvider"] == "APXMIDP") & (raw["settlementPeriod"] == 10), "volume"] = 0
    out = clean_mid(raw)
    assert np.isnan(out.loc[out["settlement_period"] == 10, "price_gbp_mwh"]).all()


def test_duplicate_rows_are_removed():
    raw = fake_raw()
    out = clean_mid(pd.concat([raw, raw.head(5)]))
    validate(out)


def test_wrong_api_timestamp_raises():
    raw = fake_raw()
    raw.loc[0, "startTime"] = "2024-10-26T05:00:00Z"
    with pytest.raises(ValueError):
        clean_mid(raw)