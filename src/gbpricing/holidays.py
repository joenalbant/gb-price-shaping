"""England and Wales bank holidays from the gov.uk feed.

Feed: https://www.gov.uk/bank-holidays.json  (public, no key)
Holidays behave like weekends for power prices (low demand), so the shaping
code treats them as weekend days.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

GOV_UK_URL = "https://www.gov.uk/bank-holidays.json"
DIVISION = "england-and-wales"


def fetch_bank_holidays(timeout: int = 30) -> dict:
    """Raw JSON from gov.uk (all divisions)."""
    r = requests.get(GOV_UK_URL, timeout=timeout)
    r.raise_for_status()
    return r.json()


def parse_bank_holidays(payload: dict, division: str = DIVISION) -> pd.DatetimeIndex:
    """Holiday dates for one division, sorted."""
    events = payload[division]["events"]
    return pd.DatetimeIndex(sorted(pd.to_datetime([e["date"] for e in events])))


def load_bank_holidays(path) -> pd.DatetimeIndex:
    """Read the saved raw JSON file and return the holiday dates."""
    with open(Path(path), encoding="utf-8") as f:
        return parse_bank_holidays(json.load(f))