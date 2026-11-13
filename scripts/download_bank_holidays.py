"""Download England and Wales bank holidays from gov.uk.

Usage:
    python scripts/download_bank_holidays.py
"""
import json
from pathlib import Path

from gbpricing.holidays import fetch_bank_holidays, parse_bank_holidays

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    payload = fetch_bank_holidays()
    path = ROOT / "data/raw/bank_holidays.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    days = parse_bank_holidays(payload)
    print(f"Saved {path.name}: {len(days)} England and Wales holidays, {days.min().date()} to {days.max().date()}")