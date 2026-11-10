"""Download MID prices from Elexon and save raw + clean parquet files.

Usage:
    python scripts/download_mid.py --start 2023-01-01 --end 2025-12-31
"""
import argparse
from pathlib import Path

from gbpricing.elexon import clean_mid, fetch_mid_raw, summary, validate

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--provider", default="APXMIDP")
    args = p.parse_args()

    print(f"Downloading MID {args.start} to {args.end}")
    raw = fetch_mid_raw(args.start, args.end)
    raw_path = ROOT / "data/raw" / f"mid_raw_{args.start}_{args.end}.parquet"
    raw.to_parquet(raw_path, index=False)

    clean = clean_mid(raw, provider=args.provider)
    validate(clean)
    clean_path = ROOT / "data/processed" / "mid_clean.parquet"
    clean.to_parquet(clean_path, index=False)

    print(f"\nSaved {raw_path.name} and {clean_path.name}\n")
    print(summary(clean).to_string())


if __name__ == "__main__":
    main()