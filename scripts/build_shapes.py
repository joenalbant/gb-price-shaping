"""Step 3: estimate shape factors from the clean MID prices.

Usage:
    python scripts/build_shapes.py                    # ratio method, all years
    python scripts/build_shapes.py --method additive
    python scripts/build_shapes.py --years 2023 2024  # restrict years used
"""
import argparse
from pathlib import Path

import pandas as pd

from gbpricing.features import add_time_features
from gbpricing.holidays import load_bank_holidays
from gbpricing.shaping import shape_factors_by_year, smooth_shape_factors, stability

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="ratio", choices=["ratio", "additive"])
    p.add_argument("--years", type=int, nargs="*")
    p.add_argument("--window", type=int, default=3)
    a = p.parse_args()

    prices = pd.read_parquet(ROOT / "data/processed/mid_clean.parquet")
    hols = load_bank_holidays(ROOT / "data/raw/bank_holidays.json")
    feat = add_time_features(prices, hols)

    by_year = shape_factors_by_year(feat, method=a.method)
    smooth = smooth_shape_factors(by_year, window=a.window, years=a.years)
    by_year.to_parquet(ROOT / f"data/processed/shape_factors_by_year_{a.method}.parquet", index=False)
    smooth.to_parquet(ROOT / f"data/processed/shape_factors_{a.method}.parquet", index=False)

    st = stability(by_year)
    print(f"Saved shape factors ({a.method}).\n\nStability across years (1 = same shape every year):")
    print(st.pivot(index="month_of_year", columns="day_type", values="mean_pair_corr").round(2).to_string())