"""Shared test fixtures: synthetic prices (no network) and assumed forwards."""
import numpy as np
import pandas as pd
import pytest

from gbpricing.features import add_time_features
from gbpricing.synthetic import make_synthetic_mid


@pytest.fixture(scope="session")
def hist():
    """Two years of synthetic half-hourly prices with time features."""
    return add_time_features(make_synthetic_mid("2023-01-01", "2024-12-31"))


@pytest.fixture(scope="session")
def forwards():
    months = pd.date_range("2027-01-01", "2027-12-01", freq="MS")
    base = np.array([95, 90, 82, 72, 65, 63, 66, 68, 74, 82, 92, 96], float)
    return pd.DataFrame({"month": months, "base_gbp_mwh": base, "peak_gbp_mwh": base * 1.15})


@pytest.fixture(scope="session")
def shape(hist):
    from gbpricing.shaping import shape_factors_by_year, smooth_shape_factors
    return smooth_shape_factors(shape_factors_by_year(hist))