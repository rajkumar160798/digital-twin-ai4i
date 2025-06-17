import pandas as pd
import numpy as np

def inject_spike(series, location, magnitude):
    """
    Inject a spike into a 1D array or pandas Series.
    """
    if isinstance(series, pd.Series):
        series = series.copy()
        series.iloc[location] += magnitude
        return series
    elif isinstance(series, np.ndarray):
        series = series.copy()
        series[location] += magnitude
        return series
    else:
        raise ValueError("Input must be a 1D NumPy array or pandas Series.")


def inject_random_noise(series, noise_level=0.1):
    """
    Inject random noise into the time series data.
    Args:
        series (pd.Series): The time series data.
        noise_level (float): The standard deviation of the noise to be added.
    Returns:
        pd.Series: The time series with random noise injected.
    """
    ...

def inject_gradual_drift(series, strength=0.1):
    ...
