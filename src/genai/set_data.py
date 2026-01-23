
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Subset
import torch
import torch.nn as nn

# ======================
# 1) Simulate weather
# ======================


def split_windows_train_val_test(dataset, train_frac=0.7, val_frac=0.15, test_frac=0.15, purge=None):
    n = len(dataset)
    if purge is None:
        purge = max(1, int(np.ceil((dataset.ctx + dataset.h) / dataset.stride)))  # safe default

    # sizes
    n_test = int(n * test_frac)
    n_val  = int(n * val_frac)
    n_train = n - (n_val + n_test) - 2 * purge  # leave room for both purges

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Not enough windows for train/val/test with purge={purge}. "
            f"Try smaller purge, larger dataset, or different fractions."
        )

    train_start = 0
    train_end = train_start + n_train

    val_start = train_end + purge
    val_end = val_start + n_val

    test_start = val_end + purge
    test_end = test_start + n_test

    train_idx = np.arange(train_start, train_end)
    val_idx   = np.arange(val_start, val_end)
    test_idx  = np.arange(test_start, test_end)

    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)

class WeatherPairs(Dataset):
    def __init__(self, data, ctx=60, horizon=90, stride=1):
        self.data = data
        self.ctx = ctx
        self.h = horizon
        self.stride = stride
        self.idx = np.arange(0, len(data) - (ctx + horizon) + 1, stride)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        s = int(self.idx[i])
        ctx = self.data[s:s+self.ctx]
        tgt = self.data[s+self.ctx:s+self.ctx+self.h]
        return torch.from_numpy(ctx), torch.from_numpy(tgt)

def simulate_weather(n_days=6000, year_period=365, p_wet_base=0.25):
    """
    Simulate correlated Tmin/Tavg/Tmax plus precipitation with seasonality.
    """
    t = np.arange(n_days)

    # Seasonal temperature baseline (Celsius)
    seasonal = 10 + 10*np.sin(2*np.pi*t/year_period)  # between ~0 and 20

    # AR(1) anomaly
    phi = 0.92
    eps = np.random.normal(0, 1.5, size=n_days)
    anom = np.zeros(n_days)
    for i in range(1, n_days):
        anom[i] = phi * anom[i-1] + eps[i]

    tavg = seasonal + anom

    # Daily temperature range (bigger in summer, smaller in winter)
    range_seasonal = 7 + 3*np.sin(2*np.pi*(t+40)/year_period)
    drange = np.clip(range_seasonal + np.random.normal(0, 1.0, size=n_days), 3.0, 18.0)

    # Ensure Tmin <= Tavg <= Tmax
    tmin = tavg - 0.6*drange + np.random.normal(0, 0.6, size=n_days)
    tmax = tavg + 0.4*drange + np.random.normal(0, 0.6, size=n_days)

    # Enforce ordering (just in case)
    tmin2 = np.minimum(tmin, np.minimum(tavg, tmax))
    tmax2 = np.maximum(tmax, np.maximum(tavg, tmin))
    tavg2 = np.clip(tavg, tmin2 + 0.1, tmax2 - 0.1)

    # Precip: seasonal wet probability + intensity lognormal when wet
    wet_season = 0.15 + 0.15*(1 - np.sin(2*np.pi*(t+20)/year_period))/2  # wetter in "winter"
    p_wet = np.clip(p_wet_base + wet_season, 0.05, 0.7)

    wet = np.random.binomial(1, p_wet, size=n_days)

    # Correlate with cold anomalies: colder -> slightly higher precip intensity
    cold_factor = np.clip((12 - tavg2)/12, 0, 1)  # 0 warm, 1 cold

    # Intensity (mm)
    intensity = np.random.lognormal(mean=1.2 + 0.6*cold_factor, sigma=0.6, size=n_days)
    precip = wet * intensity

    X = np.stack([tmin2, tavg2, tmax2, precip], axis=-1).astype(np.float32)
    return X

def load_real_weather(
    csv_path: str,
    columns = ["T2M_MIN", "T2M_MAX", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "RH2M", "WS2M"],
    n_days: int | None = None,
    lat: float | None = None,
    lon: float | None = None,
    site: int | None = None,
    agg: str = "mean",
    start_date: str | None = None,
    end_date: str | None = None,
    extend: str = "tile",
) -> np.ndarray:
    """
    Load real daily weather data from a CSV and return X with columns:
      [Tmin, Tavg, Tmax, Precip] as float32, shape (n_days, 4).

    Expected columns in your CSV:
      DATE, LAT, LON, 'ALLSKY_SFC_SW_DWN', 'PRECTOTCORR', 'RH2M',
       'WS2M', 'T2M_MAX', 'T2M_MIN'
    (other columns are ignored)

    Parameters
    ----------
    csv_path:
        Path to CSV (e.g., "/mnt/data/data_20220630_to_20240630.csv")
    n_days:
        Number of days requested. If None, returns all available days in [start_date, end_date].
    lat, lon:
        If provided, choose the nearest site (by Euclidean distance in lat/lon).
    site:
        Alternatively choose a site by index (0..n_sites-1) after sorting unique (LAT,LON).
        If both (lat/lon) and site are None, data is aggregated across all sites by `agg`.
    agg:
        Aggregation across sites when no single site is chosen: "mean" or "median".
    start_date, end_date:
        Optional ISO date strings (inclusive filtering). Example: "2022-07-01".
    extend:
        What to do if n_days > available_days:
          - "error": raise ValueError
          - "tile": repeat the historical sequence until length n_days (keeps chronology pattern)
          - "sample": sample days with replacement from available history (destroys chronology)

    Returns
    -------
    X : np.ndarray
        Shape (n_days, 6) if n_days provided; else (available_days, 6). dtype float32.
    """
    df = pd.read_csv(csv_path)
    if "DATE" not in df.columns:
        raise ValueError("CSV must include a DATE column.")

    # Parse dates
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"])

    required = {'ALLSKY_SFC_SW_DWN', 'PRECTOTCORR', 'RH2M',
                'WS2M', "T2M_MIN", "T2M_MAX", "LAT", "LON"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    # Optional date filtering (inclusive)
    if start_date is not None:
        df = df[df["DATE"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df["DATE"] <= pd.to_datetime(end_date)]

    # Identify sites
    sites = df[["LAT", "LON"]].drop_duplicates().sort_values(["LAT", "LON"]).reset_index(drop=True)

    # Choose a single site, or aggregate
    if lat is not None and lon is not None:
        # Nearest site
        d2 = (sites["LAT"] - float(lat)) ** 2 + (sites["LON"] - float(lon)) ** 2
        chosen = sites.loc[int(d2.idxmin())]
        df = df[(df["LAT"] == chosen["LAT"]) & (df["LON"] == chosen["LON"])].copy()

    elif site is not None:
        site = int(site)
        if site < 0 or site >= len(sites):
            raise ValueError(f"site must be in [0, {len(sites)-1}] but got {site}")
        chosen = sites.loc[site]
        df = df[(df["LAT"] == chosen["LAT"]) & (df["LON"] == chosen["LON"])].copy()

    else:
        # Aggregate across all sites per day
        agg = agg.lower()
        if agg not in {"mean", "median"}:
            raise ValueError("agg must be 'mean' or 'median'")
        df = (
            df.groupby("DATE", as_index=False)[["T2M_MIN", "T2M_MAX", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "RH2M", "WS2M"]]
            .agg(agg)
            .copy()
        )

    # If a single site was selected, collapse to one row per DATE (defensive)
    if "LAT" in df.columns and "LON" in df.columns:
        df = (
            df.groupby("DATE", as_index=False)[columns]
            .mean()
            .copy()
        )

    df = df.sort_values("DATE").reset_index(drop=True)

    # Build series
    tmin = df["T2M_MIN"].astype(float).to_numpy()
    tmax = df["T2M_MAX"].astype(float).to_numpy()
    precip = df["PRECTOTCORR"].astype(float).to_numpy()
    sw_dwn = df["ALLSKY_SFC_SW_DWN"].astype(float).to_numpy()
    rh2m = df["RH2M"].astype(float).to_numpy()
    ws2m = df["WS2M"].astype(float).to_numpy()


    
    # Enforce ordering and basic sanity
    tmin2 = tmin
    tmax2 = tmax
    precip2 = np.clip(precip, 0.0, None)
    sw_dwn2 = np.clip(sw_dwn, 0.0, None)
    rh2m2 = np.clip(rh2m, 0.0, 100.0)
    ws2m2 = np.clip(ws2m, 0.0, None)

    X = np.stack([tmin2, tmax2, precip2, sw_dwn2, rh2m2, ws2m2], axis=-1).astype(np.float32)

    # Handle n_days
    if n_days is None:
        return X, columns

    n_days = int(n_days)
    if n_days <= len(X):
        return X[:n_days], columns

    # Need to extend
    if extend == "error":
        raise ValueError(f"Requested n_days={n_days} but only {len(X)} available in the CSV window.")
    elif extend == "tile":
        reps = (n_days + len(X) - 1) // len(X)
        X_ext = np.tile(X, (reps, 1))[:n_days]
        return X_ext.astype(np.float32)
    elif extend == "sample":
        idx = np.random.randint(0, len(X), size=n_days)
        return X[idx].astype(np.float32), columns
    else:
        raise ValueError("extend must be one of: 'error', 'tile', 'sample'.")
