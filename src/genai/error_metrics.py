import numpy as np
import torch


def crps_ensemble_mc(Omega_all, y_all, average=False, n_pairs=256, seed=0, dtype=None):
    """
    Approximate CRPS by Monte Carlo for the pairwise term.
    Exact term1, MC estimate for term2.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(Omega_all)
    if X.ndim != 3:
        raise ValueError(f"Omega_all must be (S,N,D). Got {X.shape}")
    S, N, D = X.shape

    y = np.asarray(y_all)
    if y.ndim == 1:
        y = y[:, None]
    if y.shape != (N, D):
        raise ValueError(f"y_all must be (N,D)={(N,D)} or (N,). Got {y.shape}")

    if dtype is not None:
        X = X.astype(dtype, copy=False)
        y = y.astype(dtype, copy=False)

    term1 = np.mean(np.abs(X - y[None, :, :]), axis=0)  # (N,D)

    i = rng.integers(0, S, size=n_pairs)
    j = rng.integers(0, S, size=n_pairs)
    # MC estimate of E|X - X'|
    term2 = np.mean(np.abs(X[i] - X[j]), axis=0)  # (N,D)

    crps = term1 - 0.5 * term2
    return float(crps.mean()) if average else crps

def crps_ensemble_snd(Omega_all, y_all, average=False):
    """
    Empirical-ensemble CRPS.

    Parameters
    ----------
    Omega_all : array, shape (S, N, D)
        Ensemble/scenario forecasts.
    y_all : array, shape (N, D) or (N,)
        Observations aligned with N,D. If (N,), D is assumed 1.
    average : bool
        If True, returns scalar mean over (N,D). Otherwise returns (N,D).

    Returns
    -------
    crps : array or float
        (N,D) if average=False, else scalar.
    """
    Omega_all = np.asarray(Omega_all)
    y_all = np.asarray(y_all)

    if Omega_all.ndim != 3:
        raise ValueError(f"Omega_all must be (S,N,D). Got {Omega_all.shape}")

    S, N, D = Omega_all.shape

    if y_all.ndim == 1:  # (N,) -> (N,1)
        if y_all.shape[0] != N:
            raise ValueError(f"y_all has shape {y_all.shape}, expected (N,) with N={N}")
        y_all = y_all[:, None]
    elif y_all.ndim == 2:
        if y_all.shape != (N, D):
            raise ValueError(f"y_all must be (N,D) = {(N,D)}. Got {y_all.shape}")
    else:
        raise ValueError(f"y_all must be (N,D) or (N,). Got {y_all.shape}")

    # term1: (1/S) sum_s |x_s - y|
    term1 = np.mean(np.abs(Omega_all - y_all[None, :, :]), axis=0)  # (N,D)

    # term2: mean_{s,s'} |x_s - x_s'| via sorting along S
    x_sorted = np.sort(Omega_all, axis=0)  # (S,N,D)
    k = np.arange(1, S + 1, dtype=x_sorted.dtype).reshape(S, 1, 1)
    coeff = (2 * k - S - 1)  # (S,1,1)

    mean_abs_pair = (2.0 / (S * S)) * np.sum(coeff * x_sorted, axis=0)  # (N,D)

    crps = term1 - 0.5 * mean_abs_pair
    return float(np.mean(crps)) if average else crps


def mse_ensemble_mean_batch_fast(Omega_all, y_all, average=True):
    X = np.asarray(Omega_all)
    y = np.asarray(y_all)
    yhat = X.mean(axis=0)  # (N,H,D) or (N,D) depending on your data
    err2 = (yhat - y) ** 2
    return float(err2.mean()) if average else err2.mean(axis=0)

def mae_ensemble_mean_batch_fast(Omega_all, y_all, average=True):
    X = np.asarray(Omega_all)
    y = np.asarray(y_all)
    yhat = X.mean(axis=0)
    ae = np.abs(yhat - y)
    return float(ae.mean()) if average else ae.mean(axis=0)

def mse_ensemble_mean_batch(Omega_all, y_all):
    yhat = Omega_all.mean(axis=0)  # (N,H,D)
    yhat = np.array(yhat)
    y_all = np.array(y_all)
    return np.mean((yhat - y_all) ** 2, axis=0)


def mae_ensemble_mean_batch(Omega_all, y_all, idx=0, h=0, d=None):
    yhat = Omega_all.mean(axis=0)  
    yhat = np.array(yhat)
    y_all = np.array(y_all)
    return np.mean(np.abs(yhat - y_all), axis=0)

def mape_ensemble_mean_batch(Omega_all, y_all, eps=1e-8):
    yhat = Omega_all.mean(axis=0)  # (N,H,D)
    denom = np.maximum(np.abs(y_all), eps)
    return 100.0 * np.mean(np.abs((yhat - y_all) / denom), axis=0)


def crps_ensemble_batch(Omega_all, y_all):
    """
    Omega_all: (N,M,H,D)
    y_all:     (N,H,D)
    returns scalar average CRPS over (N,H,D)
    """
    # term1: E|X - y|
    term1 = np.mean(np.abs(Omega_all - y_all[:, None, :, :]), axis=1)  # (N,H,D)

    # term2: 0.5 E|X - X'|
    pairwise = np.abs(
        Omega_all[:, :, None, :, :] - Omega_all[:, None, :, :, :]
    )  # (N,M,M,H,D)
    term2 = 0.5 * np.mean(pairwise, axis=(1, 2))  # (N,H,D)

    return np.mean(term1 - term2)


def mse_ensemble_mean(Omega, y):
    # Omega: (M,H,D), y: (H,D)
    yhat = Omega.mean(axis=0)  # (H,D)
    return np.mean((yhat - y) ** 2)

def map_percentage_ensemble_mean(Omega, y, eps=1e-8):
    # "MAP" interpreted as MAPE (%), using ensemble mean
    yhat = Omega.mean(axis=0)  # (H,D)
    denom = np.maximum(np.abs(y), eps)
    return 100.0 * np.mean(np.abs((yhat - y) / denom))

def crps_ensemble(Omega, y):
    """
    Ensemble CRPS averaged over (H,D).
    Omega: (M,H,D)
    y:     (H,D)
    """
    M = Omega.shape[0]

    # term1: (H,D)
    term1 = np.mean(np.abs(Omega - y[None, :, :]), axis=0)

    # term2: (H,D) using pairwise diffs over ensemble members
    # shape trick: (M,1,H,D) - (1,M,H,D) -> (M,M,H,D)
    pairwise = np.abs(Omega[:, None, :, :] - Omega[None, :, :, :])
    term2 = 0.5 * np.mean(pairwise, axis=(0, 1))  # (H,D)

    return np.mean(term1 - term2)
