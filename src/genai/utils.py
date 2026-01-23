
import numpy as np

def fit_standardizer(ds):
    # approximate: sample a subset for speed
    sample_n = min(4000, len(ds))
    idxs = np.random.choice(len(ds), size=sample_n, replace=False)
    all_ = []
    for j in idxs:
        ctx, tgt = ds[j]
        all_.append(ctx.numpy())
        all_.append(tgt.numpy())
    all_ = np.concatenate(all_, axis=0)  # (2*sample_n*60, 4)
    mean = all_.mean(axis=0)
    std = all_.std(axis=0) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def standardize(x, mean_t, std_t):
    # x: (..., 4)
    return (x - mean_t) / std_t

def unstandardize(x, mean_t, std_t):
    return x * std_t + mean_t
