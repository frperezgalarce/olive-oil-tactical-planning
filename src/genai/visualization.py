
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_ctx_and_scenarios_panel(
    ctx,
    tgt,
    scens,
    feature_names=None,
    max_scen_to_plot=10,
    sample_id=None,
):
    # torch -> numpy (robust)
    if hasattr(ctx, "detach"):
        ctx = ctx.detach().cpu().numpy()[0]
    else:
        ctx = np.asarray(ctx)

    if hasattr(tgt, "detach"):
        tgt = tgt.detach().cpu().numpy()
    else:
        tgt = np.asarray(tgt)

    CTX, D = ctx.shape
    print(tgt.shape)
    H = tgt.shape[0]

    scens = np.asarray(scens)
    assert scens.ndim == 3, "scens must have shape (n_scenarios, H, D)"
    assert scens.shape[1] == H, f"scens horizon {scens.shape[1]} != target horizon {H}"
    assert scens.shape[2] == D, f"scens vars {scens.shape[2]} != target vars {D}"

    if feature_names is None:
        feature_names = [f"var_{k}" for k in range(D)]
    else:
        assert len(feature_names) == D, "feature_names length must match D"

    n_plot = min(max_scen_to_plot, scens.shape[0])

    t_ctx = np.arange(0, CTX)
    t_hor = np.arange(CTX, CTX + H)

    fig, axes = plt.subplots(D, 1, figsize=(12, 2.2 * D), sharex=True)
    if D == 1:
        axes = [axes]

    split_x = CTX - 0.5  # boundary between context and forecast

    for k, ax in enumerate(axes):
        ax.plot(t_ctx, ctx[:, k], label="context", linewidth=2.2)

        for s in range(n_plot):
            ax.plot(t_hor, scens[s, :, k], alpha=0.35, linewidth=1.0)

        ax.plot(t_hor, tgt[:, k], label="true target", linewidth=2.2)

        ax.axvline(split_x, linestyle="--", linewidth=1)
        ax.set_ylabel(feature_names[k])
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        if k == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (days, relative)")
    prefix = "" if sample_id is None else f"Sample i={sample_id}: "
    fig.suptitle(f"{prefix}context + scenarios vs true target", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
def plot_ctx_and_scenarios_panel_ci(
    ctx,
    tgt,
    scens,
    feature_names=None,
    max_scen_to_plot=0,
    center="mean",          # "mean" or "median"
    ci=(0.05, 0.95),        # quantiles
    show_minmax=False,
    sample_id=None,
):
    # torch -> numpy (robust)
    if hasattr(ctx, "detach"):
        ctx = ctx.detach().cpu().numpy()[0]
    else:
        ctx = np.asarray(ctx)

    if hasattr(tgt, "detach"):
        tgt = tgt.detach().cpu().numpy()
    else:
        tgt = np.asarray(tgt)

    CTX, D = ctx.shape
    H = tgt.shape[0]

    scens = np.asarray(scens)
    assert scens.ndim == 3, "scens must have shape (n_scenarios, H, D)"
    assert scens.shape[1] == H, f"scens horizon {scens.shape[1]} != target horizon {H}"
    assert scens.shape[2] == D, f"scens vars {scens.shape[2]} != target vars {D}"

    if feature_names is None:
        feature_names = [f"var_{k}" for k in range(D)]
    else:
        assert len(feature_names) == D, "feature_names length must match D"

    n_scen = scens.shape[0]
    n_plot = min(max_scen_to_plot, n_scen)

    t_ctx = np.arange(0, CTX)
    t_hor = np.arange(CTX, CTX + H)

    if center == "mean":
        central = np.mean(scens, axis=0)
    elif center == "median":
        central = np.median(scens, axis=0)
    else:
        raise ValueError("center must be 'mean' or 'median'")

    q_low, q_high = ci
    lo = np.quantile(scens, q_low, axis=0)
    hi = np.quantile(scens, q_high, axis=0)

    if show_minmax:
        mn = np.min(scens, axis=0)
        mx = np.max(scens, axis=0)

    fig, axes = plt.subplots(D, 1, figsize=(12, 2.2 * D), sharex=True)
    if D == 1:
        axes = [axes]

    split_x = CTX - 0.5

    for k, ax in enumerate(axes):
        ax.plot(t_ctx, ctx[:, k], label="context", linewidth=2.2)

        for s in range(n_plot):
            ax.plot(t_hor, scens[s, :, k], alpha=0.25, linewidth=0.9)

        ax.fill_between(
            t_hor, lo[:, k], hi[:, k],
            alpha=0.25,
            label=f"CI {int(q_low*100)}–{int(q_high*100)}%" if k == 0 else None
        )

        if show_minmax:
            ax.fill_between(
                t_hor, mn[:, k], mx[:, k],
                alpha=0.12,
                label="min–max" if k == 0 else None
            )

        ax.plot(
            t_hor, central[:, k],
            linewidth=2.2,
            label=f"{center} forecast" if k == 0 else None
        )

        ax.plot(t_hor, tgt[:, k], label="true target", linewidth=2.2)

        ax.axvline(split_x, linestyle="--", linewidth=1)
        ax.set_ylabel(feature_names[k])
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        if k == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (days, relative)")
    prefix = "" if sample_id is None else f"Sample i={sample_id}: "
    fig.suptitle(f"{prefix}context + {center} + CI vs true target", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_weatherpair_sample(dataset, i=0, feature_names=None):
    """
    Visualize one (ctx -> tgt) sample from WeatherPairs.
    Plots each variable over time, showing context and target segments.
    """
    ctx, tgt = dataset[i]  # torch tensors: (CTX, D), (H, D)

    ctx = ctx.detach().cpu().numpy()
    tgt = tgt.detach().cpu().numpy()

    CTX, D = ctx.shape
    H = tgt.shape[0]

    if feature_names is None:
        feature_names = [f"var_{k}" for k in range(D)]

    t_ctx = np.arange(0, CTX)
    t_tgt = np.arange(CTX, CTX + H)

    for k in range(D):
        plt.figure()
        plt.plot(t_ctx, ctx[:, k], label="context")
        plt.plot(t_tgt, tgt[:, k], label="target")
        plt.axvline(CTX - 1, linestyle="--", linewidth=1)  # split marker
        plt.xlabel("Time (days, relative)")
        plt.ylabel(feature_names[k])
        plt.title(f"WeatherPairs sample i={i} | feature: {feature_names[k]}")
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        plt.tight_layout()
        plt.show()


def visualize_loss(EPOCHS, train_hist, val_hist):
    epochs = list(range(1, EPOCHS + 1))
    plt.figure()
    plt.plot(epochs, train_hist, label="train")
    plt.plot(epochs, val_hist, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss" if "loss" in globals() else "Metric")
    plt.title("Training vs Validation")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig("training_validation_plot.png")
    plt.show()

def plot_weatherpair_panel(dataset, i=0, feature_names=None):
    """
    Visualize one (ctx -> tgt) WeatherPairs sample as a multi-panel figure.
    One subplot per variable, shared time axis.
    """
    ctx, tgt = dataset[i]  # (CTX, D), (H, D)

    ctx = ctx.detach().cpu().numpy()
    tgt = tgt.detach().cpu().numpy()

    CTX, D = ctx.shape
    H = tgt.shape[0]

    if feature_names is None:
        feature_names = [f"var_{k}" for k in range(D)]

    t_ctx = np.arange(0, CTX)
    t_tgt = np.arange(CTX, CTX + H)

    fig, axes = plt.subplots(
        nrows=D,
        ncols=1,
        figsize=(12, 2.2 * D),
        sharex=True
    )

    if D == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        ax.plot(t_ctx, ctx[:, k], label="context")
        ax.plot(t_tgt, tgt[:, k], label="target")
        ax.axvline(CTX - 1, linestyle="--", linewidth=1)

        ax.set_ylabel(feature_names[k])
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        if k == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (days, relative)")
    fig.suptitle(f"WeatherPairs sample i={i}", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_complete_data(X, columns):
    """Plot the complete data for all variables."""
    n_vars = X.shape[1]
    fig, axs = plt.subplots(n_vars, 1, figsize=(12, 3 * n_vars), sharex=True)
    for i in range(n_vars):
        axs[i].plot(X[:, i], label=columns[i])
        axs[i].set_title(columns[i])
        axs[i].legend()
    plt.xlabel("Days")
    plt.tight_layout()
    plt.show()