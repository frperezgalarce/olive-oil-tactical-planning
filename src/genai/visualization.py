
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_ctx_and_scenarios_panel(
    dataset,
    i,
    scens,
    feature_names=None,
    max_scen_to_plot=10
):
    """
    Panel plot (one subplot per variable, shared time axis):
      - Context: plotted on [0, CTX-1]
      - Scenarios: plotted on [CTX, CTX+H-1] (thin lines)
      - True target: plotted on [CTX, CTX+H-1] (thick line)
    """
    ctx, tgt = dataset[i]  # ctx: (CTX, D), tgt: (H, D) torch tensors

    ctx = ctx.detach().cpu().numpy()
    tgt = tgt.detach().cpu().numpy()

    CTX, D = ctx.shape
    H = tgt.shape[0]

    scens = np.asarray(scens)
    assert scens.ndim == 3, "scens must have shape (n_scenarios, H, D)"
    assert scens.shape[1] == H, f"scens horizon {scens.shape[1]} != target horizon {H}"
    assert scens.shape[2] == D, f"scens vars {scens.shape[2]} != target vars {D}"

    if feature_names is None:
        feature_names = [f"var_{k}" for k in range(D)]

    n_plot = min(max_scen_to_plot, scens.shape[0])

    t_ctx = np.arange(0, CTX)
    t_hor = np.arange(CTX, CTX + H)

    fig, axes = plt.subplots(
        nrows=D,
        ncols=1,
        figsize=(12, 2.2 * D),
        sharex=True
    )

    if D == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        # Context
        ax.plot(t_ctx, ctx[:, k], label="context", linewidth=2.2)

        # Scenarios (thin) + true target (thick)
        for s in range(n_plot):
            ax.plot(t_hor, scens[s, :, k], alpha=0.35, linewidth=1.0)
        ax.plot(t_hor, tgt[:, k], label="true target", linewidth=2.2)

        # Split marker
        ax.axvline(CTX - 1, linestyle="--", linewidth=1)

        ax.set_ylabel(feature_names[k])
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        if k == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (days, relative)")
    fig.suptitle(f"Sample i={i}: context + scenarios vs true target", fontsize=12)
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