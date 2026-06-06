"""Visualization utilities for ideal point estimation results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import arviz as az

# Club color palette (approximate party colors)
CLUB_COLORS = {
    "PiS":     "#003087",  # dark blue
    "KO":      "#F5821F",  # orange
    "TD":      "#00A550",  # green
    "Lewica":  "#E31E24",  # red
    "Konfederacja": "#8B0000",  # dark red
    "PL2050":  "#1DACD6",  # light blue (if separate from TD)
    "PSL":     "#009A44",  # green (if separate from TD)
    "Polska2050": "#1DACD6",
}
DEFAULT_COLOR = "#888888"


def _club_color(club: str) -> str:
    return CLUB_COLORS.get(club, DEFAULT_COLOR)


def plot_ideal_points(
    idata: object,
    mp_ids: list,
    mp_info: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """
    Strip plot of posterior mean ideal points, colored by club.
    Error bars show 90% credible interval.
    """
    x_samples = idata.posterior["x"].values  # (chains, draws, n_mps)
    x_flat = x_samples.reshape(-1, len(mp_ids))

    x_mean = x_flat.mean(axis=0)
    x_lo = np.percentile(x_flat, 5, axis=0)
    x_hi = np.percentile(x_flat, 95, axis=0)

    clubs = [mp_info.loc[mid, "club"] if mid in mp_info.index else "" for mid in mp_ids]
    colors = [_club_color(c) for c in clubs]

    order = np.argsort(x_mean)

    fig, ax = plt.subplots(figsize=(14, 5))

    for rank, idx in enumerate(order):
        ax.plot([x_lo[idx], x_hi[idx]], [rank, rank],
                color=colors[idx], alpha=0.3, linewidth=0.6)
        ax.scatter(x_mean[idx], rank, color=colors[idx], s=8, zorder=3)

    # Legend
    seen_clubs = sorted(set(clubs) - {""})
    patches = [mpatches.Patch(color=_club_color(c), label=c) for c in seen_clubs]
    ax.legend(handles=patches, loc="upper left", fontsize=8, framealpha=0.9)

    ax.set_xlabel("Ideal point (positive = right)", fontsize=11)
    ax.set_yticks([])
    ax.set_title("Ideal points — Sejm term10\n(posterior mean ± 90% CI)", fontsize=12)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_club_distributions(
    idata: object,
    mp_ids: list,
    mp_info: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """Violin plot of ideal point distributions per club."""
    x_samples = idata.posterior["x"].values.reshape(-1, len(mp_ids))
    clubs = [mp_info.loc[mid, "club"] if mid in mp_info.index else "?" for mid in mp_ids]

    club_data = {}
    for i, club in enumerate(clubs):
        club_data.setdefault(club, []).append(x_samples[:, i].mean())

    # Sort clubs by median ideal point
    club_order = sorted(club_data, key=lambda c: np.median(club_data[c]))
    medians = [np.median(club_data[c]) for c in club_order]
    means_list = [club_data[c] for c in club_order]

    fig, ax = plt.subplots(figsize=(10, 5))
    parts = ax.violinplot(means_list, positions=range(len(club_order)),
                          showmedians=True, widths=0.7)

    for i, (body, club) in enumerate(zip(parts["bodies"], club_order)):
        body.set_facecolor(_club_color(club))
        body.set_alpha(0.7)

    ax.set_xticks(range(len(club_order)))
    ax.set_xticklabels(club_order, fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Posterior mean ideal point")
    ax.set_title("Ideal point distribution by club — Sejm term10")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_trace(
    idata: object,
    save_path: str | None = None,
) -> None:
    """Trace plots for hyperparameters."""
    axes = az.plot_trace(
        idata,
        var_names=["sigma_beta", "sigma_alpha", "mu_alpha"],
        compact=True,
    )
    plt.suptitle("MCMC trace — hyperparameters", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()
