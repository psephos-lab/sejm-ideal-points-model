"""Visualization utilities for ideal point estimation results.

All functions take plain NumPy posterior draws (decoupled from any MCMC backend):
    x_flat   : (n_draws, n_mps)        flattened posterior draws of ideal points
    x_chains : (n_chains, n_draws, n)  per-chain draws (for trace plots)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Club color palette (approximate party colors), matching actual API club names
CLUB_COLORS = {
    "PiS":             "#003087",  # dark blue
    "KO":              "#F5821F",  # orange
    "PSL-TD":          "#00A550",  # green (PSL / Third Way)
    "Polska2050":      "#FACC15",  # yellow
    "Polska2050-TD":   "#FACC15",  # yellow
    "Lewica":          "#E31E24",  # red
    "Razem":           "#951B81",  # magenta
    "Konfederacja":    "#1A1A1A",  # near-black
    "Konfederacja_KP": "#4A4A4A",  # dark gray
    "Centrum":         "#1DACD6",  # light blue
    "Demokracja":      "#00BFA5",  # teal
    "niez.":           "#AAAAAA",  # gray (independents)
    # --- historical terms (VII–IX): same family hues as the X-term palette ---
    "PO":              "#F5821F",  # Platforma — same camp as KO (orange)
    "PO-KO":           "#F5821F",  # Platforma–Koalicja Obywatelska (orange)
    "PSL":             "#00A550",  # PSL — green
    "PSL-KP":          "#00A550",  # PSL / Koalicja Polska — green
    "PSL-UED":         "#4CAF50",  # PSL-UED — lighter green
    "KP":              "#00A550",  # Koalicja Polska — green
    "Kukiz15":         "#16A085",  # Kukiz'15 — teal
    "SLD":             "#E31E24",  # left — red
    "LD":              "#C0398B",  # magenta
    "PS":              "#5D6D7E",  # slate
    "UPR":             "#5D3A9B",  # purple
    "PP":              "#A0522D",  # brown
    "TERAZ!":          "#00ACC1",  # cyan
    "WiS":             "#9C27B0",  # purple-magenta
    "ZP":              "#2C5AA0",  # blue variant
    "RP":              "#C2185B",  # Ruch Palikota — crimson
    "TR":              "#AD1457",  # Twój Ruch — crimson variant
    "BC":              "#795548",  # brown
    "KPSP":            "#8D6E63",  # brown variant
    # --- V–VI kadencja (2005–2011) ---
    "Samoobrona":      "#C9A227",  # gold
    "LPR":             "#922B21",  # dark maroon
    "RLN":             "#117864",  # dark teal
    "Prawica":         "#7E5109",  # amber/olive
    "SDPL":            "#CB4335",  # brick red (left splinter)
    "PJN":             "#2471A3",  # steel blue
    "Polska_Plus":     "#5499C7",  # light steel blue
}
DEFAULT_COLOR = "#888888"


def _club_color(club: str) -> str:
    return CLUB_COLORS.get(club, DEFAULT_COLOR)


def _clubs_for(mp_ids: list, mp_info: pd.DataFrame) -> list:
    return [mp_info.loc[mid, "club"] if mid in mp_info.index else "" for mid in mp_ids]


def plot_ideal_points(x_flat, mp_ids, mp_info, save_path=None, term_label="X kadencji"):
    """Strip plot of posterior mean ideal points (±90% CI), colored by club."""
    x_mean = x_flat.mean(axis=0)
    x_lo = np.percentile(x_flat, 5, axis=0)
    x_hi = np.percentile(x_flat, 95, axis=0)

    clubs = _clubs_for(mp_ids, mp_info)
    colors = [_club_color(c) for c in clubs]
    order = np.argsort(x_mean)

    fig, ax = plt.subplots(figsize=(14, 6))
    for rank, idx in enumerate(order):
        ax.plot([x_lo[idx], x_hi[idx]], [rank, rank],
                color=colors[idx], alpha=0.35, linewidth=0.6)
        ax.scatter(x_mean[idx], rank, color=colors[idx], s=9, zorder=3)

    seen = sorted(set(clubs) - {""})
    patches = [mpatches.Patch(color=_club_color(c), label=c) for c in seen]
    ax.legend(handles=patches, loc="upper left", fontsize=8, framealpha=0.9, ncol=2)

    ax.set_xlabel("Punkt idealny — główna oś podziału", fontsize=11)
    ax.set_yticks([])
    ax.set_title(f"Punkty idealne posłów — Sejm {term_label}\n(średnia a posteriori ± 90% CI)", fontsize=12)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150); print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_club_distributions(x_flat, mp_ids, mp_info, save_path=None, term_label="X kadencji"):
    """Violin plot of per-MP posterior means grouped by club."""
    x_mean = x_flat.mean(axis=0)
    clubs = _clubs_for(mp_ids, mp_info)

    club_data = {}
    for i, club in enumerate(clubs):
        club_data.setdefault(club or "?", []).append(x_mean[i])

    club_order = sorted(club_data, key=lambda c: np.median(club_data[c]))
    means_list = [club_data[c] for c in club_order]

    fig, ax = plt.subplots(figsize=(11, 5))
    parts = ax.violinplot(means_list, positions=range(len(club_order)),
                          showmedians=True, widths=0.7)
    for body, club in zip(parts["bodies"], club_order):
        body.set_facecolor(_club_color(club)); body.set_alpha(0.7)

    ax.set_xticks(range(len(club_order)))
    ax.set_xticklabels(club_order, fontsize=9, rotation=30, ha="right")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Punkt idealny (średnia a posteriori)")
    ax.set_title(f"Rozkład punktów idealnych wg klubu — Sejm {term_label}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150); print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_trace(x_chains, mp_ids, mp_info, save_path=None, n_show=4):
    """Trace plot of a few representative MPs' ideal points across chains."""
    n_chains, n_draws, n = x_chains.shape
    x_mean = x_chains.reshape(-1, n).mean(0)
    # pick MPs spanning the axis (the extremes and two in between)
    order = np.argsort(x_mean)
    picks = [order[0], order[n // 3], order[2 * n // 3], order[-1]][:n_show]

    fig, axes = plt.subplots(len(picks), 1, figsize=(10, 2 * len(picks)), sharex=True)
    if len(picks) == 1:
        axes = [axes]
    for ax, idx in zip(axes, picks):
        mid = mp_ids[idx]
        name = f"{mp_info.loc[mid,'last_name']} ({mp_info.loc[mid,'club']})" if mid in mp_info.index else str(mid)
        for c in range(n_chains):
            ax.plot(x_chains[c, :, idx], linewidth=0.6, alpha=0.8, label=f"chain {c+1}")
        ax.set_ylabel(f"x\n{name}", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Iteracja (po burn-in)")
    fig.suptitle("Trace — wybrane punkty idealne (mieszanie łańcuchów)", y=1.0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight"); print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()
