import matplotlib.pyplot as plt
import numpy as np
from itertools import groupby


def plot_profile(
    score_profile,
    motif_ranges,
    save=False,
    save_path=None,
    show=True,
    threshold=None,
    title="Residue-wise repeat score profile",
):
    """Plot residue-wise repeat scores and highlight predicted repeat segments.

    Parameters
    ----------
    score_profile : array-like
        Residue-wise repeat probabilities.
    motif_ranges : iterable of tuple(int, int)
        Predicted ranges in Python coordinates: 0-based, end-exclusive.
    save : bool
        Save the plot to ``save_path``.
    save_path : str or pathlib.Path, optional
        Output image path.
    show : bool
        Display the plot interactively. CLI usage should set this to False.
    threshold : float, optional
        If provided, draw the decision threshold as a horizontal dashed line.
    title : str
        Plot title.
    """
    scores = np.asarray(score_profile).reshape(-1)
    positions = np.arange(1, len(scores) + 1)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.plot(positions, scores)

    # motif_ranges use [start, end) Python coordinates. Convert to the
    # 1-based inclusive coordinates shown to users in segments.tsv.
    for start, end in motif_ranges:
        ax.axvspan(start + 1, end, alpha=0.3)

    if threshold is not None:
        ax.axhline(threshold, linestyle="--", linewidth=1)

    ax.set_xlabel("Residue position")
    ax.set_ylabel("Repeat probability")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    if len(scores) > 0:
        ax.set_xlim(1, len(scores))
    fig.tight_layout()

    if save:
        if save_path is None:
            raise ValueError("save_path must be provided when save=True")
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def get_ranges(preds, cutoff1=0.8, min_len=15, cutoff2=0.8, frac2=0.5):
    preds = preds.flatten()
    above_threshold = preds > cutoff1
    peaks = []
    for k, g in groupby(enumerate(above_threshold), key=lambda x: x[1]):
        if k:
            g = list(g)
            if len(g) >= min_len:
                beg = g[0][0]
                end = g[0][0] + len(g)
                if len(np.where(preds[beg:end] > cutoff2)[0]) / len(g) >= frac2:
                    peaks.append((g[0][0], g[0][0] + len(g)))
    if len(peaks) > 0:
        return peaks
    return []


def analyze_profile(
    preds,
    cutoff1=0.8,
    min_len=15,
    cutoff2=0.8,
    frac2=0.5,
    plot=True,
    save_plot=False,
    save_path=None,
):
    motif_ranges = get_ranges(
        preds,
        cutoff1=cutoff1,
        min_len=min_len,
        cutoff2=cutoff2,
        frac2=frac2,
    )
    print("Predicted motifs are located at: ", motif_ranges)

    if plot:
        plot_profile(
            preds,
            motif_ranges,
            save=save_plot,
            save_path=save_path,
            show=True,
            threshold=cutoff1,
        )
        print("Plot done!")
