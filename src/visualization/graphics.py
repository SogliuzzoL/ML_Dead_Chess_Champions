"""
Refactored visualization utilities for the ML_Dead_Chess_Champions project.

This module provides routines to produce publication-quality figures (IEEE-friendly)
for:
 - Jensen-Shannon Divergence heatmaps (symmetric, test-only)
 - Asymmetric stability heatmaps (train vs test)
 - Moves-per-player distribution bar chart

Design notes
------------
- All plotting routines accept a `Config` object which encapsulates file paths and
  configuration parameters. This keeps plotting code pure and free of global state.
- I/O is robust: parquet read errors and missing files are handled gracefully with
  informative logging. Figure saving ensures parent directories exist.
- Helper functions centralize common operations (DataFrame loading, matrix creation,
  figure saving) so that the plotting functions focus on visual logic.
- Type hints and concise docstrings are provided for better maintainability.

Author: refactor by assistant
"""

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib.figure import Figure

from src.core.config import Config
from src.core.utils import getLogger

# Module-level logger
logger = getLogger()

# Seaborn theme configuration chosen for clarity and reproducibility in academic figures.
sns.set_theme(
    context="paper",
    style="ticks",
    font="STIXGeneral",
    palette="colorblind",
    rc={
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    },
)

# Figure size constants in inches (IEEE single-column friendly)
FIG_WIDTH = 3.5
HEATMAP_HEIGHT = 4.5
DISTRIBUTION_HEIGHT = 2.8

# Color map names used consistently across figures
HEATMAP_CMAP = "magma_r"  # reversed for heatmaps: bright = similar (low distance)
BAR_CMAP = "magma"  # forward magma for bar color mapping


# ---------- Helper utilities ----------


def _ensure_parent_dir(path: str) -> None:
    """Ensure the parent directory of `path` exists. Creates it if necessary."""
    p = Path(path)
    if not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("Created directory: %s", str(p.parent))


def _safe_read_parquet(path: str) -> Optional[pl.DataFrame]:
    """
    Read a parquet file into a Polars DataFrame with robust error handling.

    Returns None and logs a descriptive error if reading fails.
    """
    try:
        df = pl.read_parquet(path)
        logger.debug("Loaded parquet file: %s (rows=%d)", path, df.height)
        return df
    except FileNotFoundError:
        logger.error("Parquet file not found: %s", path)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to read parquet %s: %s", path, exc)
    return None


def _save_figure(
    fig: Figure, out_path: str, *, fmt: str = "pdf", dpi: int = 600
) -> None:
    """
    Save a Matplotlib figure to disk ensuring the output directory exists.

    The function logs success and handles filesystem errors gracefully.
    """
    _ensure_parent_dir(out_path)
    try:
        fig.savefig(out_path, format=fmt, dpi=dpi, bbox_inches="tight")
        logger.info("Saved figure to %s", out_path)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Unable to save figure to %s: %s", out_path, exc)


def _to_ordered_square_matrix(
    df: pl.DataFrame,
    p1_col: str,
    p2_col: str,
    value_col: str,
    ordered_players: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Convert a long-format Polars DataFrame into a pandas square matrix (DataFrame) indexed
    and columned by player names in `ordered_players`.

    If `ordered_players` is None the function determines the sorted union of players
    occurring in `p1_col` and `p2_col`.
    """
    # Compute players union in a stable sorted order if not provided
    if ordered_players is None:
        p1_names = df.select(pl.col(p1_col)).unique().to_series().to_list()
        p2_names = df.select(pl.col(p2_col)).unique().to_series().to_list()
        players = sorted(set(p1_names) | set(p2_names))
    else:
        players = list(ordered_players)

    # Create mirror and diagonal entries to ensure a full square matrix
    mirror = df.select(
        [pl.col(p2_col).alias(p1_col), pl.col(p1_col).alias(p2_col), pl.col(value_col)]
    )
    diag = pl.DataFrame(
        {p1_col: players, p2_col: players, value_col: [0.0] * len(players)}
    )

    combined = pl.concat([df, mirror, diag]).unique(subset=[p1_col, p2_col])
    # Pivot to pandas to leverage seaborn's heatmap which works naturally with pandas
    matrix_pd = (
        combined.to_pandas()
        .pivot(index=p1_col, columns=p2_col, values=value_col)
        .reindex(index=players, columns=players)
    )
    return matrix_pd


# ---------- Plotting routines ----------


def jsd_heatmap(config: Config) -> None:
    """
    Generate a symmetric Jensen-Shannon Divergence heatmap (test set only) and save it as a PDF.

    Visualization choices:
    - Colormap: reversed 'magma' so low distances (similarity) appear bright and
      large distances appear dark. This choice is documented and consistent across figures.
    - Annotated values are shown with two decimal places; annotation font sizes are tuned
      for compact multi-player matrices suitable for IEEE single-column figures.
    """
    # Load precomputed distances for the test set
    distances_path = config.paths.get_distances_path(
        method=config.jsd.method, is_test=True, kde=config.jsd.kde
    )
    df = _safe_read_parquet(distances_path)
    if df is None:
        logger.error(
            "JSD heatmap aborted: could not load distances from %s", distances_path
        )
        return

    # If the distances parquet exists but is empty, skip plotting
    try:
        if df.height == 0:
            logger.warning(
                "JSD heatmap aborted: distances file %s is empty. Run evaluation first.",
                distances_path,
            )
            return
    except Exception:
        # Defensive: if df doesn't implement height, convert to pandas
        if df.to_pandas().empty:
            logger.warning(
                "JSD heatmap aborted: distances file %s appears empty. Run evaluation first.",
                distances_path,
            )
            return

    # Build the full square matrix (symmetric) for plotting
    matrix_df = _to_ordered_square_matrix(
        df, p1_col="p1", p2_col="p2", value_col="distance"
    )

    # Figure and axis configuration
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, HEATMAP_HEIGHT))

    sns.heatmap(
        matrix_df,
        ax=ax,
        cmap=HEATMAP_CMAP,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 2.2},
        square=True,
        linewidths=0.01,
        linecolor="#CCCCCC",
        cbar_kws={"label": "Jensen-Shannon Divergence", "shrink": 0.5, "pad": 0.04},
    )

    # Minimal axis labeling to meet academic standards; use LaTeX captions externally.
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Ticks formatting for compact printing
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=3.5)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=3.5)

    # Persist the figure
    _save_figure(fig, config.paths.jsd_heatmap_path)
    plt.close(fig)


def moves_distribution(config: Config, top_n: Optional[int] = None) -> None:
    """
    Generate a bar chart showing the number of moves per player.

    Parameters
    ----------
    config : Config
        Project configuration object containing dataset path and output locations.
    top_n : Optional[int]
        If provided, restricts the visualization to the top_n players by move count.

    Rationale:
    - Bars are ordered by descending count to emphasise the most prolific players.
    - Axis titles are explicit and avoid unexplained acronyms to satisfy publication norms.

    Implementation note:
    - This implementation uses Matplotlib's `bar` instead of Seaborn's `barplot`
      to avoid a deprecation/future warning when passing a `palette` without `hue`.
    - Bar colors are mapped from the 'magma' colormap proportionally to each player's count.
    """
    df = _safe_read_parquet(config.paths.dataset_path)
    if df is None:
        logger.error(
            "Moves distribution aborted: dataset file not found at %s",
            config.paths.dataset_path,
        )
        return

    # Compute counts using Polars (fast for large datasets)
    counts = df.group_by("player_name").len().sort("len", descending=True)
    if counts.height == 0:
        logger.warning("Moves distribution: no players found in dataset.")
        return

    if top_n is not None:
        counts = counts.head(top_n)

    pdf = counts.to_pandas()

    # Create a simple bar chart using Matplotlib and map colors with magma colormap
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, DISTRIBUTION_HEIGHT))

    # Data for plotting
    players = pdf["player_name"].astype(str).tolist()
    heights = pdf["len"].astype(float).tolist()
    x = range(len(players))

    # Normalize heights to [0,1] for colormap mapping
    if heights:
        vmin, vmax = min(heights), max(heights)
    else:
        vmin, vmax = 0.0, 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(BAR_CMAP)

    # Map each bar height to an RGBA color
    bar_colors = [cmap(norm(h)) for h in heights]

    # Draw bars
    ax.bar(x, heights, color=bar_colors, edgecolor="#222222", linewidth=0.5)

    # Set tick labels centered under each bar
    ax.set_xticks(list(x))
    ax.set_xticklabels(players, rotation=90, ha="center", fontsize=6)

    # Follow IEEE-friendly labeling conventions
    ax.set_title("")  # Title omitted; prefer captioning in LaTeX
    ax.set_xlabel("Player", fontsize=8)
    ax.set_ylabel("Number of Moves", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=7)

    # Subtle grid behind bars for interpretability
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, linewidth=0.5)
    ax.set_axisbelow(True)
    sns.despine()

    _save_figure(fig, config.paths.moves_distribution_graph_path)
    plt.close(fig)


def stability_heatmap(config: Config) -> None:
    """
    Generate an asymmetric 'stability' heatmap comparing train-set players (rows)
    to test-set players (columns). The diagonal illustrates within-player similarity.

    This function reads a "cross distances" parquet and expects columns:
    - 'p_train' : player name from training set
    - 'p_test'  : player name from test set
    - 'distance' : numeric distance (e.g., Jensen-Shannon)

    If the expected file is missing, the function will log an informative error and return.
    """
    method = config.jsd.method
    kde = config.jsd.kde

    input_path = config.paths.get_full_cross_matrix_path(method, kde)
    df = _safe_read_parquet(input_path)
    if df is None:
        logger.error(
            "Stability heatmap aborted: input file missing (%s). Please run evaluation first.",
            input_path,
        )
        return

    # Determine player ordering from training player column to keep the diagonal aligned.
    try:
        players_ordered = (
            df.select("p_train").unique().sort("p_train").to_series().to_list()
        )
    except Exception:  # pragma: no cover - defensive fallback
        players_ordered = None

    # Pivot to a rectangular matrix. Use pandas pivot and then reindex to maintain order.
    matrix_pd = (
        df.to_pandas()
        .pivot(index="p_train", columns="p_test", values="distance")
        .reindex(index=players_ordered, columns=players_ordered)
    )

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, HEATMAP_HEIGHT))

    sns.heatmap(
        matrix_pd,
        ax=ax,
        cmap=HEATMAP_CMAP,
        annot=True,
        fmt=".4f",
        annot_kws={"size": 2.2},
        square=True,
        linewidths=0.01,
        linecolor="#CCCCCC",
        cbar_kws={"label": "Jensen-Shannon Divergence", "shrink": 0.5, "pad": 0.04},
    )

    # Asymmetric labels emphasize the experimental axes
    ax.set_xlabel("Test Set Players", fontsize=6)
    ax.set_ylabel("Train Set Players", fontsize=6)

    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=3.5)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=3.5)

    out_path = config.paths.jsd_stability_heatmap_path
    _save_figure(fig, out_path)
    logger.info("Asymmetric stability heatmap saved to %s", out_path)
    plt.close(fig)


def learning_curves(config: Config) -> None:
    """
    Generate learning curves (loss and accuracy) from the training history parquet file.

    The resulting figure contains two subplots:
    1. Cross-Entropy Loss on the training set.
    2. Predictive Accuracy on both the training and test sets.
    """
    history_path = config.paths.learning_curves_path
    df = _safe_read_parquet(str(history_path))

    if df is None:
        logger.error(
            "Learning curves aborted: missing %s. Run training first.", history_path
        )
        return

    pdf = df.to_pandas()
    epochs = pdf["epoch"]

    # Création de la figure avec 2 sous-graphes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH * 2.2, DISTRIBUTION_HEIGHT))

    # --- Sous-graphe 1 : Loss ---
    ax1.plot(
        epochs,
        pdf["train_loss"],
        label="Train Loss",
        color="crimson",
        marker="o",
        markersize=3,
        linewidth=1.5,
    )
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=False)

    # --- Sous-graphe 2 : Accuracy ---
    ax2.plot(
        epochs,
        pdf["train_accuracy"],
        label="Train Accuracy",
        color="navy",
        marker="o",
        markersize=3,
        linewidth=1.5,
    )
    ax2.plot(
        epochs,
        pdf["test_accuracy"],
        label="Test Accuracy",
        color="forestgreen",
        marker="s",
        markersize=3,
        linewidth=1.5,
    )
    ax2.set_title("Accuracy (Train vs Test)")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=False)

    sns.despine(fig=fig)
    fig.tight_layout()

    out_path = str(Path(config.paths.result) / "graphics" / "learning_curves.pdf")
    _save_figure(fig, out_path)
    plt.close(fig)


def generate_all_graphics(config: Config) -> None:
    """
    Convenience function to generate all standard graphics in sequence.

    Each plotting function logs its own progress and failures, allowing this routine
    to run unattended in batch evaluation workflows.
    """
    logger.info("Generating learning curves graph...")
    learning_curves(config)

    logger.info("Generating moves distribution graph...")
    moves_distribution(config)

    # Generate method-specific JSD and stability heatmaps (uses templates in config.paths)
    logger.info("Generating JSD and stability heatmaps for configured methods...")
    generate_model_graphics(config)


def generate_model_graphics(config: Config, methods: list | None = None) -> None:
    """
    Generate JSD heatmaps and stability heatmaps for a list of embedding methods.

    Parameters
    ----------
    config : Config
        Project configuration object. The function temporarily adjusts output
        paths so that figures for each method are written to distinct files
        named with the method suffix (e.g. `jsd_heatmap_maia2.pdf`).
    methods : list | None
        If omitted the function will default to a reasonable set covering both
        the installed pipeline (`umap`) and the Maia variants we produce
        (`maia2`, `maia2_ft`, `maia2_ft_mcts`). You can pass any method name that
        matches the naming convention used in the `data/processed` parquet files
        (train_{method}.parquet / test_{method}.parquet).
    """
    if methods is None:
        methods = ["umap", "maia2", "maia2_ft", "maia2_ft_mcts"]

    # Preserve originals
    orig_jsd_path = config.paths.jsd_heatmap_path
    orig_stab_path = config.paths.jsd_stability_heatmap_path
    orig_method = config.jsd.method

    for method in methods:
        logger.info("Generating graphics for method: %s", method)

        # Adjust config to point to this method and set method-specific output paths from templates
        config.jsd.method = method
        jsd_out = config.paths.method_jsd_heatmap_template.format(method=method)
        stab_out = config.paths.method_jsd_stability_template.format(method=method)

        # Temporarily override the output paths used by plotting functions
        config.paths.jsd_heatmap_path = jsd_out
        config.paths.jsd_stability_heatmap_path = stab_out

        try:
            jsd_heatmap(config)
        except Exception as exc:
            logger.error("Failed to generate JSD heatmap for %s: %s", method, exc)

        try:
            stability_heatmap(config)
        except Exception as exc:
            logger.error("Failed to generate stability heatmap for %s: %s", method, exc)

    # Restore originals
    config.paths.jsd_heatmap_path = orig_jsd_path
    config.paths.jsd_stability_heatmap_path = orig_stab_path
    config.jsd.method = orig_method
