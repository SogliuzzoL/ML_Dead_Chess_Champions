"""Plot difference heatmaps between chess_champion_distances and UMAP test distances.

This script loads:
 - `data/processed/chess_champion_distances.parquet` (the 'chess' reference)
 - `results/evaluation/distances_test_umap.parquet` (the UMAP test distances)

It constructs ordered square matrices for both, aligns them on the same player order,
computes two difference matrices:
 - signed diff = chess - umap
 - absolute diff = |chess - umap|

It then plots and saves two heatmaps using the project's exact JSD heatmap styling
for the absolute diff (magma_r) and a centered diverging style for the signed diff.

Outputs:
 - results/graphics/jsd_heatmap_diff_signed_chess_minus_umap.pdf
 - results/graphics/jsd_heatmap_diff_abs_chess_vs_umap.pdf
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

# Styling constants to match project
FIG_WIDTH = 3.5
HEATMAP_HEIGHT = 4.5
HEATMAP_CMAP = "magma_r"

# Input paths
CHESS_INPUT = Path("data/processed/chess_champion_distances.parquet")
UMAP_DISTANCES = Path("results/evaluation/distances_test_umap.parquet")

# Output paths
OUT_SIGNED = Path("results/graphics/jsd_heatmap_diff_signed_chess_minus_umap.pdf")
OUT_ABS = Path("results/graphics/jsd_heatmap_diff_abs_chess_vs_umap.pdf")

OUT_SIGNED.parent.mkdir(parents=True, exist_ok=True)

# Seaborn theme matching project
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


def _to_square_matrix(df: pl.DataFrame) -> pd.DataFrame:
    """Convert a Polars DataFrame (long or wide) into a square pandas DataFrame.

    Supported input forms:
    - long: columns containing 'p1', 'p2', 'distance' (case-insensitive)
    - wide: a column with player names used as index, or columns as player names
    """
    cols = [c.lower() for c in df.columns]
    if set(["p1", "p2", "distance"]).issubset(set(cols)):
        col_map = {c.lower(): c for c in df.columns}
        p1_col = col_map["p1"]
        p2_col = col_map["p2"]
        value_col = col_map["distance"]

        p1_names = df.select(pl.col(p1_col)).unique().to_series().to_list()
        p2_names = df.select(pl.col(p2_col)).unique().to_series().to_list()
        players = sorted(set(p1_names) | set(p2_names))

        mirror = df.select(
            [
                pl.col(p2_col).alias(p1_col),
                pl.col(p1_col).alias(p2_col),
                pl.col(value_col),
            ]
        )
        diag = pl.DataFrame(
            {p1_col: players, p2_col: players, value_col: [0.0] * len(players)}
        )
        combined = pl.concat([df, mirror, diag]).unique(subset=[p1_col, p2_col])
        matrix_pd = (
            combined.to_pandas()
            .pivot(index=p1_col, columns=p2_col, values=value_col)
            .reindex(index=players, columns=players)
        )
        return matrix_pd

    # wide-format
    pdf = df.to_pandas()

    # If there is a candidate column to use as index (object dtype, unique values == rows)
    candidate_index_cols = [
        c
        for c in pdf.columns
        if pdf[c].dtype == object and pdf[c].nunique() == pdf.shape[0]
    ]
    if candidate_index_cols:
        idx_col = candidate_index_cols[0]
        matrix_pd = pdf.set_index(idx_col)
    else:
        matrix_pd = pdf.copy()

    # If still not square, try transpose, otherwise if number of rows == number of columns set index to column names
    if matrix_pd.shape[0] != matrix_pd.shape[1]:
        t = matrix_pd.T
        if t.shape[0] == t.shape[1]:
            matrix_pd = t
        else:
            if matrix_pd.shape[0] == len(matrix_pd.columns):
                matrix_pd.index = matrix_pd.columns

    return matrix_pd


# Read inputs (will raise if files missing — can add handling if you prefer)
ch_df = pl.read_parquet(str(CHESS_INPUT))
umap_df = pl.read_parquet(str(UMAP_DISTANCES))

# Convert to square matrices
ch_mat = _to_square_matrix(ch_df)
um_mat = _to_square_matrix(umap_df)

# Normalize labels (strip quotes) and ensure strings
ch_mat.index = ch_mat.index.astype(str).str.replace('"', "")
ch_mat.columns = ch_mat.columns.astype(str).str.replace('"', "")
um_mat.index = um_mat.index.astype(str).str.replace('"', "")
um_mat.columns = um_mat.columns.astype(str).str.replace('"', "")

# Unified player order (union). Using union ensures we keep all players; missing entries become NaN
players = sorted(set(ch_mat.index.tolist()) | set(um_mat.index.tolist()))

ch_mat = ch_mat.reindex(index=players, columns=players)
um_mat = um_mat.reindex(index=players, columns=players)

# Convert to numeric and compute diffs
ch_values = ch_mat.apply(pd.to_numeric, errors="coerce")
um_values = um_mat.apply(pd.to_numeric, errors="coerce")

diff_signed = ch_values - um_values
diff_abs = diff_signed.abs()


# Plot helpers
def _save_heatmap(
    matrix, out_path, cmap, fmt=".2f", center=None, annot=True, title=None
):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, HEATMAP_HEIGHT))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        annot_kws={"size": 2.2},
        square=True,
        linewidths=0.01,
        linecolor="#CCCCCC",
        cbar_kws={
            "label": "Jensen-Shannon Divergence difference",
            "shrink": 0.5,
            "pad": 0.04,
        },
        center=center,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(
        [str(x) for x in matrix.columns], rotation=90, ha="center", fontsize=3.5
    )
    ax.set_yticklabels([str(y) for y in matrix.index], rotation=0, fontsize=3.5)
    if title:
        plt.title(title)
    plt.tight_layout()
    fig.savefig(str(out_path), format="pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)


# Signed diff: diverging colormap centered at zero
_save_heatmap(
    diff_signed,
    OUT_SIGNED,
    cmap="coolwarm",
    fmt=".2f",
    center=0.0,
    annot=True,
    title="Chess - UMAP (signed)",
)

# Absolute diff: project style
_save_heatmap(
    diff_abs,
    OUT_ABS,
    cmap=HEATMAP_CMAP,
    fmt=".3f",
    center=None,
    annot=True,
    title="|Chess - UMAP| (absolute)",
)

print(f"Saved signed diff heatmap to {OUT_SIGNED}")
print(f"Saved absolute diff heatmap to {OUT_ABS}")
