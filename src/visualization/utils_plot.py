import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_bar_distribution(
    data,
    title="Distribution",
    xlabel="Categories",
    ylabel="Values",
    output_filename="plot.pdf",
    log_scale=False,
    color="#4C72B0",
    figsize=(10, 6),
    rotate_xticks=45,
):
    """
    Generates an academic-style bar chart (LaTeX style).

    Args:
        data (dict, pd.Series, or pd.DataFrame):
            - If dict: {'Player A': 10, 'Player B': 5}
            - If Series: index=categories, values=heights
        title (str): Chart title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        output_filename (str): Output filename (e.g., 'graph.pdf').
        log_scale (bool): If True, the Y axis is logarithmic.
        color (str): Hex code or color name.
        figsize (tuple): Figure dimensions (width, height).
        rotate_xticks (int): Rotation angle for X labels.
    """

    plt.rcParams.update(
        {
            "font.serif": ["Times New Roman"],
            "axes.labelsize": 12,
            "font.size": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )

    if isinstance(data, dict):
        series = pd.Series(data)
    elif isinstance(data, pd.DataFrame):
        series = data.iloc[:, 0]
    else:
        series = data

    series = series.sort_values(ascending=False)

    categories = series.index
    values = series.to_numpy()

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(
        categories, values, color=color, edgecolor="black", linewidth=0.7, zorder=3
    )

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title, pad=15)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if rotate_xticks:
        plt.xticks(rotation=rotate_xticks, ha="right")

    plt.tight_layout()
    plt.savefig(output_filename, format="pdf", dpi=300)
    print(f"Plot saved to: {output_filename}")


def plot_player_centroids(
    df,
    title="Constellation of Chess Styles",
    xlabel="UMAP Dimension 1",
    ylabel="UMAP Dimension 2",
    output_filename="centroids_map.pdf",
    color="#2E86C1",
    figsize=(12, 10),
):
    """
    Calcule et affiche le barycentre (position moyenne) de chaque joueur.

    Args:
        df (pd.DataFrame): DataFrame contenant 'UMAP1', 'UMAP2' et 'player_name'.
    """
    centroids_df = df.groupby("player_name")[["UMAP1", "UMAP2"]].mean()

    plt.rcParams.update(
        {
            "font.serif": ["Times New Roman"],
            "font.size": 12,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        centroids_df["UMAP1"],
        centroids_df["UMAP2"],
        s=150,
        c=color,
        edgecolors="black",
        linewidth=0.8,
        zorder=3,
        alpha=0.9,
    )

    for player_name, row in centroids_df.iterrows():
        ax.annotate(
            player_name,
            (row["UMAP1"], row["UMAP2"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7),
        )

    ax.set_title(title, pad=20, fontsize=16)
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_filename, format="pdf", dpi=300)
    print(f"Centroids plot saved to: {output_filename}")


def plot_style_comparison(
    df,
    players_to_compare,
    title="Style Comparison",
    output_filename="comparison.pdf",
    figsize=(10, 8),
    palette="tab10",
):
    """
    Affiche une comparaison de densité (KDE) pour une liste de joueurs donnée.

    Args:
        df (pd.DataFrame): DataFrame contenant 'UMAP1', 'UMAP2' et 'player_name'.
        players_to_compare (list): Liste des noms de joueurs (ex: ["Tal", "Capablanca"]).
    """
    subset = df[df["player_name"].isin(players_to_compare)]

    if subset.empty:
        print("Erreur : Aucun des joueurs demandés n'a été trouvé dans le DataFrame.")
        return

    plt.rcParams.update(
        {"font.serif": ["Times New Roman"], "axes.grid": True, "grid.alpha": 0.3}
    )

    fig, ax = plt.subplots(figsize=figsize)

    sns.kdeplot(
        data=subset,
        x="UMAP1",
        y="UMAP2",
        hue="player_name",
        fill=True,
        alpha=0.3,
        palette=palette,
        levels=10,
        thresh=0.05,
        ax=ax,
    )

    ax.set_title(f"{title}: {' vs '.join(players_to_compare)}", pad=20, fontsize=16)
    ax.set_xlabel("UMAP Dimension 1", fontweight="bold")
    ax.set_ylabel("UMAP Dimension 2", fontweight="bold")

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_filename, format="pdf", dpi=300)
    print(f"Comparison plot saved to: {output_filename}")


def plot_distance_heatmap(
    df,
    player1_col="Player1",
    player2_col="Player2",
    distance_col="JSDistance",
    title="Stylistic Divergences Matrix",
    output_filename="heatmap_distances.pdf",
    cmap="Blues",
    figsize=(10, 8),
):
    """
    Generates an academic heatmap from a pairwise distance DataFrame.
    """

    players = pd.unique(df[[player1_col, player2_col]].values.ravel("K"))

    dist_matrix = pd.DataFrame(index=players, columns=players, dtype=float)

    for _, row in df.iterrows():
        dist_matrix.loc[row[player1_col], row[player2_col]] = row[distance_col]
        dist_matrix.loc[row[player2_col], row[player1_col]] = row[distance_col]

    np.fill_diagonal(dist_matrix.values, 0.0)

    plt.rcParams.update(
        {"font.serif": ["Times New Roman"], "font.size": 11, "axes.grid": False}
    )

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        dist_matrix,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "Jensen,Shannon Distance"},
        annot_kws={"size": 10, "weight": "bold"},
    )

    ax.set_title(title, pad=20, fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(output_filename, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Heatmap successfully saved to: {output_filename}")
