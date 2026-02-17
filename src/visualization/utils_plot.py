import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_bar_distribution(
    data,
    title="Distribution",
    xlabel="Catégories",
    ylabel="Valeurs",
    output_filename="plot.pdf",
    log_scale=False,
    color="#4C72B0",
    figsize=(10, 6),
    rotate_xticks=45
):
    """
    Génère un graphique en barres académique (style LaTeX).

    Args:
        data (dict, pd.Series, ou pd.DataFrame): 
            - Si dict: {'Joueur A': 10, 'Joueur B': 5}
            - Si Series: index=catégories, values=hauteurs
        title (str): Titre du graphique.
        xlabel (str): Label de l'axe X.
        ylabel (str): Label de l'axe Y.
        output_filename (str): Nom du fichier de sortie (ex: 'graph.pdf').
        log_scale (bool): Si True, l'axe Y est logarithmique.
        color (str): Code hex ou nom de couleur.
        figsize (tuple): Dimensions de la figure (largeur, hauteur).
        rotate_xticks (int): Angle de rotation des labels X.
    """

    plt.rcParams.update({
        "font.serif": ["Times New Roman"],
        "axes.labelsize": 12,
        "font.size": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3
    })

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
        categories,
        values,
        color=color,
        edgecolor="black",
        linewidth=0.7,
        zorder=3
    )

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, pad=15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if rotate_xticks:
        plt.xticks(rotation=rotate_xticks, ha='right')

    plt.tight_layout()
    plt.savefig(output_filename, format="pdf", dpi=300)
    print(f"Graphique sauvegardé sous : {output_filename}")
