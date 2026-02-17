import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_bar_distribution(
    data,
    title="Distribution",
    xlabel="Categories",
    ylabel="Values",
    output_filename="plot.pdf",
    log_scale=False,
    color="#4C72B0",
    figsize=(10, 6),
    rotate_xticks=45
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
    print(f"Plot saved to: {output_filename}")
