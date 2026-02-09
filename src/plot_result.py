import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

try:
    from config import DATA_FOLDER, MAIA_RESULT_PATH
except ImportError:
    sys.exit(1)


def generate_plot():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.autolayout": True,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })

    if not os.path.exists(MAIA_RESULT_PATH):
        return

    df = pd.read_parquet(MAIA_RESULT_PATH)
    df = df.sort_values("maia_accuracy", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.barh(
        df["player"],
        df["maia_accuracy"],
        color="#2F5597",
        edgecolor="black",
        linewidth=0.7,
        alpha=0.9
    )

    ax.set_xlabel("Top-1 Accuracy (Zero-Shot Alignment)")
    ax.set_title("Stylistic Agreement: Maia2 (Rapid) vs Historical Champions",
                 pad=15, fontweight="bold")

    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)

    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.01

        ax.text(
            label_x_pos,
            bar.get_y() + bar.get_height()/2,
            f"{width:.1%}",
            va='center',
            ha='left',
            color='black',
            fontsize=10,
            fontweight='medium'
        )

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    pdf_path = os.path.join(DATA_FOLDER, "maia_accuracy.pdf")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')

    png_path = os.path.join(DATA_FOLDER, "maia_accuracy.png")
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')

    print(f"{pdf_path}")
    print(f"{png_path}")


if __name__ == "__main__":
    generate_plot()
