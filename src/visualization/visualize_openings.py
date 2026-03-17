import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from core.config import OPENING_STATS_PATH, RESULT_FOLDER, logger
from visualization.utils_plot import plot_bar_distribution


def plot_global_top_openings(df: pd.DataFrame, top_n: int = 20):
    valid_openings = df[df["opening"] != "Unknown"]
    opening_counts = valid_openings["opening"].value_counts().head(top_n)

    output_path = os.path.join(RESULT_FOLDER, "global_top_openings.pdf")

    plot_bar_distribution(
        data=opening_counts,
        title=f"Top {top_n} Most Played Openings (ECO Codes)",
        xlabel="ECO Codes",
        ylabel="Number of Games",
        output_filename=output_path,
        rotate_xticks=45,
    )


def plot_individual_opening_profiles(df: pd.DataFrame, top_n: int = 10):
    """
    Generates tailored opening repertoire profiles for each chess champion.
    Produces a side-by-side bar chart (White vs. Black) representing the most
    frequently played openings as a percentage of their total games with that color.
    """
    valid_df = df[df["opening"] != "Unknown"]

    players = valid_df["player_name"].unique()

    plt.rcParams.update(
        {
            "font.serif": ["Times New Roman"],
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )

    for player in players:
        player_df = valid_df[valid_df["player_name"] == player]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
        fig.suptitle(
            f"Opening Repertoire Profile: {player}",
            fontsize=16,
            fontweight="bold",
            y=1.05,
        )

        for idx, color in enumerate(["White", "Black"]):
            color_df = player_df[player_df["player_color"] == color]

            if not color_df.empty:
                top_openings = (
                    color_df["opening"].value_counts(normalize=True).head(top_n) * 100
                )

                sns.barplot(
                    x=top_openings.index,
                    y=top_openings.values,
                    hue=top_openings.index,
                    ax=axes[idx],
                    palette="YlGnBu" if color == "White" else "RdPu",
                    edgecolor="black",
                    linewidth=0.7,
                    zorder=3,
                    legend=False,
                )

                axes[idx].set_title(
                    f"Top {top_n} Openings as {color}", pad=15, fontweight="bold"
                )
                axes[idx].set_xlabel("ECO Codes", fontweight="bold")

                axes[idx].set_ylabel("Usage Percentage (%)", fontweight="bold")
                axes[idx].tick_params(axis="x", rotation=45)

                axes[idx].spines["top"].set_visible(False)
                axes[idx].spines["right"].set_visible(False)
            else:
                axes[idx].set_title(f"No empirical data available for {color}", pad=15)
                axes[idx].axis("off")

        plt.tight_layout()

        output_filename = f"opening_profile_{player.lower().replace(' ', '_')}.pdf"
        output_path = os.path.join(RESULT_FOLDER, output_filename)

        plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(
            f"Repertoire profile successfully generated and saved to: {output_path}"
        )


def run_individual_profiles():
    if not os.path.exists(OPENING_STATS_PATH):
        logger.info(
            f"Error: The required dataset {OPENING_STATS_PATH} could not be located."
        )
        return

    df = pd.read_parquet(OPENING_STATS_PATH)

    logger.info("Initiating the generation of the global top openings visualization...")
    plot_global_top_openings(df)

    logger.info(
        "Initiating the generation of individual opening profiles for all champions..."
    )
    plot_individual_opening_profiles(df)
