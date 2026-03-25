import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from core.config import ProjectConfig
from visualization.utils_plot import plot_bar_distribution

logger = logging.getLogger(__name__)


def plot_global_top_openings(config: ProjectConfig, df: pd.DataFrame, top_n: int = 20):
    """
    Generates a global visualization of the most frequently utilized chess openings
    across the entire champion dataset.
    """
    valid_openings = df[df["opening"] != "Unknown"]
    opening_counts = valid_openings["opening"].value_counts().head(top_n)

    output_path = os.path.join(config.result_folder, "global_top_openings.pdf")

    plot_bar_distribution(
        data=opening_counts,
        title=f"Top {top_n} Most Played Openings (ECO Codes)",
        xlabel="ECO Codes",
        ylabel="Number of Games",
        output_filename=output_path,
        rotate_xticks=45,
    )


def plot_individual_opening_profiles(
    config: ProjectConfig, df: pd.DataFrame, top_n: int = 10
):
    """
    Constructs comprehensive opening repertoire profiles for each individual champion,
    presenting a comparative analysis of White versus Black strategies as a percentage
    of total games played with each color.
    """
    valid_df = df[df["opening"] != "Unknown"]
    players = valid_df["player_name"].unique()

    # Configuring academic visual aesthetics
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    for player in players:
        player_df = valid_df[valid_df["player_name"] == player]
        fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
        fig.suptitle(
            f"Opening Repertoire Profile: {player}",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        for idx, color in enumerate(["White", "Black"]):
            color_df = player_df[player_df["player_color"] == color]

            if not color_df.empty:
                counts = color_df["opening"].value_counts().head(top_n)
                # Calculating usage as a percentage of total games for the specific color
                opening_pct = (counts / counts.sum()) * 100

                sns.barplot(
                    x=opening_pct.index,
                    y=opening_pct.values,
                    ax=axes[idx],
                    palette="viridis" if color == "White" else "magma",
                )

                axes[idx].set_title(
                    f"Top {top_n} Openings as {color}", pad=15, fontweight="bold"
                )
                axes[idx].set_xlabel("ECO Codes", fontweight="bold")
                axes[idx].set_ylabel("Usage Frequency (%)", fontweight="bold")
                axes[idx].tick_params(axis="x", rotation=45)
            else:
                axes[idx].set_title(f"Insufficient Empirical Data for {color}", pad=15)
                axes[idx].axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Dynamic path resolution for individual PDF artifacts
        output_filename = f"opening_profile_{player.lower().replace(' ', '_')}.pdf"
        output_path = os.path.join(config.result_folder, output_filename)

        plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(
            f"Repertoire profile for {player} successfully generated and saved to: {output_path}"
        )


def run_individual_profiles(config: ProjectConfig):
    """
    Orchestrates the entire opening analysis workflow, verifying dataset integrity
    before initiating the graphical rendering sequence.
    """
    if not os.path.exists(config.opening_stats_path):
        logger.error(
            f"Critical Error: Opening statistics dataset not found at {config.opening_stats_path}"
        )
        return

    logger.info(f"Loading opening statistics from {config.opening_stats_path}")
    df = pd.read_parquet(config.opening_stats_path)

    logger.info("Initiating global opening distribution visualization...")
    plot_global_top_openings(config, df)

    logger.info("Generating individual champion repertoire profiles...")
    plot_individual_opening_profiles(config, df)
