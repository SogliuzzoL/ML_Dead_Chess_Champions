import os

import pandas as pd

from core.config import DATA_FOLDER, DATASET_PATH, MAIA_RESULT_PATH
from visualization.utils_plot import plot_bar_distribution


def plot_games_distributions():
    df = pd.read_parquet(DATASET_PATH)
    player_counts = df["player_name"].value_counts()
    plot_bar_distribution(
        player_counts,
        title="Distribution of moves played per historical champion",
        xlabel="Historical Champions",
        ylabel="Number of Moves",
        output_filename=os.path.join(DATA_FOLDER, "player_distribution.pdf")
    )

    game_counts = df.groupby("player_name")["game_id"].nunique()
    plot_bar_distribution(
        game_counts,
        title="Distribution of games per historical champion",
        xlabel="Historical Champions",
        ylabel="Number of Games",
        output_filename=os.path.join(DATA_FOLDER, "player_game_count.pdf")
    )


def plot_maia_accuracy_distribution():
    df = pd.read_parquet(MAIA_RESULT_PATH)
    plot_bar_distribution(
        df.set_index("player")["maia_accuracy"],
        title="Maia-2 move accuracy per historical champion",
        xlabel="Historical Champions",
        ylabel="Maia-2 move accuracy",
        output_filename=os.path.join(DATA_FOLDER, "maia_accuracy.pdf")
    )
