import os

import pandas as pd

from config import DATA_FOLDER, DATASET_PATH, MAIA_RESULT_PATH
from utils_plot import plot_bar_distribution

if __name__ == "__main__":
    # Plot de la distribution du nombre de coups joués par champion historique
    df = pd.read_parquet(DATASET_PATH)
    player_counts = df["player_name"].value_counts()
    plot_bar_distribution(
        player_counts,
        title="Distribution du nombre de coups joués par champion historique",
        xlabel="Champions Historiques",
        ylabel="Nombre de Coups",
        output_filename=os.path.join(DATA_FOLDER, "player_distribution.pdf")
    )

    # Plot de la distribution du nombre de parties par champion historique
    game_counts = df.groupby("player_name")["game_id"].nunique()
    plot_bar_distribution(
        game_counts,
        title="Distribution du nombre de parties par champion historique",
        xlabel="Champions Historiques",
        ylabel="Nombre de Parties",
        output_filename=os.path.join(DATA_FOLDER, "player_game_count.pdf")
    )

    # Plot de la précision (move-accuracy) de Maia-2 par champion historique
    df = pd.read_parquet(MAIA_RESULT_PATH)
    plot_bar_distribution(
        df.set_index("player")["maia_accuracy"],
        title="Précision (move-accuracy) de Maia-2 par champion historique",
        xlabel="Champions Historiques",
        ylabel="Précision (move-accuracy) de Maia-2",
        output_filename=os.path.join(DATA_FOLDER, "maia_accuracy.pdf")
    )
