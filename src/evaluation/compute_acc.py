import polars as pl

from src.core.config import Config
from src.core.utils import getLogger

logger = getLogger()


def compute_accuracy(config: Config):
    df = pl.read_parquet(config.paths.predictions_path)

    df_results = []

    for player in df["player_name"].unique():
        player_df = df.filter(pl.col("player_name") == player)
        baseline_accuracy = (
            player_df["pred_baseline"] == player_df["true_move"]
        ).mean()
        custom_accuracy = (player_df["pred_custom"] == player_df["true_move"]).mean()
        mcts_accuracy = (player_df["pred_mcts"] == player_df["true_move"]).mean()

        df_results.append(
            {
                "player_name": player,
                "baseline_accuracy": baseline_accuracy,
                "custom_accuracy": custom_accuracy,
                "mcts_accuracy": mcts_accuracy,
            }
        )

    df_results = pl.DataFrame(df_results)
    df_results.write_parquet(config.paths.accuracy_path)
    logger.info(f"Player accuracies computed and saved to {config.paths.accuracy_path}")
