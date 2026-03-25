import os

import chess.pgn as pgn
import pandas as pd
import tqdm

from core.config import ProjectConfig


def extract_opening_stats(config: ProjectConfig):
    import logging

    logger = logging.getLogger(__name__)
    data = []

    for player_id, player_name in config.base_player_dict.items():
        pgn_folder = os.path.join(config.data_folder, player_id)
        if not os.path.exists(pgn_folder):
            continue

        for filename in tqdm.tqdm(os.listdir(pgn_folder), desc=f"Stats: {player_name}"):
            if not filename.endswith(".pgn"):
                continue

            with open(os.path.join(pgn_folder, filename), encoding="utf-8") as f:
                game = pgn.read_game(f)

            if game is None:
                continue

            white_player = game.headers.get("White", "")
            player_color = "White" if player_name in white_player else "Black"
            opening = game.headers.get("ECO", "Unknown")

            data.append(
                {
                    "player_name": player_name,
                    "player_color": player_color,
                    "opening": opening,
                }
            )

    df = pd.DataFrame(data)
    df.to_parquet(config.opening_stats_path)
    logger.info(
        f"Opening statistics successfully persisted to {config.opening_stats_path}"
    )
