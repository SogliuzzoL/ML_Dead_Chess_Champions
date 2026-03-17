import os

import chess.pgn as pgn
import pandas as pd
import tqdm

from core.config import DATA_FOLDER, OPENING_STATS_PATH, base_player_dict, logger


def extract_opening_stats():
    data = []
    for player_id, player_name in base_player_dict.items():
        logger.info(f"Converting PGN files for player {player_name} (ID: {player_id})")

        pgn_folder = os.path.join(DATA_FOLDER, player_id)
        if not os.path.exists(pgn_folder):
            logger.warning(
                f"No PGN folder found for player {player_name} (ID: {player_id}). Skipping."
            )
            continue

        progress_bar = tqdm.tqdm(os.listdir(pgn_folder))
        for filename in progress_bar:
            if not filename.endswith(".pgn"):
                continue

            pgn_path = os.path.join(pgn_folder, filename)

            with open(pgn_path, encoding="utf-8") as f:
                game = pgn.read_game(f)

            if game is None:
                continue

            white_player = game.headers.get("White", "")
            black_player = game.headers.get("Black", "")

            player_color = "Unknown"
            if player_name in white_player:
                player_color = "White"
            elif player_name in black_player:
                player_color = "Black"

            opening = game.headers.get("ECO", "Unknown")

            data.append(
                {
                    "player_name": player_name,
                    "player_color": player_color,
                    "opening": opening,
                }
            )

    df = pd.DataFrame(data)
    df.to_parquet(OPENING_STATS_PATH, index=False)
