import os

import chess
import pandas as pd
import tqdm
from chess import pgn
from sklearn.model_selection import train_test_split

from core.config import ProjectConfig


def build_dataset(config: ProjectConfig):
    """
    Constructs a unified dataset from raw PGN files utilizing the dynamic ProjectConfig.
    """
    import logging

    logger = logging.getLogger(__name__)
    data = []

    for player_id, player_name in config.base_player_dict.items():
        logger.info(f"Processing PGN files for {player_name} (ID: {player_id})")
        pgn_folder = os.path.join(config.data_folder, player_id)

        if not os.path.exists(pgn_folder):
            logger.warning(f"PGN directory absent for {player_name}. Skipping.")
            continue

        for filename in tqdm.tqdm(os.listdir(pgn_folder), desc=player_name):
            if not filename.endswith(".pgn"):
                continue

            pgn_path = os.path.join(pgn_folder, filename)
            with open(pgn_path, encoding="utf-8") as f:
                game = pgn.read_game(f)

            if game is None:
                continue

            # Determine player color and result
            white_player = game.headers.get("White", "")
            result = game.headers.get("Result", "*")
            player_color = chess.WHITE if player_name in white_player else chess.BLACK

            board = game.board()
            for move in game.mainline_moves():
                if board.turn == player_color:
                    data.append(
                        {
                            "game_id": filename.split(".")[0],
                            "round": board.fullmove_number,
                            "player_name": player_name,
                            "player_color": "white"
                            if board.turn == chess.WHITE
                            else "black",
                            "fen": board.fen(),
                            "move": move.uci(),
                            "repetition": board.is_repetition(2),
                            "result": result,
                        }
                    )
                board.push(move)

    df = pd.DataFrame(data, columns=config.dataset_col_order)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    df_train.to_parquet(config.train_set_path)
    df_test.to_parquet(config.test_set_path)
    df.to_parquet(config.dataset_path)
    logger.info(
        f"Dataset construction complete. Train/Test split saved to {config.data_folder}"
    )
