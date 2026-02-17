import os

import chess
import pandas as pd
from chess import pgn

from core.config import DATA_FOLDER, DATASET_PATH, logger, player_dict


def build_dataset():
    """
    Builds a dataset from downloaded PGN files for each player.
    The dataset contains the following columns:
        - player_name : the player's name (e.g., "Capablanca")
        - player_color : the player's color in the game ("white" or "black")
        - fen : the board position in FEN format before the move is played
        - move : the move played in UCI format (e.g., "e2e4")
        - result : the game result ("1-0" for a White win, "0-1" for a Black win, "1/2-1/2" for a draw)
    The dataset is saved in Parquet format at the location specified by DATASET_PATH.
    """
    data = []
    data_count = {}
    for player_id, player_name in player_dict.items():
        logger.info(
            f"Converting PGN files for player {player_name} (ID: {player_id})")

        pgn_folder = os.path.join(DATA_FOLDER, player_id)
        if not os.path.exists(pgn_folder):
            logger.warning(
                f"No PGN folder found for player {player_name} (ID: {player_id}). Skipping.")
            continue

        for filename in os.listdir(pgn_folder):
            if not filename.endswith(".pgn"):
                continue

            pgn_path = os.path.join(pgn_folder, filename)
            game = pgn.read_game(open(pgn_path))
            if game is None:
                logger.warning(
                    f"Failed to read game from file {filename} for player {player_name}. Skipping.")
                continue

            header = game.headers
            white_player = header.get("White", "")
            black_player = header.get("Black", "")

            player_color = None

            if player_name in white_player:
                player_color = chess.WHITE
            elif player_name in black_player:
                player_color = chess.BLACK
            else:
                logger.warning(
                    f"Player {player_name} not found in game headers for file {filename}. Skipping.")
                continue

            result = header.get("Result", "")
            if result not in ["1-0", "0-1", "1/2-1/2"]:
                logger.warning(
                    f"Unexpected game result '{result}' in file {filename} for player {player_name}. Skipping.")
                continue

            if player_name not in data_count:
                data_count[player_name] = 1
            else:
                data_count[player_name] += 1

            logger.info(f"Processing game {filename} for player {player_name}")
            board = game.board()
            for move in game.mainline_moves():
                if board.turn == player_color:
                    color = "white" if board.turn == chess.WHITE else "black"
                    fen = board.fen()
                    move_uci = move.uci()

                    data.append({
                        "player_name": player_name,
                        "player_color": color,
                        "fen": fen,
                        "move": move_uci,
                        "result": result,
                        "game_id": filename.split(".")[0]
                    })
                board.push(move)

    df = pd.DataFrame(
        data, columns=["player_name", "player_color", "fen", "move", "result", "game_id"])
    df.to_parquet(DATASET_PATH)


if __name__ == "__main__":
    build_dataset()
