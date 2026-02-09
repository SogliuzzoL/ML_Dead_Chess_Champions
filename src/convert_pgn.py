import os

import chess
import pandas as pd
from chess import pgn

from config import DATA_FOLDER, logger, player_dict

if __name__ == "__main__":
    data = []
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
                        "move": move_uci
                    })
                board.push(move)

    df = pd.DataFrame(
        data, columns=["player_name", "player_color", "fen", "move"])
    df.to_parquet(os.path.join(DATA_FOLDER, "chess_positions.parquet"))
