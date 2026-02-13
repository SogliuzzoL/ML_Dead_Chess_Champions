import os

import chess
import pandas as pd
from chess import pgn

from config import DATA_FOLDER, DATASET_PATH, logger, player_dict


def build_dataset():
    """
    Construit un dataset à partir des fichiers PGN téléchargés pour chaque joueur.
    Le dataset contient les colonnes suivantes :
        - player_name : le nom du joueur (ex: "Capablanca")
        - player_color : la couleur du joueur dans la partie ("white" ou "black")
        - fen : la position de l'échiquier au format FEN avant le coup joué
        - move : le coup joué au format UCI (ex: "e2e4")
        - result : le résultat de la partie ("1-0" pour une victoire des blancs, "0-1" pour une victoire des noirs, "1/2-1/2" pour une nulle)
    Le dataset est sauvegardé au format Parquet à l'emplacement spécifié par DATASET_PATH.
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
