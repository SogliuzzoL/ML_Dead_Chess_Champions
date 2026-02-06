# %%
import os

import chess.pgn

from fetch_games import DATA_FOLDER

player_id = "15940"  # Kasparov's player ID
pgn_path = os.path.join('..', DATA_FOLDER, player_id)
game = None
board = None
for filename in os.listdir(pgn_path):
    if not filename.endswith(".pgn"):
        continue

    file_path = os.path.join(pgn_path, filename)
    with open(file_path, "r", encoding="utf-8") as pgn_file:
        game = chess.pgn.read_game(pgn_file)
        print(f"Game: {game.headers.get('Event', 'N/A')}")  # type: ignore
        print(f"White: {game.headers.get('White', 'N/A')}")  # type: ignore
        print(f"Black: {game.headers.get('Black', 'N/A')}")  # type: ignore
        print(f"Result: {game.headers.get('Result', 'N/A')}")  # type: ignore
        print(f"Date: {game.headers.get('Date', 'N/A')}\n")  # type: ignore
        board = game.board()  # type: ignore
        for move in game.mainline_moves():  # type: ignore
            board.push(move)

        break

board  # type: ignore

# %%
