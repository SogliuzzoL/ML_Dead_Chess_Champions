import os

import chess.pgn

from fetch_games import DATA_FOLDER

player_id = "15940"  # Kasparov's player ID
pgn_path = os.path.join(DATA_FOLDER, player_id)
first = True
game_count = 0

for filename in os.listdir(pgn_path):
    if not filename.endswith(".pgn"):
        continue

    game_count += 1

    if first:
        file_path = os.path.join(pgn_path, filename)
        with open(file_path, "r", encoding="utf-8") as pgn_file:
            game = chess.pgn.read_game(pgn_file)
            print(f"Game: {game.headers.get('Event', 'N/A')}")
            print(f"White: {game.headers.get('White', 'N/A')}")
            print(f"Black: {game.headers.get('Black', 'N/A')}")
            print(f"Result: {game.headers.get('Result', 'N/A')}")
            print(f"Date: {game.headers.get('Date', 'N/A')}\n")
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
            print(board)
        first = False

print(f"Total games: {game_count}")
