import os
from datetime import datetime

import chess
import chess.pgn

from core.config import RESULT_FOLDER, logger
from core.engine import MaiaEngine


def run_match_series(player_a="Tal", player_b="Karpov", num_games=2):
    engine = MaiaEngine()
    pgn_output_dir = os.path.join(RESULT_FOLDER, "matches")
    os.makedirs(pgn_output_dir, exist_ok=True)
    series_results = []

    for i in range(num_games):
        board = chess.Board()
        if i % 2 == 0:
            white_name, black_name = player_a, player_b
        else:
            white_name, black_name = player_b, player_a

        game = chess.pgn.Game()
        game.headers["Event"] = f"{white_name} vs {black_name} Match Series"
        game.headers["Site"] = "Local"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = str(i + 1)
        game.headers["White"] = white_name
        game.headers["Black"] = black_name

        node = game

        logger.info(
            f"Starting game {i + 1}/{num_games}: {white_name} (White) vs {black_name} (Black)")

        while not board.is_game_over():
            fen = board.fen()
            active_style = white_name if board.turn == chess.WHITE else black_name
            opponent_style = black_name if board.turn == chess.WHITE else white_name

            move_uci, _ = engine.predict_mcts(
                fen,
                str(game),
                active_elo=active_style,
                opponent_elo=opponent_style
            )

            move_obj = chess.Move.from_uci(move_uci)
            node = node.add_main_variation(move_obj)
            prefix = f"{board.fullmove_number}." if board.turn == chess.WHITE else f"{board.fullmove_number}..."
            logger.info(f"{prefix} {board.san(move_obj)}")

            board.push(move_obj)

        result = board.result()
        game.headers["Result"] = result
        logger.info(f"Game {i + 1} result: {result}")
        series_results.append(result)

        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        filename = f"{timestamp}_{white_name}_vs_{black_name}.pgn"
        filepath = os.path.join(pgn_output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as pgn_file:
            pgn_file.write(str(game))

        logger.info(f"Saved game {i + 1} PGN to: {filepath}")

    logger.info(f"Match series completed: {player_a} vs {player_b}")
    logger.info(f"Series results: {series_results}")


if __name__ == "__main__":
    run_match_series("Fischer", "Kasparov", 2)
