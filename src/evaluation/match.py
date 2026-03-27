import os
from datetime import datetime
from typing import List

import chess
import chess.pgn

from src.core.config import Config
from src.core.utils import getLogger
from src.models.maia import MaiaEngine

logger = getLogger()


def run_match_series(
    engine: MaiaEngine, config: Config, player_a: str, player_b: str, num_games: int = 2
) -> List[str]:
    """Exécute une série de matchs entre deux joueurs et sauvegarde les PGNs."""

    # On utilise le dossier d'évaluation de ta config pour créer le dossier des matchs
    pgn_output_dir = os.path.join(config.paths.evaluation_dir, "matches")
    os.makedirs(pgn_output_dir, exist_ok=True)
    series_results = []

    for i in range(num_games):
        board = chess.Board()
        # Alternance des couleurs
        if i % 2 == 0:
            white_name, black_name = player_a, player_b
        else:
            white_name, black_name = player_b, player_a

        game = chess.pgn.Game()
        game.headers["Event"] = f"{white_name} vs {black_name} Match Series"
        game.headers["Site"] = "Local"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = str(i + 1)
        game.headers["White"] = str(white_name)
        game.headers["Black"] = str(black_name)

        node = game

        logger.info(
            f"[Game {i + 1}/{num_games}] Match initialization: {white_name} (White) vs {black_name} (Black)."
        )

        while not board.is_game_over():
            fen = board.fen()
            active_style = white_name if board.turn == chess.WHITE else black_name
            opponent_style = black_name if board.turn == chess.WHITE else white_name

            # Plus aucun Stockfish ! Le MCTS est 100% autonome
            move_uci, _ = engine.predict_mcts(
                fen=fen,
                pgn=str(game),
                active_elo=active_style,
                opponent_elo=opponent_style,
            )

            move_obj = chess.Move.from_uci(move_uci)
            node = node.add_main_variation(move_obj)
            board.push(move_obj)

        result = board.result()
        game.headers["Result"] = result
        series_results.append(result)

        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        filename = f"{timestamp}_{white_name}_vs_{black_name}.pgn"
        filepath = os.path.join(pgn_output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as pgn_file:
            pgn_file.write(str(game))

        logger.info(
            f"[Game {i + 1}/{num_games}] Match concluded. Duration: {board.fullmove_number} moves. Result: {result}."
        )

    logger.info(f"Match series terminated. Aggregate results: {series_results}.")
    return series_results
