import os
import random
from datetime import datetime
from typing import List, Optional

import chess
import chess.pgn
from maia2 import inference

from core.config import RESULT_FOLDER, base_player_dict, logger
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

            move_uci, _ = engine.predict_mcts(
                fen,
                str(game),
                active_elo=active_style,
                opponent_elo=opponent_style
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

    logger.info(
        f"Match series terminated. Aggregate results: {series_results}.")
    return series_results


class MatchNode:
    def __init__(self, left: 'Optional[MatchNode]' = None, right: 'Optional[MatchNode]' = None, player: Optional[str] = None):
        self.left = left
        self.right = right
        self.player = player
        self.winner: Optional[str] = None


class TournamentManager:
    def __init__(self, players: List[str], num_games: int = 2):
        self.players = players
        self.num_games = num_games

    def run_tournament(self):
        raise NotImplementedError(
            "This method must be implemented by the subclass."
        )

    def _determine_winner(self, player_a: str, player_b: str, results: List[str]) -> str:
        score_a = 0.0
        score_b = 0.0

        for i, res in enumerate(results):
            if res == '1/2-1/2':
                score_a += 0.5
                score_b += 0.5
            elif res == '1-0':
                if i % 2 == 0:
                    score_a += 1
                else:
                    score_b += 1
            elif res == '0-1':
                if i % 2 == 0:
                    score_b += 1
                else:
                    score_a += 1

        if score_a > score_b:
            return player_a
        elif score_b > score_a:
            return player_b

        return random.choice([player_a, player_b])


class SingleElimination(TournamentManager):
    def __init__(self, players: List[str], num_games: int = 2):
        super().__init__(players, num_games)
        random.shuffle(self.players)
        self.root = self._build_tree(self.players)

    def _build_tree(self, players: List[str]) -> MatchNode:
        if len(players) == 1:
            return MatchNode(player=players[0])

        mid = len(players) // 2
        left_child = self._build_tree(players[:mid])
        right_child = self._build_tree(players[mid:])

        return MatchNode(left=left_child, right=right_child)

    def _play_match(self, player_a: str, player_b: str) -> str:
        logger.info(f"Scheduled Matchup: {player_a} vs {player_b}")

        results = run_match_series(player_a, player_b, self.num_games)
        winner = self._determine_winner(player_a, player_b, results)

        logger.info(f"Matchup concluded. Victory: {winner}")
        return winner

    def _resolve(self, node: MatchNode) -> str:
        if node.player:
            return node.player

        assert node.left is not None and node.right is not None

        player_a = self._resolve(node.left)
        player_b = self._resolve(node.right)

        node.winner = self._play_match(player_a, player_b)

        logger.info("\n" + "="*50)
        logger.info(
            f"Tournament bracket updated. Advancing player: {node.winner}")
        self.display_bracket(self.root)
        logger.info("="*50 + "\n")

        return node.winner

    def display_bracket(self, node: MatchNode, prefix: str = "", is_left: bool = True, is_root: bool = True):
        if node is None:
            return

        if node.right:
            new_prefix = prefix + \
                ("" if is_root else ("│   " if is_left else "    "))
            self.display_bracket(node.right, new_prefix, False, False)

        name = f"{node.winner} (Winner)" if node.winner else (
            node.player if node.player else "[Pending Match]")

        if is_root:
            logger.info(f"{name}")
        else:
            indicator = "└── " if is_left else "┌── "
            logger.info(f"{prefix}{indicator}{name}")

        if node.left:
            new_prefix = prefix + \
                ("" if is_root else ("    " if is_left else "│   "))
            self.display_bracket(node.left, new_prefix, True, False)

    def run_tournament(self) -> str:
        logger.info("\nInitial Tournament Bracket:")
        self.display_bracket(self.root)
        logger.info("\n" + "="*50 + "\n")

        champion = self._resolve(self.root)

        logger.info(f"\nTournament concluded. Grand Champion: {champion}\n")
        return champion


if __name__ == "__main__":
    champions_list = list(base_player_dict.values())

    _, elo_dict, _ = inference.prepare()
    standard_elos = list(elo_dict.keys())

    all_participants = champions_list + standard_elos
    # all_participants = random.sample(all_participants, 4)

    tournament = SingleElimination(players=all_participants, num_games=2)
    tournament.run_tournament()
