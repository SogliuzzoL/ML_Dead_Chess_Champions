import itertools
import os
import random
from datetime import datetime
from typing import List, Optional

import chess
import chess.pgn
import numpy as np
from chess.engine import SimpleEngine
from maia2 import inference

from core.config import RESULT_FOLDER, STOCKFISH_MODEL_PATH, base_player_dict, logger
from core.engine import MaiaEngine


def run_match_series(player_a="Tal", player_b="Karpov", num_games=2):
    engine = MaiaEngine()
    stockfish = SimpleEngine.popen_uci(STOCKFISH_MODEL_PATH)
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
                stockfish,
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
    stockfish.quit()
    return series_results


class MatchNode:
    def __init__(
        self,
        left: "Optional[MatchNode]" = None,
        right: "Optional[MatchNode]" = None,
        player: Optional[str] = None,
    ):
        self.left = left
        self.right = right
        self.player = player
        self.winner: Optional[str] = None


class TournamentManager:
    def __init__(self, players: List[str], num_games: int = 2):
        self.players = players
        self.num_games = num_games

    def run_tournament(self):
        raise NotImplementedError("This method must be implemented by the subclass.")

    def _determine_winner(
        self, player_a: str, player_b: str, results: List[str]
    ) -> str:
        score_a = 0.0
        score_b = 0.0

        for i, res in enumerate(results):
            if res == "1/2-1/2":
                score_a += 0.5
                score_b += 0.5
            elif res == "1-0":
                if i % 2 == 0:
                    score_a += 1
                else:
                    score_b += 1
            elif res == "0-1":
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

        logger.info("=" * 50)
        logger.info(f"Tournament bracket updated. Advancing player: {node.winner}")
        self.display_bracket(self.root)
        logger.info("=" * 50 + "\n")

        return node.winner

    def display_bracket(
        self,
        node: MatchNode,
        prefix: str = "",
        is_left: bool = True,
        is_root: bool = True,
    ):
        if node is None:
            return

        if node.right:
            new_prefix = prefix + ("" if is_root else ("│   " if is_left else "    "))
            self.display_bracket(node.right, new_prefix, False, False)

        name = (
            f"{node.winner} (Winner)"
            if node.winner
            else (node.player if node.player else "[Pending Match]")
        )

        if is_root:
            logger.info(f"{name}")
        else:
            indicator = "└── " if is_left else "┌── "
            logger.info(f"{prefix}{indicator}{name}")

        if node.left:
            new_prefix = prefix + ("" if is_root else ("    " if is_left else "│   "))
            self.display_bracket(node.left, new_prefix, True, False)

    def run_tournament(self) -> str:
        logger.info("Initial Tournament Bracket:")
        self.display_bracket(self.root)
        logger.info("=" * 50 + "\n")

        champion = self._resolve(self.root)

        logger.info(f"Tournament concluded. Grand Champion: {champion}\n")
        return champion


class RoundRobin(TournamentManager):
    def __init__(self, players: List[str], num_games: int = 2):
        super().__init__(players, num_games)
        self.scores = {player: 0.0 for player in self.players}

    def run_tournament(self) -> str:
        logger.info("=" * 50)
        logger.info("Starting Round Robin tournament")
        logger.info("=" * 50 + "\n")

        matchups = list(itertools.combinations(self.players, 2))

        for player_a, player_b in matchups:
            logger.info(f"Scheduled match: {player_a} vs {player_b}")

            results = run_match_series(player_a, player_b, self.num_games)

            for i, res in enumerate(results):
                if res == "1/2-1/2":
                    self.scores[player_a] += 0.5
                    self.scores[player_b] += 0.5
                elif res == "1-0":
                    if i % 2 == 0:
                        self.scores[player_a] += 1.0
                    else:
                        self.scores[player_b] += 1.0
                elif res == "0-1":
                    if i % 2 == 0:
                        self.scores[player_b] += 1.0
                    else:
                        self.scores[player_a] += 1.0

        standings = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)

        logger.info("=" * 50)
        logger.info("Final Round Robin Standings,")
        for rank, (player, score) in enumerate(standings, 1):
            logger.info(f"{rank}. {player} with {score} points")
        logger.info("=" * 50 + "\n")

        champion = standings[0][0]
        logger.info(f"The undisputed champion is {champion}!\n")
        return champion


class SwissSystem(TournamentManager):
    def __init__(self, players: List[str], num_games: int = 2, num_rounds: int = 5):
        super().__init__(players, num_games)
        self.num_rounds = num_rounds
        self.scores = {p: 0.0 for p in self.players}
        self.history = {p: set() for p in self.players}

        self.received_bye = set()

    def _generate_pairings(self) -> tuple[List[tuple], Optional[str]]:
        sorted_players = sorted(
            self.scores.keys(), key=lambda p: self.scores[p], reverse=True
        )
        pairings = []
        paired = set()
        bye_player = None

        if len(sorted_players) % 2 != 0:
            for p in reversed(sorted_players):
                if p not in self.received_bye:
                    bye_player = p
                    self.received_bye.add(p)
                    paired.add(p)
                    logger.info(
                        f"The player {p} receives a Bye this round due to an odd number of participants."
                    )
                    break

        for i, p1 in enumerate(sorted_players):
            if p1 in paired:
                continue

            opponent_found = False

            for j in range(i + 1, len(sorted_players)):
                p2 = sorted_players[j]

                if p2 not in paired and p2 not in self.history[p1]:
                    pairings.append((p1, p2))
                    paired.add(p1)
                    paired.add(p2)
                    self.history[p1].add(p2)
                    self.history[p2].add(p1)
                    opponent_found = True
                    break

            if not opponent_found:
                for j in range(i + 1, len(sorted_players)):
                    p2 = sorted_players[j]
                    if p2 not in paired:
                        pairings.append((p1, p2))
                        paired.add(p1)
                        paired.add(p2)
                        self.history[p1].add(p2)
                        self.history[p2].add(p1)
                        logger.warning(
                            f"Forced pairing between {p1} and {p2}, they have already faced each other."
                        )
                        break

        return pairings, bye_player

    def run_tournament(self) -> str:
        logger.info("=" * 50)
        logger.info(
            f"Starting Swiss System tournament, {self.num_rounds} rounds scheduled"
        )
        logger.info("=" * 50 + "\n")

        for round_num in range(1, self.num_rounds + 1):
            logger.info(f"--- ROUND {round_num} ---")

            pairings, bye_player = self._generate_pairings()

            if bye_player:
                self.scores[bye_player] += 1.0
                logger.info(
                    f"{bye_player} receives a Bye and is awarded 1 point for this round."
                )

            for p1, p2 in pairings:
                logger.info(
                    f"Scheduled match: {p1} ({self.scores[p1]} pts) vs {p2} ({self.scores[p2]} pts)"
                )

                results = run_match_series(p1, p2, self.num_games)

                for i, res in enumerate(results):
                    if res == "1/2-1/2":
                        self.scores[p1] += 0.5
                        self.scores[p2] += 0.5
                    elif res == "1-0":
                        if i % 2 == 0:
                            self.scores[p1] += 1.0
                        else:
                            self.scores[p2] += 1.0
                    elif res == "0-1":
                        if i % 2 == 0:
                            self.scores[p2] += 1.0
                        else:
                            self.scores[p1] += 1.0

        standings = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)

        logger.info("=" * 50)
        logger.info("Final Swiss System Standings")
        for rank, (player, score) in enumerate(standings, 1):
            logger.info(f"{rank}. {player} with {score} points")
        logger.info("=" * 50 + "\n")

        champion = standings[0][0]
        logger.info(f"The winner of the Swiss tournament is {champion}!\n")
        return champion


if __name__ == "__main__":
    champions_list = list(base_player_dict.values())

    _, elo_dict, _ = inference.prepare()
    standard_elos = list(elo_dict.keys())

    all_participants = champions_list + standard_elos
    # all_participants = random.sample(all_participants, 5)

    # tournament = SingleElimination(players=all_participants, num_games=2)

    n_rounds = int(np.log2(len(all_participants))) + 1
    tournament = SwissSystem(players=all_participants, num_games=2, num_rounds=n_rounds)

    tournament.run_tournament()
