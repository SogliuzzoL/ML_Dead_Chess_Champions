import io

import chess
import chess.pgn
import numpy as np
import torch
from maia2 import inference, model

from .mcts import MCTS


class MaiaEngine:
    def __init__(self, model_type="rapid"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.from_pretrained(model_type, self.device)
        self.prepare = inference.prepare()

    def get_board_from_fen(self, fen, pgn):
        board = chess.Board()
        if pgn != '':
            try:
                pgn_io = io.StringIO(pgn)
                game = chess.pgn.read_game(pgn_io)
            except Exception:
                game = None
            if game is not None:
                for move in game.mainline_moves():
                    board.push(move)
            else:
                board = chess.Board(fen)
        return board

    def predict_move(self, fen, active_elo=2500, opponent_elo=2500):
        result, _ = inference.inference_each(
            self.model, self.prepare, fen, active_elo, opponent_elo)
        best_move = list(result.keys())[0]
        return best_move, result

    def predict_proba_move(self, fen, active_elo=2500, opponent_elo=2500):
        result, _ = inference.inference_each(
            self.model, self.prepare, fen, active_elo, opponent_elo)
        normalized_result = {
            move: prob / sum(result.values()) for move, prob in result.items()}
        proba_move = np.random.choice(
            list(normalized_result.keys()), p=list(normalized_result.values()))
        return proba_move, result

    def predict_move_without_repetition(self, fen, pgn, active_elo=2500, opponent_elo=2500):
        board = self.get_board_from_fen(fen, pgn)

        result, _ = inference.inference_each(
            self.model, self.prepare, fen, active_elo, opponent_elo)

        for move, _ in result.items():
            board.push_uci(move)
            if not board.is_repetition(2):
                return move, result
            board.pop()

        return list(result.keys())[0], result

    def predict_mcts(self, fen, pgn, num_simulations=50, max_depth=4, threshold=0.05, penalty_value=10.0, active_elo=2500, opponent_elo=2500):
        board = self.get_board_from_fen(fen, pgn)
        mcts = MCTS(self.model, self.prepare)
        best_move, result = mcts.run(board, num_simulations, max_depth,
                                     threshold=threshold, penalty_value=penalty_value, activ_elo=active_elo, opp_elo=opponent_elo)

        return best_move, result
