import io

import chess
import chess.pgn
import numpy as np
import torch
from maia2 import inference, model


class MaiaEngine:
    def __init__(self, model_type="rapid"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.from_pretrained(model_type, self.device)
        self.prepare = inference.prepare()

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

        result, _ = inference.inference_each(
            self.model, self.prepare, fen, active_elo, opponent_elo)

        for move, _ in result.items():
            board.push_uci(move)
            if not board.is_repetition(2):
                return move, result
            board.pop()

        return list(result.keys())[0], result
