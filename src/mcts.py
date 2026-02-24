import chess
import numpy as np
import torch
from maia2 import inference, model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = model.from_pretrained("rapid", DEVICE)
PREPARE = inference.prepare()


class Node:
    def __init__(self, maia_prob=1.0) -> None:
        self.maia_prob = maia_prob
        self.children = {}
        self.visits = 0
        self.value = 0.0

    def compute_Q(self):
        if self.visits == 0:
            return 0.0
        return self.value / self.visits

    def compute_U(self, parent_visits, c_puct=1.0):
        return c_puct * self.maia_prob * np.sqrt(parent_visits) / (1 + self.visits)

    def generate_child(self, fen, activ_elo, opp_elo, threshold=0.1):
        results, _ = inference.inference_each(
            MODEL, PREPARE, fen, activ_elo, opp_elo)
        for move, prob in results:
            if prob > threshold:
                self.children[move] = Node(prob)


class MCTS:
    def __init__(self):
        self.root = Node()

    def run(self, board: chess.Board, num_simulations: int, activ_elo=2500, opp_elo=2500):
        for _ in range(num_simulations):
            self.current_node = self.root
            self.board = board.copy()
            self.path = [self.current_node]
            self.current_node.generate_child(
                self.board.fen(), activ_elo, opp_elo)
