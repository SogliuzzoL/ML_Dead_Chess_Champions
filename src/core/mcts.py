import chess
import numpy as np
import torch
from chess.engine import PovScore, SimpleEngine

# import nnue_interface

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Node:
    def __init__(self, maia_prob=1.0) -> None:
        self.maia_prob = maia_prob
        self.children: dict[str, Node] = {}
        self.visits = 0
        self.value = 0.0
        self.stockfish_score: PovScore | None = None

    def compute_Q(self):
        if self.visits == 0:
            return 0.0
        return self.value / self.visits

    def compute_U(self, parent_visits, c_puct=1.0):
        return c_puct * self.maia_prob * np.sqrt(parent_visits) / (1 + self.visits)

    def generate_child(self, child_generator, fen, activ_elo, opp_elo, threshold=0.01):
        _, results, value = child_generator(fen, activ_elo, opp_elo)
        for move, prob in results.items():
            if prob > threshold:
                self.children[move] = Node(prob)
        return value

    def to_dict(self, move_name="Racine"):
        if self.visits == 0:
            return None

        children_data = []
        for move, child in self.children.items():
            child_dict = child.to_dict(move)
            if child_dict is not None:
                children_data.append(child_dict)

        children_data.sort(key=lambda x: x["visits"], reverse=True)

        return {
            "name": move_name,
            "visits": self.visits,
            "q_value": round(self.compute_Q(), 3),
            "prob": round(self.maia_prob, 3),
            "children": children_data,
        }


class MCTS:
    def __init__(self, child_generator, stockfish: SimpleEngine):
        self.child_generator = child_generator
        self.root = Node()
        self.stockfish: SimpleEngine = stockfish

    def run(
        self,
        board: chess.Board,
        num_simulations: int,
        c_puct=1.5,
        threshold=0.01,
        scale=400.0,
        activ_elo: int | str = 2500,
        opp_elo: int | str = 2500,
    ):
        self.root.generate_child(
            self.child_generator, board.fen(), activ_elo, opp_elo, threshold
        )
        best_root_move = None
        for _ in range(num_simulations):
            current_node = self.root
            sim_board = board.copy()
            path = [current_node]

            while current_node.children:
                best_score = -float("inf")
                best_move = None
                best_child = None

                for move, child in current_node.children.items():
                    score = child.compute_Q() + child.compute_U(
                        current_node.visits, c_puct=c_puct
                    )
                    if score > best_score:
                        best_score = score
                        best_move = move
                        best_child = child

                assert best_move is not None, "No valid moves found"
                assert best_child is not None, "Best child is None"

                sim_board.push_uci(best_move)
                current_node = best_child
                path.append(current_node)

            value = 0

            # if current_node.stockfish_score is None:
            #     # print(nnue_interface.get_evaluation(sim_board.fen()))
            #     analyse = self.stockfish.analyse(sim_board, Limit(depth=5))
            #     score = analyse.get("score", None)
            #     current_node.stockfish_score = score
            #     if score is not None:
            #         cp = score.relative.score(mate_score=100000)
            #         normalized_score = np.tanh(cp / scale)
            #         value -= normalized_score

            if not sim_board.is_game_over():
                value -= current_node.generate_child(
                    self.child_generator, sim_board.fen(), activ_elo, opp_elo, threshold
                )

            for node in reversed(path):
                node.visits += 1
                node.value += value
                value = -value

            best_root_move = None
            max_visits = -1
            for move, child in self.root.children.items():
                if child.visits > max_visits:
                    max_visits = child.visits
                    best_root_move = move

        assert best_root_move is not None, "Best root move is None"

        result = {move: child.maia_prob for move, child in self.root.children.items()}

        # for move, child in self.root.children.items():
        #     print(f"Move: {move}, Visits: {child.visits}, Value: {child.value}, Q: {child.compute_Q():.4f}, U: {child.compute_U(self.root.visits):.4f}, Stockfish Score: {child.stockfish_score}, Maia Prob: {child.maia_prob:.4f}")
        tree_data = self.root.to_dict()
        return best_root_move, result, tree_data
