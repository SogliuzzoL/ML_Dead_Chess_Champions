import pickle

import chess
import cuml.accel as accel
import torch
from cuml.manifold import UMAP
from maia2.utils import board_to_tensor

accel.install(log_level="debug")


class StyleUMAP(UMAP):
    def save_model(self, path):
        pickle.dump(self, open(path, "wb"))

    def load_model(self, path):
        return pickle.load(open(path, "rb"))


def position_to_vector(fen: str, move: str) -> torch.Tensor:
    board = chess.Board(fen)
    board_before = board_to_tensor(board)
    board.push_uci(move)
    board_after = board_to_tensor(board)

    vector = torch.cat((board_before, board_after), dim=0).flatten()

    return vector
