import chess
import pandas as pd
from maia2 import utils

from config import DATASET_PATH

if __name__ == "__main__":
    df = pd.read_parquet(DATASET_PATH)
    fen = df.iloc[0]["fen"]
    board = chess.Board(fen)
    tensor = utils.board_to_tensor(board)
