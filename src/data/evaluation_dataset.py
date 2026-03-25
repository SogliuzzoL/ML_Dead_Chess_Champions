import chess
import pandas as pd
import torch
from maia2.utils import board_to_tensor, mirror_move
from torch.utils.data import Dataset


class EvaluationDataset(Dataset):
    def __init__(self, data_path, player_to_idx, all_moves_dict, base_elo_idx):
        self.df = pd.read_parquet(data_path)
        self.player_to_idx = player_to_idx
        self.all_moves_dict = all_moves_dict
        self.base_elo_idx = base_elo_idx
        self.num_moves = len(all_moves_dict)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        board = chess.Board(row["fen"])
        move_uci = row["move"]

        if row["player_color"] == "black":
            board = board.mirror()
            move_uci = mirror_move(move_uci)

        board_tensor = board_to_tensor(board)

        legal_mask = torch.zeros(self.num_moves, dtype=torch.bool)
        for move in board.legal_moves:
            if move.uci() in self.all_moves_dict:
                legal_mask[self.all_moves_dict[move.uci()]] = True

        active_player = row["player_name"]
        active_player_idx = self.player_to_idx.get(active_player, self.base_elo_idx)
        opponent_idx = self.base_elo_idx
        move_label = self.all_moves_dict[move_uci]

        return board_tensor, active_player_idx, opponent_idx, move_label, legal_mask
