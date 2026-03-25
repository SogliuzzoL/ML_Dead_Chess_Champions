import chess
import pandas as pd
from maia2.utils import board_to_tensor, create_elo_dict, map_to_category, mirror_move
from torch.utils.data import Dataset

from core.config import ProjectConfig


class PlayerDataset(Dataset):
    def __init__(self, config: ProjectConfig, all_moves_dict: dict):
        self.df = pd.read_parquet(config.train_set_path)
        self.player_dict = config.base_player_dict
        self.all_moves_dict = all_moves_dict
        self.elo_dict = create_elo_dict()
        self.max_maia_idx = max(self.elo_dict.values())

        self.player_to_idx = {
            player: idx + self.max_maia_idx + 1
            for idx, player in enumerate(self.player_dict.values())
        }

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

        active_player = row["player_name"]
        opponent_elo = 2500

        if active_player in self.player_to_idx:
            active_player_idx = self.player_to_idx[active_player]
        else:
            active_player_idx = map_to_category(2500, self.elo_dict)

        opponent_idx = map_to_category(opponent_elo, self.elo_dict)
        move_label = self.all_moves_dict[move_uci]

        return board_tensor, active_player_idx, opponent_idx, move_label
