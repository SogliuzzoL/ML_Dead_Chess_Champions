import argparse

import chess
import pandas as pd
import torch
from maia2.inference import inference_batch
from maia2.model import from_pretrained
from maia2.utils import board_to_tensor, get_all_possible_moves, mirror_move
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from core.config import RESULT_FOLDER, TEST_SET_PATH, base_player_dict
from core.engine import MaiaEngine
from models.train_players import run_training
from visualization.visualize_accuracies import visualize_player_accuracies


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", type=int, default=0, help="Set to 1 to enable TRAIN_MODE."
    )
    args = parser.parse_args()

    if bool(args.train):
        run_training()

    engine = MaiaEngine()
    baseline_model = from_pretrained("rapid", device=engine.device)

    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    base_elo_idx = engine._get_style_idx(2500)

    test_dataset = EvaluationDataset(
        TEST_SET_PATH, engine.player_to_idx, all_moves_dict, base_elo_idx
    )
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)

    correct_preds_custom, player_ids = engine.evaluate_batch(test_loader)

    idx_to_player = {idx: player for player, idx in engine.player_to_idx.items()}
    metrics = {
        player: {"n_positions": 0, "custom_correct": 0, "baseline_accuracy": 0.0}
        for player in base_player_dict.values()
    }

    for i in range(len(player_ids)):
        if player_ids[i] in idx_to_player:
            player_name = idx_to_player[player_ids[i]]
            metrics[player_name]["n_positions"] += 1
            if correct_preds_custom[i]:
                metrics[player_name]["custom_correct"] += 1

    df_full = pd.read_parquet(TEST_SET_PATH)
    df_full["active_elo"] = 2500
    df_full["opponent_elo"] = 2500
    maia_col_order = ["fen", "move", "active_elo", "opponent_elo"]

    for player_name in tqdm(base_player_dict.values()):
        player_mask = df_full["player_name"] == player_name
        if not player_mask.any():
            continue

        df_player = df_full.loc[player_mask, maia_col_order].copy()
        _, baseline_acc = inference_batch(
            df_player, baseline_model, batch_size=512, num_workers=4, verbose=False
        )
        metrics[player_name]["baseline_accuracy"] = baseline_acc

    results = []
    for p, stats in metrics.items():
        if stats["n_positions"] > 0:
            custom_acc = stats["custom_correct"] / stats["n_positions"]
            baseline_acc = stats["baseline_accuracy"]
            results.append(
                {
                    "player": p,
                    "n_positions": stats["n_positions"],
                    "baseline_accuracy": baseline_acc,
                    "custom_accuracy": custom_acc,
                    "absolute_improvement": custom_acc - baseline_acc,
                }
            )

    df_results = pd.DataFrame(results).sort_values(
        "absolute_improvement", ascending=False
    )
    print(df_results)

    output_path = RESULT_FOLDER + "/player_accuracies_comparison.parquet"
    df_results.to_parquet(output_path, index=False)

    visualize_player_accuracies()
