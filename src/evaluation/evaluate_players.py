"""Evaluation routines for assessing per-player predictive performance.

This module provides dataset utilities and an evaluation driver that compares
the predictive accuracy of a Maia model conditioned on learned per-player
embeddings against a baseline Maia model. The principal entry point,
`evaluate_players(config, force_train=False)`, optionally triggers training of
per-player embeddings, evaluates the customized model on the test split, and
computes baseline accuracies using the unmodified Maia backbone.

All log messages and documentation are expressed in formal academic English.
"""

from typing import Dict, Tuple

import chess
import polars as pl
import torch
from maia2.inference import inference_batch
from maia2.model import from_pretrained
from maia2.utils import board_to_tensor, get_all_possible_moves, mirror_move
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.core.config import Config
from src.core.utils import getLogger
from src.models.maia import MaiaEngine
from src.training.train_players import run_training

logger = getLogger()


class EvaluationDataset(Dataset):
    """Dataset providing examples for evaluation of move-prediction accuracy.

    Each item is a tuple (board_tensor, active_player_idx, opponent_idx,
    move_label, legal_mask) suitable for model inference and accuracy
    computation. When the active player is Black the board and the move label
    are mirrored to maintain a White-to-move canonical representation.
    """

    def __init__(
        self,
        data_path: str,
        player_to_idx: Dict[str, int],
        all_moves_dict: Dict[str, int],
        base_elo_idx: int,
    ) -> None:
        self.df = pl.read_parquet(data_path)
        self.player_to_idx = player_to_idx
        self.all_moves_dict = all_moves_dict
        self.base_elo_idx = base_elo_idx
        self.num_moves = len(all_moves_dict)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, int, torch.Tensor]:
        row = self.df.row(idx, named=True)
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


def evaluate_players(config: Config, force_train: bool = False) -> None:
    """Evaluate per-player predictive accuracy and compare to the baseline model.

    This function optionally triggers per-player embedding training (if
    `force_train` is True), evaluates the customised Maia model on the test set,
    computes baseline accuracies using the unmodified Maia backbone, aggregates
    per-player metrics and persists the comparison table to disk.
    """
    if force_train:
        logger.info(
            "Forcing (re)training of per-player embeddings prior to evaluation..."
        )
        run_training(config)

    engine = MaiaEngine(config)
    baseline_model = from_pretrained("rapid", device=engine.device)

    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    base_elo_idx = engine._get_style_idx(2500)
    assert base_elo_idx is not None

    test_dataset = EvaluationDataset(
        config.paths.test_set_path, engine.player_to_idx, all_moves_dict, base_elo_idx
    )
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)

    logger.info("Commencing evaluation on the test split...")
    correct_preds_custom, player_ids = engine.evaluate_batch(test_loader)

    idx_to_player = {idx: player for player, idx in engine.player_to_idx.items()}

    # Initialize per-player metrics using the configured player names
    metrics = {
        player: {"n_positions": 0, "custom_correct": 0, "baseline_accuracy": 0.0}
        for player in config.data.players.values()
    }

    for i in range(len(player_ids)):
        if player_ids[i] in idx_to_player:
            player_name = idx_to_player[player_ids[i]]
            metrics[player_name]["n_positions"] += 1
            if correct_preds_custom[i]:
                metrics[player_name]["custom_correct"] += 1

    # Prepare a DataFrame to compute baseline accuracies via Maia's batch inference
    df_full = pl.read_parquet(config.paths.test_set_path).with_columns(
        active_elo=pl.lit(2500), opponent_elo=pl.lit(2500)
    )
    maia_col_order = ["fen", "move", "active_elo", "opponent_elo"]

    logger.info("Computing baseline accuracies using the unmodified Maia backbone...")
    for player_name in tqdm(config.data.players.values(), desc="Baseline evaluation"):
        player_mask = df_full["player_name"] == player_name
        if not player_mask.any():
            continue

        df_player = df_full.filter(player_mask).select(maia_col_order)

        _, baseline_acc = inference_batch(
            df_player.to_pandas(),
            baseline_model,
            batch_size=512,
            num_workers=4,
            verbose=False,
        )
        metrics[player_name]["baseline_accuracy"] = baseline_acc

    # Aggregate results and compute per-player absolute improvement
    results = []
    for p, stats in metrics.items():
        if stats["n_positions"] > 0:
            custom_acc = round(stats["custom_correct"] / stats["n_positions"], 4)
            baseline_acc = round(stats["baseline_accuracy"], 4)

            results.append(
                {
                    "player": p,
                    "n_positions": stats["n_positions"],
                    "baseline_accuracy": baseline_acc,
                    "custom_accuracy": custom_acc,
                    "absolute_improvement": round(custom_acc - baseline_acc, 4),
                }
            )

    df_results = pl.DataFrame(results).sort("absolute_improvement", descending=True)

    output_path = config.paths.player_accuracies_path
    df_results.write_parquet(output_path)
    logger.info(f"Final per-player accuracy comparison saved to {output_path}")

    # Log a concise textual summary for quick inspection
    logger.info("Summary of per-player predictive improvement:")
    logger.info(str(df_results))
