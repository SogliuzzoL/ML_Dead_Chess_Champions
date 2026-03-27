"""Training utilities for per-player style embeddings using the Maia backbone.

This module provides a dataset wrapper and a training routine intended to learn
per-player style embeddings that complement Maia's canonical Elo-category
embeddings. The learned per-player embeddings are initialised from Maia's most
representative Elo vector and are trained while preserving the original Maia
Elo embeddings as fixed (non-trainable) parameters.

The principal entry point is `run_training(config)`, which constructs a
`PlayerDataset`, configures the Maia model for per-player embedding training,
and persists the trained embeddings to disk.
"""

import chess
import polars as pl
import torch
import torch.nn as nn
from maia2 import model
from maia2.utils import (
    board_to_tensor,
    create_elo_dict,
    get_all_possible_moves,
    map_to_category,
    mirror_move,
)
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.core.config import Config
from src.core.utils import getLogger
from src.models.player_style import PlayerStyleEmbedding

logger = getLogger()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PlayerDataset(Dataset):
    """Dataset providing board tensors and style indices for per-player training.

    Each item yielded by this dataset is a tuple
    (board_tensor, active_player_idx, opponent_idx, move_label) suitable for
    training Maia's policy head. Player indices for project-specific players
    are offset so they do not collide with Maia's internal Elo-category indices.

    Parameters
    ----------
    data_path : str
        Path to a Parquet file containing the move-level dataset.
    player_dict : dict
        Mapping from remote player identifiers to human-readable player names.
    all_moves_dict : dict
        Mapping from UCI move strings to integer labels used by Maia.
    """

    def __init__(self, data_path: str, player_dict: dict, all_moves_dict: dict):
        self.df = pl.read_parquet(data_path)
        self.player_dict = player_dict
        self.all_moves_dict = all_moves_dict
        self.elo_dict = create_elo_dict()
        self.max_maia_idx = max(self.elo_dict.values())

        self.player_to_idx = {
            player: idx + self.max_maia_idx + 1
            for idx, player in enumerate(player_dict.values())
        }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        """Return a single training example."""
        row = self.df.row(idx, named=True)
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


def run_training(config: Config) -> None:
    """Train project-specific per-player embeddings using a frozen Maia backbone.

    The routine loads the pretrained Maia backbone, attaches a `PlayerStyleEmbedding`
    instance (initialised from Maia's highest-index Elo vector), and trains only
    the per-player embedding matrix. Trained parameters are persisted to the path
    defined in the configuration.

    Parameters
    ----------
    config : Config
        Application configuration providing training hyperparameters and paths.
    """
    epochs = config.player_training.epochs
    batch_size = config.player_training.batch_size
    lr = config.player_training.learning_rate

    maia_model = model.from_pretrained("rapid", DEVICE)
    n_players = len(config.data.players)
    maia_model.elo_embedding = PlayerStyleEmbedding(
        maia_model.elo_embedding, n_players
    ).to(DEVICE)

    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}

    dataset = PlayerDataset(
        config.paths.train_set_path, config.data.players, all_moves_dict
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    maia_model.requires_grad_(False)
    maia_model.elo_embedding.players_embeddings.weight.requires_grad = True

    optimizer = Adam(maia_model.elo_embedding.players_embeddings.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    logger.info(
        f"Commencing per-player embedding training for {epochs} epochs, batch_size={batch_size}, lr={lr}"
    )
    pbar_epochs = tqdm(range(epochs), desc="Epochs", unit="epoch")

    for epoch in pbar_epochs:
        maia_model.train()
        epoch_loss = 0.0
        pbar_batches = tqdm(
            loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False, unit="batch"
        )

        for boards, active_ids, opponent_ids, labels in pbar_batches:
            boards, active_ids, opponent_ids, labels = (
                boards.to(DEVICE),
                active_ids.to(DEVICE),
                opponent_ids.to(DEVICE),
                labels.to(DEVICE),
            )

            logits_maia, _, _ = maia_model(boards, active_ids, opponent_ids)
            loss = criterion(logits_maia, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            epoch_loss += current_loss
            pbar_batches.set_postfix({"batch_loss": f"{current_loss:.4f}"})

        avg_loss = epoch_loss / len(loader)
        pbar_epochs.set_postfix({"avg_loss": f"{avg_loss:.4f}"})
        logger.info(
            f"Completed epoch {epoch + 1}/{epochs} | Average loss: {avg_loss:.4f}"
        )

    torch.save(
        maia_model.elo_embedding.players_embeddings.state_dict(),
        config.paths.champions_embeddings_path,
    )
    logger.info(
        f"Per-player embedding model saved to {config.paths.champions_embeddings_path}"
    )
