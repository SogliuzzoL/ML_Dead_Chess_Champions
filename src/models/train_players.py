import chess
import pandas as pd
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

from core.config import (
    CHAMPIONS_EMBEDDINGS_PATH,
    TRAIN_SET_PATH,
    base_player_dict,
    logger,
)

from .player_style import PlayerStyleEmbedding

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PlayerDataset(Dataset):
    def __init__(self, data_path: str, player_dict: dict, all_moves_dict: dict):
        self.df = pd.read_parquet(data_path)
        self.player_dict = player_dict
        self.all_moves_dict = all_moves_dict
        self.elo_dict = create_elo_dict()
        self.max_maia_idx = max(self.elo_dict.values())

        self.player_to_idx = {
            player: idx + self.max_maia_idx + 1
            for idx, player in enumerate(player_dict.values())
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


def run_training(epochs=30, batch_size=2048, lr=1e-4):
    maia_model = model.from_pretrained("rapid", DEVICE)
    n_players = len(base_player_dict)
    maia_model.elo_embedding = PlayerStyleEmbedding(
        maia_model.elo_embedding, n_players
    ).to(DEVICE)

    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    dataset = PlayerDataset(TRAIN_SET_PATH, base_player_dict, all_moves_dict)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    maia_model.requires_grad_(False)
    maia_model.elo_embedding.players_embeddings.weight.requires_grad = True

    optimizer = Adam(maia_model.elo_embedding.players_embeddings.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    logger.info(
        f"Starting training for {epochs} epochs with batch size {batch_size} and learning rate {lr}"
    )

    pbar_epochs = tqdm(range(epochs), desc="Total Epochs", unit="epoch")

    for epoch in pbar_epochs:
        maia_model.train()
        epoch_loss = 0
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
        logger.info(f"Final Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

    torch.save(
        maia_model.elo_embedding.players_embeddings.state_dict(),
        CHAMPIONS_EMBEDDINGS_PATH,
    )
    logger.info(f"Model saved to {CHAMPIONS_EMBEDDINGS_PATH}")
