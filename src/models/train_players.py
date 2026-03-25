import logging

import torch
import torch.nn as nn
from maia2 import model
from maia2.utils import get_all_possible_moves
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import ProjectConfig
from data.player_dataset import PlayerDataset

from .player_style import PlayerStyleEmbedding

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


def run_training(config: ProjectConfig, epochs=30, batch_size=2048, lr=1e-4):
    maia_model = model.from_pretrained("rapid", DEVICE)
    n_players = len(config.base_player_dict)
    maia_model.elo_embedding = PlayerStyleEmbedding(
        maia_model.elo_embedding, n_players
    ).to(DEVICE)

    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}

    # Utilizing the extracted data module
    dataset = PlayerDataset(config, all_moves_dict)
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
        config.champions_embeddings_path,
    )
    logger.info(f"Model saved to {config.champions_embeddings_path}")
