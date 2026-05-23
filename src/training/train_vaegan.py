"""Training script for the conditional VAE-GAN (player-conditioned).

This module provides `run_vaegan_pipeline(config)` which constructs the
dataset / dataloaders, instantiates the `ChessVAEGAN` model, and runs a
training loop that alternates discriminator and generator/encoder updates.

The implementation includes configurable stability options:
- separate learning rates for encoder/generator and discriminator
- multiple discriminator steps per generator step
- KL annealing (beta warm-up)
- label smoothing

Save checkpoints and a small training history for inspection.
"""

from typing import Dict

import chess
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from maia2.utils import board_to_tensor, get_all_possible_moves, mirror_move
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.core.config import Config
from src.core.utils import getLogger
from src.models.chess_vaegan import ChessVAEGAN

logger = getLogger()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VaeganDataset(Dataset):
    """Dataset for VAE-GAN training. Returns (board_tensor, player_idx, move_idx).

    The dataset returns flattened board vectors (1D) to match the MLP encoder
    used by the VAE-GAN implementation.
    """

    def __init__(
        self,
        data_path: str,
        player_dict: Dict[str, str],
        all_moves_dict: Dict[str, int],
    ):
        self.df = pl.read_parquet(data_path)
        self.player_dict = player_dict
        self.all_moves_dict = all_moves_dict

        # map player name -> compact index 0..n_players-1
        self.player_to_idx = {
            player: i for i, player in enumerate(player_dict.values())
        }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.row(idx, named=True)
        board = chess.Board(row["fen"]) if "fen" in row else None

        move_uci = row["move"]
        if row.get("player_color", "white") == "black":
            board = board.mirror()
            move_uci = mirror_move(move_uci)

        board_tensor = board_to_tensor(board)
        # Flatten board tensor to a 1D vector so it matches model expectations
        if hasattr(board_tensor, "view"):
            board_tensor = board_tensor.view(-1).float()

        player_name = row["player_name"]
        player_idx = self.player_to_idx.get(player_name, 0)

        move_label = self.all_moves_dict[move_uci]
        return board_tensor, player_idx, move_label


def run_vaegan_pipeline(config: Config) -> None:
    """Run the full training pipeline for the conditional VAE-GAN.

    The function reads hyperparameters from `config.vaegan` and `config.player_training`.
    """
    # Hyperparameters
    epochs = int(config.player_training.epochs)
    batch_size = int(config.player_training.batch_size)

    # Read vaegan-specific hyperparameters (with sensible fallbacks)
    lr_ge = float(getattr(config.vaegan, "lr_ge", config.player_training.learning_rate))
    lr_d = float(getattr(config.vaegan, "lr_d", config.player_training.learning_rate))
    d_steps_per_g = int(getattr(config.vaegan, "d_steps_per_g", 1))
    kl_warmup_epochs = int(getattr(config.vaegan, "kl_warmup_epochs", 10))
    label_smoothing = float(getattr(config.vaegan, "label_smoothing", 0.0))
    recon_weight = float(getattr(config.vaegan, "recon_weight", 1.0))
    target_beta = float(getattr(config.vaegan, "beta_kl", 1.0))
    gan_weight = float(getattr(config.vaegan, "gan_weight", 1.0))

    # Moves vocabulary
    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}

    logger.info("Loading datasets for VAE-GAN training...")
    train_dataset = VaeganDataset(
        config.paths.train_set_path, config.data.players, all_moves_dict
    )
    test_dataset = VaeganDataset(
        config.paths.test_set_path, config.data.players, all_moves_dict
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=4
    )

    # Infer board_dim from a sample
    sample_board, _, _ = train_dataset[0]
    board_dim = (
        int(sample_board.size(0))
        if hasattr(sample_board, "size")
        else int(config.vaegan.board_dim)
    )

    num_players = len(config.data.players)
    vocab_size = int(config.vaegan.vocab_size)

    # Instantiate model
    model = ChessVAEGAN(
        num_players=num_players,
        board_dim=board_dim,
        vocab_size=vocab_size,
        embedding_dim=int(config.vaegan.embedding_dim),
        latent_dim=int(config.vaegan.latent_dim),
    ).to(DEVICE)

    # Optimizers
    opt_D = Adam(model.discriminator.parameters(), lr=lr_d)
    opt_GE = Adam(
        list(model.encoder.parameters())
        + list(model.generator.parameters())
        + list(model.player_emb.parameters()),
        lr=lr_ge,
    )

    # Training history
    history = {
        "epoch": [],
        "train_loss": [],
        "train_recon": [],
        "train_kl": [],
        "train_d_loss": [],
        "train_g_adv": [],
        "val_acc": [],
    }

    # Optionally enable anomaly detection from config (off by default)
    if getattr(config, "debug_detect_anomaly", False):
        torch.autograd.set_detect_anomaly(True)

    for epoch in range(epochs):
        model.train()
        running = {"total": 0.0, "recon": 0.0, "kl": 0.0, "d": 0.0, "g_adv": 0.0}

        # KL annealing (Delayed linear warmup)
        delay_epochs = 5
        if epoch < delay_epochs:
            beta = 0.0
        elif kl_warmup_epochs > 0:
            beta = target_beta * min(
                1.0, float(epoch + 1 - delay_epochs) / float(kl_warmup_epochs)
            )
        else:
            beta = target_beta

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for boards, player_idxs, labels in pbar:
            boards = boards.to(DEVICE)
            player_idxs = player_idxs.to(DEVICE).long()
            labels = labels.to(DEVICE).long()

            # --- Discriminator updates ---
            d_loss_val = 0.0
            for _ in range(d_steps_per_g):
                outputs_d = model(boards, player_idxs, sample=True)
                gen_log_probs_d = outputs_d["gen_log_probs"]
                player_emb_d = outputs_d["player_embeddings"].detach()

                real_moves = F.one_hot(labels, num_classes=vocab_size).float()
                fake_moves = F.gumbel_softmax(
                    gen_log_probs_d, tau=1.0, hard=True
                ).detach()

                d_real = model.discriminator(boards, real_moves, player_emb_d)
                d_fake = model.discriminator(boards, fake_moves, player_emb_d)

                real_label = 1.0 - label_smoothing
                fake_label = label_smoothing
                real_labels = torch.full_like(d_real, real_label, device=boards.device)
                fake_labels = torch.full_like(d_fake, fake_label, device=boards.device)

                d_loss_real = nn.BCELoss(reduction="mean")(d_real, real_labels)
                d_loss_fake = nn.BCELoss(reduction="mean")(d_fake, fake_labels)
                d_loss = d_loss_real + d_loss_fake

                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()

                d_loss_val = d_loss.item()

            # --- Generator + Encoder updates ---
            outputs = model(boards, player_idxs, sample=True)
            mu = outputs["mu"]
            logvar = outputs["logvar"]
            gen_log_probs = outputs["gen_log_probs"]
            player_emb = outputs["player_embeddings"]

            # Reconstruction
            recon_loss = nn.NLLLoss(reduction="mean")(gen_log_probs, labels)

            # KL
            kl_per_example = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp(), dim=1
            )
            kl_loss = torch.mean(kl_per_example)

            # Generator adversarial objective (encourage D(fake) -> real)
            fake_moves_for_g = F.gumbel_softmax(gen_log_probs, tau=1.0, hard=True)
            d_on_fake = model.discriminator(boards, fake_moves_for_g, player_emb)
            real_labels_for_g = torch.full_like(
                d_on_fake, 1.0 - label_smoothing, device=boards.device
            )
            g_adv_loss = nn.BCELoss(reduction="mean")(d_on_fake, real_labels_for_g)

            total_loss = (
                recon_weight * recon_loss + beta * kl_loss + gan_weight * g_adv_loss
            )

            opt_GE.zero_grad()
            total_loss.backward()
            opt_GE.step()

            # accumulate
            running["total"] += total_loss.item()
            running["recon"] += recon_loss.item()
            running["kl"] += kl_loss.item()
            running["d"] += d_loss_val
            running["g_adv"] += g_adv_loss.item()

            mu_mean = (
                float(mu.mean().item()) if torch.isfinite(mu).all() else float("nan")
            )
            mu_std = (
                float(mu.std().item()) if torch.isfinite(mu).all() else float("nan")
            )

            pbar.set_postfix(
                {
                    "tot_loss": f"{running['total'] / (len(pbar) + 1):.4f}",
                    "recon": f"{running['recon'] / (len(pbar) + 1):.4f}",
                    "kl": f"{running['kl'] / (len(pbar) + 1):.6f}",
                    "d_loss": f"{running['d'] / (len(pbar) + 1):.4f}",
                    "mu_mean": f"{mu_mean:.4f}",
                    "mu_std": f"{mu_std:.4f}",
                }
            )

        # Validation quick check
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for boards, player_idxs, labels in test_loader:
                boards = boards.to(DEVICE)
                player_idxs = player_idxs.to(DEVICE).long()
                labels = labels.to(DEVICE).long()

                out = model(boards, player_idxs, sample=False)
                preds = out["gen_log_probs"].argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total if total > 0 else 0.0

        epoch_loss = running["total"] / len(train_loader)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(epoch_loss)
        history["train_recon"].append(running["recon"] / len(train_loader))
        history["train_kl"].append(running["kl"] / len(train_loader))
        history["train_d_loss"].append(running["d"] / len(train_loader))
        history["train_g_adv"].append(running["g_adv"] / len(train_loader))
        history["val_acc"].append(val_acc)

        logger.info(
            f"Epoch {epoch + 1}/{epochs} | loss={epoch_loss:.4f} | val_acc={val_acc:.4f} | beta={beta:.4f}"
        )

    # Save checkpoint
    ckpt_path = "models/saved/vaegan.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_GE_state_dict": opt_GE.state_dict(),
            "optimizer_D_state_dict": opt_D.state_dict(),
            "history": history,
        },
        ckpt_path,
    )

    logger.info("VAE-GAN training completed. Checkpoint saved to %s", ckpt_path)

    # Persist history
    try:
        import pandas as pd

        df = pd.DataFrame(history)
        df.to_parquet(config.paths.learning_curves_path)
        logger.info(
            "Training history persisted to %s", config.paths.learning_curves_path
        )
    except Exception:
        logger.warning(
            "Could not persist history as parquet. Please ensure pandas is installed."
        )
