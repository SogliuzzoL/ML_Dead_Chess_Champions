"""Conditional VAE-GAN implementation for player-conditioned move generation.

This module provides a compact, modular implementation of a conditional
Variational Autoencoder combined with a GAN discriminator to model player
styles. The architecture mirrors the project's autoencoder convention and is
intended to be integrated into the existing training pipeline.

Primary classes
- PlayerEmbedding: learnable embedding table for players
- EncoderVAE: encoder producing (mu, logvar) for the latent Gaussian
- GeneratorDecoder: decoder mapping (z, board) -> move distribution
- Discriminator: binary classifier for (board, move, player)
- ChessVAEGAN: top-level container and loss skeleton for coordinated training

Notes
- The generator returns log-probabilities by default (compatible with NLLLoss).
- The discriminator expects dense move vectors of size `vocab_size`.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_EMBEDDING_DIM = 128
DEFAULT_LATENT_DIM = 128
DEFAULT_VOCAB_SIZE = 4672


class PlayerEmbedding(nn.Module):
    """Learnable player embedding table.

    This module is a thin wrapper around ``nn.Embedding`` that produces dense
    vectors for player identifiers. The table is trainable and is intended to
    be learned jointly with the VAE-GAN.
    """

    def __init__(self, num_players: int, embedding_dim: int = DEFAULT_EMBEDDING_DIM):
        super().__init__()
        self.embedding = nn.Embedding(num_players, embedding_dim)

    def forward(self, player_ids: torch.LongTensor) -> torch.FloatTensor:
        """Return embeddings for `player_ids` (shape: (batch,))."""
        return self.embedding(player_ids.view(-1))


class EncoderVAE(nn.Module):
    """VAE encoder producing Gaussian parameters (mu, logvar)."""

    def __init__(
        self,
        board_dim: int,
        player_embedding_dim: int,
        hidden_dims: Tuple[int, ...] = (1024, 512, 256),
        latent_dim: int = DEFAULT_LATENT_DIM,
    ) -> None:
        super().__init__()
        input_dim = board_dim + player_embedding_dim

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.mlp = nn.Sequential(*layers)

        self.mu_layer = nn.Linear(prev, latent_dim)
        self.logvar_layer = nn.Linear(prev, latent_dim)

    def forward(
        self, board: torch.Tensor, player_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([board, player_emb], dim=1)
        h = self.mlp(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class GeneratorDecoder(nn.Module):
    """Generator / Decoder mapping (z, board) -> move log-probabilities.

    Uses BatchNorm and ReLU activations for stability.
    """

    def __init__(
        self,
        board_dim: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_dims: Tuple[int, ...] = (256, 512, 1024),
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        output_log_softmax: bool = True,
    ) -> None:
        super().__init__()
        input_dim = latent_dim + board_dim
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            prev = h
        self.mlp = nn.Sequential(*layers)

        self.output_layer = nn.Linear(prev, vocab_size)
        self.output_log_softmax = output_log_softmax

    def forward(self, z: torch.Tensor, board: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, board], dim=1)
        h = self.mlp(x)
        logits = self.output_layer(h)
        if self.output_log_softmax:
            return F.log_softmax(logits, dim=1)
        return F.softmax(logits, dim=1)


class Discriminator(nn.Module):
    """Discriminator receiving (board, move_vec, player_emb) and returning prob.

    The move vector may be a one-hot (real) or a probability vector (generated).
    """

    def __init__(
        self,
        board_dim: int,
        player_embedding_dim: int,
        move_dim: int = DEFAULT_VOCAB_SIZE,
        hidden_dims: Tuple[int, ...] = (1024, 512, 256),
    ) -> None:
        super().__init__()
        input_dim = board_dim + move_dim + player_embedding_dim

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.3))
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(prev, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, board: torch.Tensor, move_vec: torch.Tensor, player_emb: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([board, move_vec, player_emb], dim=1)
        h = self.mlp(x)
        logits = self.out(h)
        return self.sigmoid(logits)


class ChessVAEGAN(nn.Module):
    """Top-level VAE-GAN that assembles modules and provides loss skeleton."""

    def __init__(
        self,
        num_players: int,
        board_dim: int,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        latent_dim: int = DEFAULT_LATENT_DIM,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")

        self.player_emb = PlayerEmbedding(num_players, embedding_dim)
        self.encoder = EncoderVAE(
            board_dim=board_dim,
            player_embedding_dim=embedding_dim,
            latent_dim=latent_dim,
        )
        self.generator = GeneratorDecoder(
            board_dim=board_dim,
            latent_dim=latent_dim,
            vocab_size=vocab_size,
            output_log_softmax=True,
        )
        self.discriminator = Discriminator(
            board_dim=board_dim, player_embedding_dim=embedding_dim, move_dim=vocab_size
        )

        self.recon_loss_fn = nn.NLLLoss(reduction="mean")
        self.bce_loss_fn = nn.BCELoss(reduction="mean")

        self.vocab_size = vocab_size
        self.board_dim = board_dim
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim

    def forward(
        self, board: torch.Tensor, player_ids: torch.LongTensor, sample: bool = True
    ) -> dict:
        player_embeddings = self.player_emb(player_ids.to(board.device))
        mu, logvar = self.encoder(board, player_embeddings)
        z = self.encoder.reparameterize(mu, logvar) if sample else mu
        gen_log_probs = self.generator(z, board)

        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "gen_log_probs": gen_log_probs,
            "player_embeddings": player_embeddings,
        }

    def loss_function(
        self,
        outputs: dict,
        real_move_indices: torch.LongTensor,
        board: torch.Tensor,
        player_ids: torch.LongTensor,
        beta_kl: float = 1.0,
        gan_weight: float = 1.0,
        recon_weight: float = 1.0,
    ) -> dict:
        """Compute losses for VAE-GAN training step (skeleton).

        Returns a dictionary with 'recon_loss', 'kl_loss', 'd_loss', 'g_adv_loss'
        and 'total_loss'. The discriminator and generator/encoder should be
        optimized in separate optimizer steps in the training loop.
        """
        mu = outputs["mu"]
        logvar = outputs["logvar"]
        gen_log_probs = outputs["gen_log_probs"]
        player_embeddings = outputs["player_embeddings"]

        device = board.device

        # Reconstruction (NLL expects log-probs)
        recon_loss = self.recon_loss_fn(gen_log_probs, real_move_indices.to(device))

        # KL divergence
        kl_per_example = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = torch.mean(kl_per_example)

        # GAN objective: build real and fake move representations
        real_moves_onehot = F.one_hot(
            real_move_indices.to(device), num_classes=self.vocab_size
        ).float()
        fake_moves_soft = gen_log_probs.exp().detach()

        # Detach player embeddings when computing discriminator loss so gradients
        # do not flow into player embedding table during D updates.
        player_embeddings_det = player_embeddings.detach()

        d_real = self.discriminator(board, real_moves_onehot, player_embeddings_det)
        d_fake = self.discriminator(board, fake_moves_soft, player_embeddings_det)

        real_labels = torch.ones_like(d_real, device=device)
        fake_labels = torch.zeros_like(d_fake, device=device)

        d_loss_real = self.bce_loss_fn(d_real, real_labels)
        d_loss_fake = self.bce_loss_fn(d_fake, fake_labels)
        d_loss = d_loss_real + d_loss_fake

        # Generator adversarial loss — when updating generator/encoder we want D(fake)=1
        # Use non-detached player_embeddings here so gradients propagate into player embeddings
        g_adv_loss = self.bce_loss_fn(
            self.discriminator(board, fake_moves_soft, player_embeddings), real_labels
        )

        total_loss = (
            recon_weight * recon_loss + beta_kl * kl_loss + gan_weight * g_adv_loss
        )

        return {
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "d_loss": d_loss,
            "g_adv_loss": g_adv_loss,
            "total_loss": total_loss,
        }
