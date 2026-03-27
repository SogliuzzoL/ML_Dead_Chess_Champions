"""Per-player style embedding helper for integration with the Maia backbone.

This module provides `PlayerStyleEmbedding`, a thin wrapper that composes an
existing Elo-based embedding table (provided by the Maia model) with a small,
learnable per-player embedding matrix. The combined embedding space enables the
Maia model to represent both canonical Elo categories and repository-specific
player identities within a single embedding tensor.

Design notes
- The class concatenates Maia's pre-existing Elo embedding indices with
  additional indices reserved for project-specific players. Indices greater
  than `max_maia_idx` are interpreted as project-specific players and are
  looked up in the `players_embeddings` matrix.
- Maia's original Elo embeddings are held fixed (non-trainable) while the
  per-player embeddings are trainable parameters.
"""

from typing import Any

import torch
import torch.nn as nn


class PlayerStyleEmbedding(nn.Embedding):
    """Compose Maia's Elo embeddings with trainable per-player embeddings.

    Parameters
    ----------
    elo_embeddings : nn.Embedding
        Pre-existing embedding module from the Maia backbone (indexed by Elo
        category).
    n_players : int
        Number of project-specific players to allocate additional embeddings for.

    Behavior
    --------
    The module constructs an embedding space of size (num_maia_embeddings +
    n_players). When called with an input tensor of indices, indices <=
    `max_maia_idx` are mapped to the fixed Maia Elo embeddings, whereas indices
    > `max_maia_idx` are mapped to the corresponding learnable per-player
    embeddings.
    """

    def __init__(self, elo_embeddings: nn.Embedding, n_players: int) -> None:
        total_embeddings = elo_embeddings.num_embeddings + n_players
        super().__init__(
            num_embeddings=total_embeddings, embedding_dim=elo_embeddings.embedding_dim
        )

        # Maintain compatibility with modules that introspect `weight`.
        # The module composes separate embedding tables; the `weight` attribute
        # is set to an empty parameter to avoid accidental reuse of the parent
        # `Embedding` storage.
        self.weight = nn.Parameter(torch.empty(0))

        # Reference to Maia's Elo embeddings; keep them frozen.
        self.elo_embeddings: nn.Embedding = elo_embeddings
        self.elo_embeddings.requires_grad_(False)

        # Index of the last Maia Elo embedding (used to distinguish indices).
        self.max_maia_idx: int = elo_embeddings.num_embeddings - 1
        self.dim: int = elo_embeddings.embedding_dim

        # Learnable embeddings for project-specific players.
        self.players_embeddings = nn.Embedding(n_players, self.dim)

        # Initialize per-player embeddings from Maia's most-representative vector
        # (here: the embedding at index `max_maia_idx`) to provide a sensible
        # starting point for fine-tuning.
        with torch.no_grad():
            best_weights: Any = (
                self.elo_embeddings.weight[self.max_maia_idx].detach().clone()
            )
            self.players_embeddings.weight.data = best_weights.repeat(n_players, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return embeddings for the provided index tensor.

        Parameters
        ----------
        input : torch.Tensor
            Tensor of integer indices. Indices <= `max_maia_idx` are interpreted
            as Maia Elo categories; indices > `max_maia_idx` are interpreted as
            project-specific player indices (offset by `max_maia_idx + 1`).

        Returns
        -------
        torch.Tensor
            A tensor of shape (*input.shape, embedding_dim) containing the
            corresponding embedding vectors.
        """
        # Boolean mask identifying indices that correspond to project players.
        is_player = input > self.max_maia_idx

        # Allocate output tensor with the correct shape and device.
        out = torch.zeros(*input.shape, self.dim, device=input.device)

        # Populate entries that refer to Maia's Elo categories (non-player indices).
        if (~is_player).any():
            out[~is_player] = self.elo_embeddings(input[~is_player])

        # Populate entries that correspond to project-specific players.
        if is_player.any():
            shifted_indices = input[is_player] - (self.max_maia_idx + 1)
            out[is_player] = self.players_embeddings(shifted_indices)

        return out
