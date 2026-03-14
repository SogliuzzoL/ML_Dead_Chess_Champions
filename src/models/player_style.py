import torch
import torch.nn as nn


class PlayerStyleEmbedding(nn.Embedding):
    def __init__(self, elo_embeddings: nn.Embedding, n_players: int):
        total_embeddings = elo_embeddings.num_embeddings + n_players
        super().__init__(
            num_embeddings=total_embeddings, embedding_dim=elo_embeddings.embedding_dim
        )
        self.weight = nn.Parameter(torch.empty(0))

        self.elo_embeddings = elo_embeddings
        self.elo_embeddings.requires_grad_(False)

        self.max_maia_idx = elo_embeddings.num_embeddings - 1
        self.dim = elo_embeddings.embedding_dim

        self.players_embeddings = nn.Embedding(n_players, self.dim)

        with torch.no_grad():
            best_weights = (
                self.elo_embeddings.weight[self.max_maia_idx].detach().clone()
            )
            self.players_embeddings.weight.data = best_weights.repeat(n_players, 1)

    def forward(self, input: torch.Tensor):
        is_player = input > self.max_maia_idx
        out = torch.zeros(*input.shape, self.dim, device=input.device)

        if (~is_player).any():
            out[~is_player] = self.elo_embeddings(input[~is_player])

        if is_player.any():
            shifted_indices = input[is_player] - (self.max_maia_idx + 1)
            out[is_player] = self.players_embeddings(shifted_indices)

        return out
