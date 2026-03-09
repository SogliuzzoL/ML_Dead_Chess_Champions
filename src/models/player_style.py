import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PlayerStyleEmbedding(nn.Module):
    def __init__(self, elo_embeddings: nn.Embedding, n_players: int):
        super().__init__()
        self.elo_embeddings = elo_embeddings
        self.elo_embeddings.requires_grad_(False)

        self.dim = self.elo_embeddings.embedding_dim
        self.max_maia_idx = self.elo_embeddings.num_embeddings - 1
        self.n_players = n_players
        self.players_embeddings = nn.Embedding(self.n_players, self.dim)

        self.best_elo_weights = self.elo_embeddings.weight[self.max_maia_idx].detach(
        ).clone()
        self.players_embeddings.weight.data = self.best_elo_weights.repeat(
            self.n_players, 1)

    def forward(self, indices: torch.Tensor):
        is_player = indices > self.max_maia_idx
        out = torch.zeros(*indices.shape, self.dim, device=DEVICE)

        if (~is_player).any():
            out[~is_player] = self.elo_embeddings(indices[~is_player])

        if is_player.any():
            shifted_indices = indices[is_player] - (self.max_maia_idx + 1)
            out[is_player] = self.players_embeddings(shifted_indices)

        return out
