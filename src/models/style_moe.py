"""Lightweight Style adapter: VAE encoder + Router (MoE) + low-rank LoRA experts.

This module implements a compact, easy-to-run prototype that sits on top of a
frozen Maia backbone. The adapter reads a sequence of past board tensors (K
previous positions), encodes them into a continuous latent z (VAE), routes z to
an ensemble of small experts (LoRA-style low-rank adaptors) and produces a
logits delta that is added to Maia's raw logits.

Design notes
- The adapter deliberately operates on Maia's returned hidden vector `v` and
  on Maia logits. This avoids patching Maia internals: we compute
  delta_logits = Sum_i g_i * Expert_i(v) and add it to Maia logits.
- Experts are low-rank (Down/Up) linear adapters: v -> (r) -> num_moves.
- The VAE encodes a flattened sequence of board tensors (seq_len * board_dim).

This is a prototype to validate the pipeline; it favors clarity over maximum
memory/perf optimizations.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqVAE(nn.Module):
    """Simple feed-forward VAE over flattened sequences of board tensors.

    Input shape: (B, seq_len * board_dim)
    Outputs: mu (B, latent_dim), logvar (B, latent_dim)
    """

    def __init__(
        self,
        board_dim: int,
        seq_len: int = 15,
        latent_dim: int = 64,
        hidden_dims: tuple = (1024, 512),
    ) -> None:
        super().__init__()
        self.board_dim = board_dim
        self.seq_len = seq_len
        self.input_dim = board_dim * seq_len
        self.latent_dim = latent_dim

        layers = []
        prev = self.input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.mu_layer = nn.Linear(prev, latent_dim)
        self.logvar_layer = nn.Linear(prev, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, logvar) for input x.

        Accepts inputs of shape (B, F) or (B, C, H, W, ...) and flattens
        non-batch dimensions automatically so callers may pass 2D or high-D
        tensors (the dataset may produce unflattened channel/H/W tensors).
        """
        # Ensure the input is flattened to shape [B, input_dim]
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        h = self.mlp(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class Router(nn.Module):
    """Small MLP router that maps latent z -> gating weights over experts.

    Supports optional top-k sparsification in the forward pass.
    """

    def __init__(self, z_dim: int, n_experts: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_experts)
        )

    def forward(
        self, z: torch.Tensor, temp: float = 1.0, topk: Optional[int] = None
    ) -> torch.Tensor:
        logits = self.net(z) / max(temp, 1e-8)
        weights = F.softmax(logits, dim=-1)
        if topk is not None and topk < weights.size(-1):
            vals, idx = weights.topk(topk, dim=-1)
            mask = torch.zeros_like(weights).scatter_(1, idx, vals)
            # Renormalize to keep a proper distribution
            weights = mask / (mask.sum(dim=1, keepdim=True) + 1e-9)
        return weights


class LoRAExpert(nn.Module):
    """Low-rank adaptor mapping Maia hidden vector v -> logits delta.

    E(v) = Up(Down(v)) * (alpha / rank)
    - v_dim : dimensionality of Maia hidden vector
    - out_dim: number of logits (Maia moves vocabulary size)
    """

    def __init__(
        self, v_dim: int, out_dim: int, rank: int = 8, alpha: float = 1.0
    ) -> None:
        super().__init__()
        self.down = nn.Linear(v_dim, rank, bias=False)
        self.up = nn.Linear(rank, out_dim, bias=False)
        self.scale = alpha / max(1.0, rank)

        # Initialize LoRA with small values for stability
        nn.init.zeros_(self.up.weight)
        nn.init.normal_(self.down.weight, std=0.02)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # v: [B, v_dim] -> returns [B, out_dim]
        return self.up(self.down(v)) * self.scale


class StyleMoE(nn.Module):
    """Top-level adapter containing a SeqVAE, Router, and N LoRA experts.

    The router can be conditioned on a per-player embedding by setting
    `n_players` and `player_emb_dim`. The router MLP input then becomes
    (latent_dim + player_emb_dim).

    Usage:
        adapter = StyleMoE(v_dim, out_dim, board_dim=773, seq_len=15, ...)
        delta_logits, kl, g = adapter(v, seq_inputs, player_ids)
        final_logits = logits_maia + delta_logits
    """

    def __init__(
        self,
        v_dim: int,
        out_dim: int,
        board_dim: int = 773,
        seq_len: int = 15,
        latent_dim: int = 64,
        n_experts: int = 8,
        lora_rank: int = 8,
        router_hidden: int = 128,
        n_players: int = 0,
        player_emb_dim: int = 32,
    ) -> None:
        super().__init__()
        self.v_dim = v_dim
        self.out_dim = out_dim
        self.board_dim = board_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.n_experts = n_experts

        # Optional per-player embedding used to condition the router
        self.n_players = max(0, int(n_players))
        self.player_emb_dim = int(player_emb_dim) if self.n_players > 0 else 0
        if self.n_players > 0:
            self.player_emb = nn.Embedding(self.n_players, self.player_emb_dim)
        else:
            self.player_emb = None

        # Router input dim = latent + player_emb (if present)
        router_input_dim = self.latent_dim + (
            self.player_emb_dim if self.player_emb is not None else 0
        )
        self.vae = SeqVAE(board_dim=board_dim, seq_len=seq_len, latent_dim=latent_dim)
        self.router = Router(router_input_dim, n_experts, hidden=router_hidden)
        self.experts = nn.ModuleList(
            [
                LoRAExpert(v_dim=v_dim, out_dim=out_dim, rank=lora_rank)
                for _ in range(n_experts)
            ]
        )

    def forward(
        self,
        v: torch.Tensor,
        seq_inputs: torch.Tensor,
        player_ids: Optional[torch.LongTensor] = None,
        temp: float = 1.0,
        topk: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute adapter output.

        Parameters
        - v: Maia hidden vector [B, v_dim] (detached; Maia trunk is frozen)
        - seq_inputs: flattened sequence input [B, seq_len * board_dim]
        - player_ids: optional LongTensor [B] of player indices for conditioning
        - temp: softmax temperature for router
        - topk: if set, sparsify gating to top-k experts

        Returns
        - delta_logits: [B, out_dim]
        - kl_loss_per_batch_mean: scalar tensor (mean KL)
        - g: gating weights [B, n_experts]
        """
        # VAE encode + sample
        mu, logvar = self.vae(seq_inputs)
        z = SeqVAE.reparameterize(mu, logvar)

        # Prepare router input (optionally conditioned on player embedding)
        if player_ids is not None and self.player_emb is not None:
            p_emb = self.player_emb(player_ids)
            router_in = torch.cat([z, p_emb], dim=-1)
        else:
            router_in = z

        # Router -> gating
        g = self.router(router_in, temp=temp, topk=topk)  # [B, M]

        # Experts: compute deltas
        # compute per-expert outputs [B, out_dim] and stack
        expert_outs = [expert(v) for expert in self.experts]  # list of [B, out_dim]
        # Stack -> [B, M, out_dim]
        deltas = torch.stack(expert_outs, dim=1)

        # Weighted sum
        delta_logits = (g.unsqueeze(-1) * deltas).sum(dim=1)

        # KL
        kl_per_example = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_mean = torch.mean(kl_per_example)

        return delta_logits, kl_mean, g

    def count_adapter_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Small convenience factory used by training scripts when shapes must be inferred at runtime
def build_adapter_from_maia_sample(
    maia_model,
    seq_len: int = 15,
    latent_dim: int = 64,
    n_experts: int = 8,
    lora_rank: int = 8,
    n_players: int = 0,
    player_emb_dim: int = 32,
) -> StyleMoE:
    """Run a single forward through Maia to infer v_dim and out_dim and build an adapter.

    The maia_model should accept (boards, active_ids, opponent_ids) and return
    (logits, hidden_v, value) or similar. This helper performs a no-grad dummy
    forward on a small zero batch to infer shapes.
    """
    # Build a tiny dummy batch: Maia utilities expect a board tensor shape [B, C, H, W]
    import chess
    import torch
    from maia2.utils import board_to_tensor

    try:
        # Infer device from the model's parameters (works even if requires_grad is False)
        device = next(maia_model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    # Create a starting board tensor
    b = chess.Board()
    board_tensor = board_to_tensor(b).unsqueeze(0).to(device)

    # Map representative Elo categories using maia inference helpers
    # For a minimal shape inference we can pass the same index for active/opponent
    # Maia's model API normally returns (logits, hidden, value)
    with torch.no_grad():
        logits, hidden_v, _ = maia_model(
            board_tensor, torch.tensor([0]).to(device), torch.tensor([0]).to(device)
        )

    v_dim = hidden_v.size(-1)
    out_dim = logits.size(-1)

    adapter = StyleMoE(
        v_dim=v_dim,
        out_dim=out_dim,
        board_dim=board_to_tensor(b).numel(),
        seq_len=seq_len,
        latent_dim=latent_dim,
        n_experts=n_experts,
        lora_rank=lora_rank,
        n_players=n_players,
        player_emb_dim=player_emb_dim,
    )
    return adapter
