"""Engine wrapper exposing a Maia trunk augmented with a Style MoE adapter.

This module provides StyleMaiaEngine which loads a frozen Maia backbone and a
trained StyleMoE adapter (if available). The interface mirrors `MaiaEngine` so
it can be used interchangeably by the web UI: `predict_move` and
`predict_mcts` are provided.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional, Tuple

import chess
import polars as pl
import torch
from maia2 import inference, model
from maia2.utils import board_to_tensor, mirror_move

from src.models.mcts import MCTS
from src.models.style_moe import StyleMoE, build_adapter_from_maia_sample

logger = logging.getLogger(__name__)


class StyleMaiaEngine:
    """Maia backbone wrapped with a StyleMoE adapter.

    Parameters
    - config: project config (used for player lists / paths)
    - adapter_path: path to adapter state dict (optional). If not found the
      engine will raise on prediction requests that require the adapter.
    """

    def __init__(
        self,
        config,
        maia_model: Optional[object] = None,
        model_type: str = "rapid",
        adapter_path: Optional[str] = None,
        seq_len: int = 15,
        latent_dim: int = 64,
        n_experts: int = 8,
        lora_rank: int = 8,
    ) -> None:
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load or reuse Maia backbone (frozen)
        if maia_model is not None:
            self.maia = maia_model
            # Ensure model is on the desired device
            try:
                self.maia.to(self.device)
            except Exception:
                pass
        else:
            self.maia = model.from_pretrained(model_type, self.device)
        self.maia.requires_grad_(False)
        self.prepare = inference.prepare()

        # Player mapping for adapter conditioning
        players_list = list(config.data.players.values())
        self.player_name_to_idx = {name: idx for idx, name in enumerate(players_list)}
        # reserve last index for unknown
        self.n_players_adapter = len(players_list) + 1
        player_emb_dim = 32

        # Build adapter skeleton from a sample forward to infer shapes
        self.adapter = build_adapter_from_maia_sample(
            self.maia,
            seq_len=seq_len,
            latent_dim=latent_dim,
            n_experts=n_experts,
            lora_rank=lora_rank,
            n_players=self.n_players_adapter,
            player_emb_dim=player_emb_dim,
        ).to(self.device)

        # Load adapter params if available
        if adapter_path is None:
            adapter_path = Path(config.paths.model) / "saved" / "style_moe.pth"
        else:
            adapter_path = Path(adapter_path)

        if adapter_path.exists():
            try:
                state = torch.load(adapter_path, map_location=self.device)
                self.adapter.load_state_dict(state)
                logger.info(f"Style adapter loaded from {adapter_path}")
            except Exception as e:
                logger.warning(f"Failed to load adapter state from {adapter_path}: {e}")
        else:
            logger.warning(
                f"Adapter not found at {adapter_path}; style adapter inactive."
            )
            self.adapter = None

    def _build_seq_input_from_pgn(
        self, pgn_text: str, fen: str, seq_len: int
    ) -> torch.Tensor:
        """Reconstruct last `seq_len` board tensors from a PGN string.

        Returns a tensor shaped (1, seq_len * board_dim) on the engine device.
        If reconstruction fails, returns a zero tensor.
        """
        board_dim = board_to_tensor(chess.Board()).numel()
        out = torch.zeros((1, seq_len * board_dim), device=self.device)

        if not pgn_text:
            return out

        try:
            game = chess.pgn.read_game(io.StringIO(pgn_text))
        except Exception:
            game = None

        if game is None:
            # fallback: try to parse moves line-by-line
            return out

        board_iter = game.board()
        fen_list = []
        for mv in game.mainline_moves():
            fen_list.append(board_iter.fen())
            board_iter.push(mv)

        # Find index of the current fen in the reconstructed list
        try:
            pos_idx = fen_list.index(fen)
        except ValueError:
            # try loose match on piece placement
            pos_idx = None
            target_pieces = fen.split(" ")[0]
            for i, cand in enumerate(fen_list):
                if cand.split(" ")[0] == target_pieces:
                    pos_idx = i
                    break
            if pos_idx is None:
                return out

        start_idx = max(0, pos_idx - seq_len)
        selected = fen_list[start_idx:pos_idx]

        seq_tensors = []
        for sfen in selected:
            sb = chess.Board(sfen)
            seq_tensors.append(board_to_tensor(sb).flatten())

        pad_needed = seq_len - len(seq_tensors)
        if pad_needed > 0:
            seq_tensors = [
                torch.zeros(board_dim, device=self.device) for _ in range(pad_needed)
            ] + seq_tensors

        seq_flat = torch.cat([t.to(self.device) for t in seq_tensors], dim=0).unsqueeze(
            0
        )
        return seq_flat

    def _mask_and_probs(
        self, logits: torch.Tensor, fen: str
    ) -> Tuple[dict, torch.Tensor]:
        """Mask illegal moves and return move->prob dict and masked logits tensor.

        logits: [1, V]
        Returns: (move_dict, masked_logits)
        """
        all_moves_dict, _, all_moves_dict_reversed = self.prepare
        device = logits.device
        legal_mask = torch.zeros(logits.size(-1), device=device, dtype=torch.bool)

        board = chess.Board(fen)
        # apply mirror if black to move (Maia convention)
        is_mirrored = False
        if board.turn == chess.BLACK:
            board = board.mirror()
            is_mirrored = True

        for m in board.legal_moves:
            idx = all_moves_dict.get(m.uci())
            if idx is not None and 0 <= idx < logits.size(-1):
                legal_mask[idx] = True

        logits = logits.masked_fill(~legal_mask, -1e9)
        probs = logits[0].softmax(dim=-1).cpu().numpy()

        move_dict = {}
        for i in legal_mask.nonzero().flatten().tolist():
            mv = all_moves_dict_reversed[i]
            final_move = mirror_move(mv) if is_mirrored else mv
            move_dict[final_move] = float(probs[i])

        return move_dict, logits

    def predict_move(
        self,
        fen: str,
        pgn: str = "",
        active_elo: int | str = 2500,
        opponent_elo: int | str = 2500,
    ):
        """Predict move probabilities and scalar value for a single position applying the adapter if present."""
        board = chess.Board(fen)
        is_mirrored = False
        if board.turn == chess.BLACK:
            board = board.mirror()
            is_mirrored = True

        board_tensor = board_to_tensor(board).unsqueeze(0).to(self.device)

        _, elo_dict, _ = self.prepare
        a_idx = inference.map_to_category(
            int(active_elo) if isinstance(active_elo, int) else 2500, elo_dict
        )
        o_idx = inference.map_to_category(
            int(opponent_elo) if isinstance(opponent_elo, int) else 2500, elo_dict
        )
        a_t = torch.tensor([a_idx], device=self.device)
        o_t = torch.tensor([o_idx], device=self.device)

        self.maia.eval()
        with torch.no_grad():
            logits_maia, v, value = self.maia(board_tensor, a_t, o_t)

            if self.adapter is None:
                # Fallback to Maia predictions
                move_dict, _ = self._mask_and_probs(logits_maia, fen)
                return None, move_dict, float(value[0].cpu().item())

            # Build seq inputs from provided PGN and current fen
            seq_input = self._build_seq_input_from_pgn(pgn, fen, self.adapter.seq_len)
            if seq_input is None:
                seq_input = torch.zeros(
                    (1, self.adapter.seq_len * self.adapter.board_dim),
                    device=self.device,
                )

            # Determine player index for conditioning adapter (if possible)
            player_idx = None
            try:
                if (
                    isinstance(active_elo, str)
                    and active_elo in self.player_name_to_idx
                ):
                    player_idx = int(self.player_name_to_idx[active_elo])
                else:
                    # not a known champion: use reserved unknown index
                    player_idx = int(self.n_players_adapter - 1)
            except Exception:
                player_idx = int(self.n_players_adapter - 1)

            player_ids = torch.tensor(
                [player_idx], dtype=torch.long, device=self.device
            )

            # Adapter forward
            delta_logits, _, g = self.adapter(v, seq_input, player_ids=player_ids)
            final_logits = logits_maia + delta_logits

            move_dict, masked_logits = self._mask_and_probs(final_logits, fen)

            move_dict = dict(sorted(move_dict.items(), key=lambda x: x[1]))

            return None, move_dict, float(value[0].cpu().item())

    def predict_mcts(
        self,
        fen: str,
        pgn: str = "",
        num_simulations: int = 800,
        c_puct: float = 1.5,
        threshold: float = 0.01,
        active_elo: int | str = 2500,
        opponent_elo: int | str = 2500,
    ):
        """Run lightweight MCTS using this engine's predict_move as child generator."""
        board = chess.Board(fen)
        mcts = MCTS(self.predict_move)
        best_move, result = mcts.run(
            board,
            num_simulations,
            c_puct=c_puct,
            threshold=threshold,
            activ_elo=active_elo,
            opp_elo=opponent_elo,
        )
        return best_move, result
