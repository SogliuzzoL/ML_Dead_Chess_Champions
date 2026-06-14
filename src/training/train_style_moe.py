"""Training script for the Style MoE adapter.

This training routine implements a prototyping workflow:
- Build a dataset that provides the current board tensor plus the last K positions
  (reconstructed from the raw PGN file for the game), flattened for the SeqVAE.
- Load a frozen Maia backbone and run forward to obtain logits + hidden vector v.
- Run the adapter (VAE + Router + LoRA experts) to compute delta_logits.
- Optimize adapter parameters (VAE, Router, Experts) to minimize move-prediction
  cross-entropy augmented with KL and balance regularizers.

Usage: invoked via `main.py train_style_moe` which calls `run_training(config)`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import chess
import chess.pgn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from maia2 import inference, model
from maia2.utils import board_to_tensor, mirror_move
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.core.config import Config
from src.core.utils import getLogger
from src.models.style_moe import StyleMoE

logger = getLogger()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StyleMoEDataset(Dataset):
    """Dataset that provides (board_tensor, lastK_flat, active_idx, opponent_idx, label, fen).

    lastK_flat is the flattened concatenation of the previous `seq_len` board
    tensors (each produced by `board_to_tensor`); if fewer than `seq_len` are
    available the sequence is left-padded with zeros.
    """

    def __init__(
        self,
        data_path: str,
        config: Config,
        seq_len: int = 15,
    ) -> None:
        self.df = pl.read_parquet(data_path)
        self.seq_len = seq_len
        self.config = config
        # invert players mapping (id -> name) -> (name -> id)
        self.name_to_id: Dict[str, str] = {v: k for k, v in config.data.players.items()}

        # Build an adapter-specific player index mapping (0..N-1). Reserve the
        # last index for unknown players not present in the config.
        players_list = list(config.data.players.values())
        self.player_name_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(players_list)
        }
        self.n_players_adapter: int = len(players_list) + 1

        self.raw_root = Path(config.paths.raw_data)

        # Use Maia's canonical move vocabulary (prepare returns (all_moves_dict, elo_dict, all_moves_dict_reversed))
        self.prepare = inference.prepare()
        all_moves_dict, _, _ = self.prepare
        # all_moves_dict maps move_uci -> maia_index; use it as dataset's canonical mapping
        self.move_to_idx = all_moves_dict

        # Cache parsed PGN fen-lists for games: { (player_id, game_id) : [fen0, fen1, ...] }
        self._game_cache: Dict[Tuple[str, str], List[str]] = {}

    def __len__(self) -> int:
        return len(self.df)

    def _load_game_fen_list(self, game_id: str, player_id: str) -> List[str]:
        key = (player_id, game_id)
        if key in self._game_cache:
            return self._game_cache[key]

        # Candidate path: raw_root / player_id / f"{game_id}.pgn"
        candidate = self.raw_root / player_id / f"{game_id}.pgn"
        if not candidate.exists():
            # Fallback: scan the raw_root for the matching stem (slow but robust)
            found = None
            for p in self.raw_root.rglob(f"{game_id}.pgn"):
                found = p
                break
            if found is None:
                raise FileNotFoundError(
                    f"PGN file for game '{game_id}' not found under raw data"
                )
            candidate = found

        # Parse game and record FENs for positions *before* each mainline move
        with open(candidate, "r", encoding="utf-8", errors="ignore") as f:
            try:
                game = chess.pgn.read_game(f)
            except Exception as e:
                logger.warning(f"Failed to parse PGN {candidate}: {e}")
                # Cache empty list to avoid repeated attempts
                self._game_cache[key] = []
                return []

        if game is None:
            logger.warning(
                f"Unable to parse PGN for {candidate}; caching empty fen list."
            )
            self._game_cache[key] = []
            return []

        board = game.board()
        fen_list: List[str] = []
        for mv in game.mainline_moves():
            fen_list.append(board.fen())
            board.push(mv)
        # It's fine if fen_list is shorter than expected; dataset code will pad
        self._game_cache[key] = fen_list
        return fen_list

    def _canonicalize_move(self, board: chess.Board, move_str: str) -> str | None:
        """Try multiple strategies to canonicalize a move string to UCI on the provided board.

        Returns the UCI string if successful, otherwise None.
        """
        s = move_str.strip()
        # Remove common annotations
        s = re.sub(r"[\+#!?]\$?\d*", "", s)
        s = s.replace("=", "").lower()

        # Strategy 1: try UCI directly
        try:
            mv = chess.Move.from_uci(s)
            if mv in board.legal_moves:
                return mv.uci()
        except Exception:
            pass

        # Strategy 2: try SAN parsing
        try:
            mv = board.parse_san(move_str)
            if mv in board.legal_moves:
                return mv.uci()
        except Exception:
            pass

        # Strategy 3: loose matching against legal moves (endswith / contains)
        for lu in board.legal_moves:
            u = lu.uci()
            if u == s or u.endswith(s) or s.endswith(u) or s in u:
                return u

        return None

    def __getitem__(self, idx: int):
        row = self.df.row(idx, named=True)
        fen = row["fen"]
        move_uci = row["move"]
        game_id = row["game_id"]
        player_name = row["player_name"]

        # Board tensor for current position (as Maia expects)
        board = chess.Board(fen)

        # Mirror board & move for Black to follow Maia's internal mirror convention
        if row.get("player_color", "white") == "black":
            board = board.mirror()
            move_uci = mirror_move(move_uci)

        board_tensor = board_to_tensor(board)  # shape: (board_dim,)

        # Normalize move string early to avoid whitespace/newline issues
        move_uci = move_uci.strip().lower()

        # Compute adapter player index (unknown players -> last reserved index)
        if player_name in self.player_name_to_idx:
            player_idx = int(self.player_name_to_idx[player_name])
        else:
            player_idx = int(self.n_players_adapter - 1)

        # Last-K sequence reconstruction
        if player_name not in self.name_to_id:
            # Unknown player mapping -> attempt naive search under raw_root
            player_id = None
            # Try to locate file by searching any directory for game_id.pgn
            p = None
            for cand in self.raw_root.rglob(f"{game_id}.pgn"):
                p = cand
                break
            if p is None:
                raise FileNotFoundError(f"Unable to find PGN for {game_id}")
            # player_id = parent folder name
            player_id = p.parent.name
        else:
            player_id = self.name_to_id[player_name]

        fen_list = self._load_game_fen_list(game_id, player_id)

        # Find position index matching our fen
        try:
            pos_idx = fen_list.index(fen)
        except ValueError:
            # If not found, fall back to using the last `seq_len` positions available
            # (rare: due to slight FEN formatting differences)
            # We'll attempt a relaxed matching by comparing normalized piece placements
            pos_idx = None
            target_pieces = fen.split(" ")[0]
            for i, candidate_fen in enumerate(fen_list):
                if candidate_fen.split(" ")[0] == target_pieces:
                    pos_idx = i
                    break
            if pos_idx is None:
                # Give up and return a zero-padded sequence
                pos_idx = 0

        start_idx = max(0, pos_idx - self.seq_len)
        selected_fens = fen_list[start_idx:pos_idx]

        # Build flattened lastK tensor (pad on the left with zeros if needed)
        seq_tensors = []
        for sfen in selected_fens:
            sb = chess.Board(sfen)
            seq_tensors.append(board_to_tensor(sb))

        # Left-pad if needed
        pad_needed = self.seq_len - len(seq_tensors)
        if pad_needed > 0:
            seq_tensors = [
                torch.zeros_like(board_tensor) for _ in range(pad_needed)
            ] + seq_tensors

        # Now seq_tensors has length == seq_len
        seq_flat = torch.cat(seq_tensors, dim=0).flatten()  # (seq_len * board_dim,)

        # Active/opponent indices for Maia API mapping
        # Map active player to Maia style index: best-effort using the Elo mapping helper
        # We will provide the canonical category index for players not in mapping
        # Use the Maia inference helpers cached on the dataset
        _, elo_dict, _ = self.prepare

        # Map a representative Elo (2500) to the Maia category index. We do not attempt
        # to attach project-specific player indices here; adapter training will be
        # conditioned on the VAE-derived latent instead.
        active_idx = int(inference.map_to_category(2500, elo_dict))
        opponent_idx = int(inference.map_to_category(2500, elo_dict))

        # Canonicalize move and drop sample if we can't map it
        canonical = self._canonicalize_move(board, move_uci)
        if canonical is None:
            # Return None so a collate_fn can filter this sample out
            logger.debug(f"Dropping unmappable move {move_uci} in game {game_id}")
            return None

        move_label = self.move_to_idx.get(canonical, -1)
        if move_label == -1:
            logger.debug(
                f"Canonical move {canonical} not found in mapping for game {game_id}; dropping"
            )
            return None

        return (
            board_tensor,
            seq_flat,
            torch.tensor(player_idx, dtype=torch.long),
            torch.tensor(active_idx, dtype=torch.long),
            torch.tensor(opponent_idx, dtype=torch.long),
            torch.tensor(move_label, dtype=torch.long),
            fen,
        )


def run_training(config: Config) -> None:
    """Entry point called from `main.py`.

    The function trains the adapter and persists the adapter state dict to
    `models/saved/style_moe.pth`.
    """
    # Hyperparameters (sane defaults; feel free to tune)
    seq_len = 15
    latent_dim = 64
    n_experts = 8
    lora_rank = 8
    player_emb_dim = 32
    batch_size = 256
    epochs = 5
    lr = 1e-3
    beta_kl = 1e-2
    gamma_balance = 1.0
    weight_decay = 1e-5
    topk = None  # set to 1 or 2 for sparse routing at some point

    logger.info("Initializing dataset and dataloader...")
    train_dataset = StyleMoEDataset(
        config.paths.train_set_path, config, seq_len=seq_len
    )

    # Number of adapter player slots (dataset created n_players_adapter)
    n_players_adapter = train_dataset.n_players_adapter

    # Collate function that filters out None-returning samples (dropped during canonicalization)
    from torch.utils.data._utils.collate import default_collate

    drop_counter = {"dropped": 0}

    def collate_filter(batch):
        filtered = [b for b in batch if b is not None]
        dropped = len(batch) - len(filtered)
        if dropped > 0:
            drop_counter["dropped"] += dropped
        if not filtered:
            return None
        return default_collate(filtered)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_filter,
    )

    logger.info("Loading frozen Maia backbone...")
    maia_model = model.from_pretrained("rapid", DEVICE)
    maia_model.requires_grad_(False)
    maia_model.eval()

    # Use a single batch to infer shapes for adapter construction
    sample_batch = next(iter(train_loader))
    boards_sample = sample_batch[0].to(DEVICE)
    player_sample = sample_batch[2].to(DEVICE)
    active_sample = sample_batch[3].to(DEVICE)
    opponent_sample = sample_batch[4].to(DEVICE)

    with torch.no_grad():
        logits_sample, v_sample, _ = maia_model(
            boards_sample, active_sample, opponent_sample
        )

    v_dim = int(v_sample.size(-1))
    out_dim = int(logits_sample.size(-1))

    # Flatten single-example board dimension (handles both flattened and CHW inputs)
    board_dim = int(boards_sample.view(boards_sample.size(0), -1).size(1))

    logger.info(f"Inferred v_dim={v_dim}, out_dim={out_dim}, board_dim={board_dim}")

    adapter = StyleMoE(
        v_dim=v_dim,
        out_dim=out_dim,
        board_dim=board_dim,
        seq_len=seq_len,
        latent_dim=latent_dim,
        n_experts=n_experts,
        lora_rank=lora_rank,
        n_players=n_players_adapter,
        player_emb_dim=player_emb_dim,
    ).to(DEVICE)

    logger.info(f"Adapter parameter count: {adapter.count_adapter_params():,}")

    optimizer = Adam(adapter.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Convenience helper to build legal masks for a batch from FENs
    def build_legal_masks(fens: List[str]) -> torch.Tensor:
        masks = torch.zeros((len(fens), out_dim), dtype=torch.bool, device=DEVICE)
        # Reuse dataset's move mapping from Maia's prepare
        move_to_idx_local = train_dataset.move_to_idx
        for i, f in enumerate(fens):
            b = chess.Board(f)
            # Mirror black positions to match Maia's input convention
            if b.turn == chess.BLACK:
                b = b.mirror()
            for m in b.legal_moves:
                idx = move_to_idx_local.get(m.uci())
                if idx is not None:
                    # Defensive: ensure index fits Maia output dimension
                    if 0 <= int(idx) < out_dim:
                        masks[i, int(idx)] = True
                    else:
                        logger.debug(
                            f"Skipping legal move with idx {idx} >= out_dim {out_dim} for fen {f}"
                        )
        return masks

    logger.info("Starting training loop...")
    history = {
        "epoch": [],
        "loss": [],
        "ce": [],
        "kl": [],
        "balance": [],
        "accuracy": [],
        "dropped": [],
    }
    gating_history: List[np.ndarray] = []

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        running_loss = 0.0
        # Epoch accumulators for sample-weighted averages
        epoch_loss_sum = 0.0
        epoch_ce_sum = 0.0
        epoch_kl_sum = 0.0
        epoch_balance_sum = 0.0
        epoch_total = 0
        epoch_correct = 0
        gating_sum = np.zeros(adapter.n_experts, dtype=float)

        for batch in pbar:
            if batch is None:
                # Entire batch was filtered out (all examples unmappable); skip
                continue

            boards, seq_flat, player_ids, active_ids, opponent_ids, labels, fens = batch
            boards = boards.to(DEVICE)
            seq_flat = seq_flat.to(DEVICE)
            player_ids = player_ids.to(DEVICE)
            active_ids = active_ids.to(DEVICE)
            opponent_ids = opponent_ids.to(DEVICE)
            labels = labels.to(DEVICE)

            batch_size = labels.size(0)

            # Obtain Maia logits + hidden v (no grad into Maia)
            with torch.no_grad():
                logits_maia, v_maia, _ = maia_model(boards, active_ids, opponent_ids)

            # Adapter forward (condition on player_ids)
            delta_logits, kl_mean, g = adapter(
                v_maia.detach(), seq_flat, player_ids=player_ids, topk=topk
            )

            final_logits = logits_maia + delta_logits

            # Legal masks
            legal_masks = build_legal_masks(fens)
            final_logits = final_logits.masked_fill(~legal_masks, -1e9)

            ce_loss = criterion(final_logits, labels)

            # Balance loss (encourage router to use experts uniformly)
            g_mean = g.mean(dim=0)
            balance_target = torch.full_like(g_mean, 1.0 / adapter.n_experts)
            balance_loss = F.mse_loss(g_mean, balance_target)

            l2_reg = 0.0
            for p in adapter.parameters():
                l2_reg = l2_reg + (p.norm(2) ** 2)

            loss = (
                ce_loss
                + beta_kl * kl_mean
                + gamma_balance * balance_loss
                + 1e-6 * l2_reg
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics accumulation (sample-weighted)
            epoch_total += int(batch_size)
            epoch_loss_sum += loss.item() * int(batch_size)
            epoch_ce_sum += ce_loss.item() * int(batch_size)
            epoch_kl_sum += kl_mean.item() * int(batch_size)
            epoch_balance_sum += balance_loss.item() * int(batch_size)

            preds = final_logits.argmax(dim=-1)
            epoch_correct += int((preds == labels).sum().item())

            # Gating usage accumulation (sum over all samples in batch)
            gating_sum += g.detach().sum(dim=0).cpu().numpy()

            running_loss += loss.item()
            current_acc = epoch_correct / max(1, epoch_total)
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "ce": f"{ce_loss.item():.4f}",
                    "kl": f"{kl_mean.item():.4f}",
                    "acc": f"{current_acc:.3f}",
                }
            )

        # Compute epoch averages (sample-weighted)
        if epoch_total == 0:
            logger.warning(f"Epoch {epoch + 1}: no samples processed (all dropped)")
            continue

        epoch_avg_loss = epoch_loss_sum / epoch_total
        epoch_avg_ce = epoch_ce_sum / epoch_total
        epoch_avg_kl = epoch_kl_sum / epoch_total
        epoch_avg_balance = epoch_balance_sum / epoch_total
        epoch_acc = epoch_correct / epoch_total
        gating_mean = (gating_sum / epoch_total).tolist()

        history["epoch"].append(epoch + 1)
        history["loss"].append(float(epoch_avg_loss))
        history["ce"].append(float(epoch_avg_ce))
        history["kl"].append(float(epoch_avg_kl))
        history["balance"].append(float(epoch_avg_balance))
        history["accuracy"].append(float(epoch_acc))
        history["dropped"].append(int(drop_counter["dropped"]))
        gating_history.append(gating_mean)

        logger.info(
            f"Epoch {epoch + 1} finished. loss={epoch_avg_loss:.4f} | ce={epoch_avg_ce:.4f} | kl={epoch_avg_kl:.4f} | acc={epoch_acc:.4f} | dropped={drop_counter['dropped']}"
        )

    # Report dropped examples (if any)
    logger.info(
        f"Dropped {drop_counter['dropped']} examples during loading (unmappable moves)"
    )

    # Persist training metrics and gating history
    metrics_dir = Path(config.paths.evaluation_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "style_moe_training.parquet"

    # Save epoch-level metrics
    df_metrics = pl.DataFrame(history)
    df_metrics.write_parquet(str(metrics_path))
    logger.info(f"Training metrics saved to {metrics_path}")

    # Save gating history (epochs x experts)
    if gating_history:
        gating_arr = np.array(gating_history)  # shape: (epochs_done, n_experts)
        cols = {"epoch": list(range(1, gating_arr.shape[0] + 1))}
        for i in range(gating_arr.shape[1]):
            cols[f"expert_{i}"] = gating_arr[:, i].tolist()
        gating_df = pl.DataFrame(cols)
        gating_path = metrics_dir / "style_moe_gating.parquet"
        gating_df.write_parquet(str(gating_path))
        logger.info(f"Gating history saved to {gating_path}")

    # Generate and save plots
    graphics_dir = Path(config.paths.result) / "graphics"
    graphics_dir.mkdir(parents=True, exist_ok=True)

    # Plot training curves
    epochs_arr = history["epoch"]
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_arr, history["loss"], label="total_loss")
    plt.plot(epochs_arr, history["ce"], label="ce_loss")
    plt.plot(epochs_arr, history["kl"], label="kl_loss")
    plt.plot(epochs_arr, history["balance"], label="balance_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Style MoE training losses")
    curves_path = graphics_dir / "style_moe_training_curves.pdf"
    plt.savefig(curves_path, dpi=200)
    plt.close()
    logger.info(f"Training curves saved to {curves_path}")

    # Plot accuracy
    plt.figure(figsize=(8, 4))
    plt.plot(epochs_arr, history["accuracy"], marker="o")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Style MoE training accuracy")
    acc_path = graphics_dir / "style_moe_training_accuracy.pdf"
    plt.savefig(acc_path, dpi=200)
    plt.close()
    logger.info(f"Training accuracy plot saved to {acc_path}")

    # Plot gating heatmap
    if gating_history:
        gating_arr = np.array(gating_history)
        plt.figure(figsize=(12, max(2, gating_arr.shape[0] * 0.5)))
        plt.imshow(gating_arr, aspect="auto", cmap="viridis")
        plt.colorbar(label="mean gate weight")
        plt.xlabel("expert index")
        plt.ylabel("epoch")
        plt.title("Router gating usage over epochs")
        gating_fig_path = graphics_dir / "style_moe_gating_heatmap.pdf"
        plt.savefig(gating_fig_path, dpi=200)
        plt.close()
        logger.info(f"Gating heatmap saved to {gating_fig_path}")

    # Persist adapter
    save_dir = Path(config.paths.model) / "saved"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "style_moe.pth"
    torch.save(adapter.state_dict(), save_path)
    logger.info(f"Adapter saved to {save_path}")


if __name__ == "__main__":
    # Allow ad-hoc testing of the training script directly
    cfg = Config.from_yaml("config/default.yml")
    run_training(cfg)
