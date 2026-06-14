"""Inference / prediction workflow for the trained StyleMoE adapter.

This script demonstrates how to load the frozen Maia backbone and a trained
StyleMoE adapter, apply the adapter to a batch of positions (from the test set)
and return top-k predicted moves. It mirrors the prediction flow used during
training: obtain Maia logits + hidden v, compute adapter delta_logits and add
before applying legal masking and softmax.

The script is intentionally minimal and intended for ad-hoc evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import chess
import polars as pl
import torch
from maia2 import inference, model
from maia2.utils import board_to_tensor

from src.core.config import Config
from src.core.utils import getLogger
from src.models.style_moe import StyleMoE

logger = getLogger()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_adapter_for_inference(config: Config) -> StyleMoE:
    save_path = Path(config.paths.model) / "saved" / "style_moe.pth"
    if not save_path.exists():
        raise FileNotFoundError(f"Adapter not found at {save_path}; train it first.")

    # Instantiate a small Maia -> forward sample to infer shapes
    maia_model = model.from_pretrained("rapid", DEVICE)
    maia_model.eval()

    # Dummy board for shape inference
    b = chess.Board()
    board_t = board_to_tensor(b).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits, v, _ = maia_model(
            board_t, torch.tensor([0]).to(DEVICE), torch.tensor([0]).to(DEVICE)
        )

    v_dim = v.size(-1)
    out_dim = logits.size(-1)
    board_dim = board_t.numel() // board_t.size(0)

    adapter = StyleMoE(v_dim=v_dim, out_dim=out_dim, board_dim=board_dim)
    adapter.load_state_dict(torch.load(save_path, map_location=DEVICE))
    adapter.to(DEVICE)
    adapter.eval()

    logger.info(f"Loaded adapter from {save_path}")
    return adapter


def predict_batch(
    adapter: StyleMoE,
    maia_model,
    fens: List[str],
    active_idx: int = 2500,
    top_k: int = 3,
):
    # Use Maia's canonical prepared move mapping so indices align with model outputs
    all_moves_dict, elo_dict, all_moves_dict_reversed = inference.prepare()
    move_to_idx = all_moves_dict
    idx_to_move = all_moves_dict_reversed

    board_tensors = []
    for f in fens:
        b = chess.Board(f)
        # Mirror black positions to match Maia input convention
        if b.turn == chess.BLACK:
            b = b.mirror()
        board_tensors.append(board_to_tensor(b))

    boards = torch.stack(board_tensors, dim=0).to(DEVICE)
    # Map provided active_idx (elo) to Maia category index using prepared elo_dict
    active_maia_idx = inference.map_to_category(active_idx, elo_dict)
    active_ids = torch.tensor([active_maia_idx] * len(fens)).to(DEVICE)
    opponent_ids = active_ids

    with torch.no_grad():
        logits_maia, v, _ = maia_model(boards, active_ids, opponent_ids)

    # For inference we don't have the last-K sequence here; we simply set seq input to zero
    seq_len = adapter.seq_len
    seq_input = torch.zeros((len(fens), seq_len * adapter.board_dim), device=DEVICE)

    with torch.no_grad():
        delta_logits, _, g = adapter(v, seq_input)

    final_logits = logits_maia + delta_logits

    # Mask illegal moves and compute probabilities
    results = []
    for i, f in enumerate(fens):
        b = chess.Board(f)
        # Mirror black positions to match Maia internal representation
        if b.turn == chess.BLACK:
            b = b.mirror()
        out_dim = int(final_logits.size(-1))
        mask = torch.zeros(out_dim, dtype=torch.bool, device=DEVICE)
        for m in b.legal_moves:
            idx = move_to_idx.get(m.uci())
            if idx is not None:
                if 0 <= int(idx) < out_dim:
                    mask[int(idx)] = True
                else:
                    logger.debug(
                        f"Skipping illegal move idx {idx} >= out_dim {out_dim} for fen {f}"
                    )
        # Copy logits for this example and mask
        logits_i = final_logits[i].masked_fill(~mask, -1e9)
        probs = torch.softmax(logits_i, dim=-1)
        topk = torch.topk(probs, top_k)
        topk_inds = topk.indices.cpu().tolist()
        topk_vals = topk.values.cpu().tolist()
        moves = [
            (idx_to_move[int(ind)], float(val))
            for ind, val in zip(topk_inds, topk_vals)
        ]
        results.append(moves)

    return results


def run_inference_demo(config: Config, n_examples: int = 20, top_k: int = 3) -> None:
    maia_model = model.from_pretrained("rapid", DEVICE)
    maia_model.eval()

    adapter = load_adapter_for_inference(config)

    logger.info("Loading a few examples from the test set for demonstration...")
    df_test = pl.read_parquet(config.paths.test_set_path)
    fens = df_test[:n_examples]["fen"].to_list()

    logger.info(f"Predicting top-{top_k} moves for {len(fens)} positions...")
    preds = predict_batch(adapter, maia_model, fens, top_k=top_k)

    for i, moves in enumerate(preds):
        logger.info(f"Position {i + 1} top-{top_k}: {moves}")


if __name__ == "__main__":
    cfg = Config.from_yaml("config/default.yml")
    run_inference_demo(cfg)
