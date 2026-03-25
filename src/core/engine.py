import io
import os

import chess
import chess.pgn
import numpy as np
import torch
from chess.engine import SimpleEngine
from maia2 import inference, model
from maia2.utils import board_to_tensor, mirror_move
from tqdm import tqdm

from models.player_style import PlayerStyleEmbedding

from .config import (
    CHAMPIONS_EMBEDDINGS_PATH,
    base_player_dict,
    logger,
)
from .mcts import MCTS


class MaiaEngine:
    def __init__(self, model_type="rapid"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.from_pretrained(model_type, self.device)
        self.prepare = inference.prepare()

        n_players = len(base_player_dict)
        self.model.elo_embedding = PlayerStyleEmbedding(
            self.model.elo_embedding, n_players
        ).to(self.device)

        if os.path.exists(CHAMPIONS_EMBEDDINGS_PATH):
            state_dict = torch.load(CHAMPIONS_EMBEDDINGS_PATH, map_location=self.device)
            self.model.elo_embedding.players_embeddings.load_state_dict(state_dict)
        else:
            logger.warning(
                f"Champions embeddings not found at {CHAMPIONS_EMBEDDINGS_PATH}."
            )

        self.player_to_idx = {
            player: idx + self.model.elo_embedding.max_maia_idx + 1
            for idx, player in enumerate(base_player_dict.values())
        }

    def get_board_from_fen(self, fen, pgn):
        board = chess.Board()
        if pgn != "":
            try:
                pgn_io = io.StringIO(pgn)
                game = chess.pgn.read_game(pgn_io)
            except Exception:
                game = None
            if game is not None:
                for move in game.mainline_moves():
                    board.push(move)
            else:
                board = chess.Board(fen)
        else:
            board = chess.Board(fen)
        return board

    def predict_mcts(
        self,
        fen,
        pgn,
        stockfish: SimpleEngine,
        num_simulations=50,
        c_puct=1.5,
        scale=400.0,
        threshold=0.01,
        active_elo: int | str = 2500,
        opponent_elo: int | str = 2500,
    ):
        board = self.get_board_from_fen(fen, pgn)
        mcts = MCTS(self.predict_move, stockfish)
        best_move, result, tree_data = mcts.run(
            board,
            num_simulations,
            threshold=threshold,
            c_puct=c_puct,
            scale=scale,
            activ_elo=active_elo,
            opp_elo=opponent_elo,
        )

        return best_move, result, tree_data

    def _get_style_idx(self, val: int | str):
        if isinstance(val, str) and val in self.player_to_idx:
            return self.player_to_idx[val]

        _, elo_dict, _ = self.prepare
        if isinstance(val, str) and val in elo_dict:
            return elo_dict[val]

        return inference.map_to_category(int(val), elo_dict)

    def predict_move(
        self, fen, active_elo: int | str = 2500, opponent_elo: int | str = 2500
    ):
        board = chess.Board(fen)
        is_mirrored = False
        if board.turn == chess.BLACK:
            board = board.mirror()
            is_mirrored = True

        device = self.device
        board_tensor = board_to_tensor(board).unsqueeze(0).to(device)
        s_self = torch.tensor([self._get_style_idx(active_elo)]).to(device)
        s_oppo = torch.tensor([self._get_style_idx(opponent_elo)]).to(device)

        self.model.eval()
        with torch.no_grad():
            logits_maia, _, logits_value = self.model(board_tensor, s_self, s_oppo)
            all_moves_dict, _, all_moves_dict_reversed = self.prepare
            legal_mask = torch.zeros(logits_maia.size(-1)).to(device)
            for move in board.legal_moves:
                legal_mask[all_moves_dict[move.uci()]] = 1

            probs = (logits_maia[0] * legal_mask).softmax(dim=-1).cpu().numpy()

        move_probs = {}
        for i in legal_mask.nonzero().flatten().tolist():
            move_uci = all_moves_dict_reversed[i]
            final_move = mirror_move(move_uci) if is_mirrored else move_uci
            move_probs[final_move] = float(probs[i])

        sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
        best_move = sorted_moves[0][0]

        return best_move, dict(sorted_moves), logits_value.item()

    def evaluate_batch(self, dataloader):
        self.model.eval()
        all_correct_preds = []
        all_player_ids = []

        with torch.no_grad():
            for boards, active_ids, opponent_ids, labels, legal_masks in tqdm(
                dataloader, desc="Batch Evaluation"
            ):
                boards = boards.to(self.device, non_blocking=True)
                active_ids = active_ids.to(self.device, non_blocking=True)
                opponent_ids = opponent_ids.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                legal_masks = legal_masks.to(self.device, non_blocking=True)

                logits, _, _ = self.model(boards, active_ids, opponent_ids)

                logits = logits.masked_fill(~legal_masks, -float("inf"))
                predictions = logits.argmax(dim=-1)

                all_correct_preds.append((predictions == labels).cpu().numpy())
                all_player_ids.append(active_ids.cpu().numpy())

        return np.concatenate(all_correct_preds), np.concatenate(all_player_ids)
