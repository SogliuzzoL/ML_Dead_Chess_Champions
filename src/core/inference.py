import chess
import torch
from maia2.inference import (
    board_to_tensor,
    create_elo_dict,
    get_all_possible_moves,
    mirror_move,
)

from .config import PLAYER_REFERENCE


def create_player_dict(start: int) -> dict:
    player_dict = {}
    for i, player in enumerate(PLAYER_REFERENCE):
        player_dict[player] = i + start
    return player_dict


def prepare() -> list[dict]:
    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}

    elo_dict = create_elo_dict()
    last_elo = max(elo_dict.values())

    player_dict = create_player_dict(last_elo + 1)

    final_dict = {**player_dict, **elo_dict}

    all_moves_dict_reversed = {v: k for k, v in all_moves_dict.items()}

    return [all_moves_dict, final_dict, all_moves_dict_reversed]


def preprocessing(fen: str, player: str, player_dict: dict, all_moves_dict: dict):
    if fen.split(' ')[1] == 'w':
        board = chess.Board(fen)
    elif fen.split(' ')[1] == 'b':
        board = chess.Board(fen).mirror()
    else:
        raise ValueError(f"Invalid fen: {fen}")

    board_input = board_to_tensor(board)

    player_idx = player_dict.get(player)

    legal_moves = torch.zeros(len(all_moves_dict))
    legal_moves_idx = torch.tensor(
        [all_moves_dict[move.uci()] for move in board.legal_moves])
    legal_moves[legal_moves_idx] = 1

    return board_input, player_idx, legal_moves


def inference_each(model, prepared, fen, player):
    all_moves_dict, player_dict, all_moves_dict_reversed = prepared

    board_input, player_idx, legal_moves = preprocessing(
        fen, player, player_dict, all_moves_dict)

    device = next(model.parameters()).device

    model.eval()

    board_input = board_input.unsqueeze(dim=0).to(device)
    player = torch.tensor([player_idx]).to(device)
    legal_moves = legal_moves.unsqueeze(dim=0).to(device)

    logits_maia, _, logits_value = model(board_input, player)
    logits_maia_legal = logits_maia * legal_moves
    probs = logits_maia_legal.softmax(dim=-1).cpu().tolist()

    logits_value = (logits_value / 2 + 0.5).clamp(0, 1).item()

    black_flag = False
    if fen.split(" ")[1] == "b":
        logits_value = 1 - logits_value
        black_flag = True
    win_prob = round(logits_value, 4)

    move_probs = {}
    legal_move_indices = legal_moves.nonzero().flatten().cpu().numpy().tolist()
    legal_moves_mirrored = []
    for move_idx in legal_move_indices:
        move = all_moves_dict_reversed[move_idx]
        if black_flag:
            move = mirror_move(move)
        legal_moves_mirrored.append(move)

    for j in range(len(legal_move_indices)):
        move_probs[legal_moves_mirrored[j]] = round(
            probs[0][legal_move_indices[j]], 4)

    move_probs = dict(
        sorted(move_probs.items(), key=lambda item: item[1], reverse=True))

    return move_probs, win_prob
