import chess.engine
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from core.config import (
    DATASET_PATH,
    STOCKFISH_CPL_ANALYSE_PATH,
    STOCKFISH_MODEL_PATH,
    logger,
)


def analyse_with_stockfish(fen: str, move_uci: str, deph_limit: int = 20):
    board = chess.Board(fen)
    if board is None:
        print(f"Invalid FEN: {fen}")
        return

    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        logger.warning(f"Illegal move: {move_uci} for position {fen}")
        return

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_MODEL_PATH)
    limit = chess.engine.Limit(depth=deph_limit)
    info_before = engine.analyse(board, limit=limit)
    score_before = info_before.get("score", None)
    if score_before is None:
        logger.warning(f"Could not get score for position {fen}")
        engine.quit()
        return

    board.push(move)
    info_after = engine.analyse(board, limit=limit)
    score_after = info_after.get("score", None)

    if score_after is None:
        logger.warning(
            f"Could not get score for position after move {move_uci}")
        engine.quit()
        return

    score_before = score_before.relative.score(mate_score=100000)
    score_after = score_after.relative.score(mate_score=100000)
    diff = score_before - (-score_after)
    cpl = max(0, diff)

    engine.quit()

    return cpl


if __name__ == "__main__":
    df = pd.read_parquet(DATASET_PATH)
    result = []
    player_list = ["Tal", "Capablanca"]
    for player in player_list:
        logger.info(f"Analyzing games for player: {player}")
        player_df = df[df["player_name"] == player]
        for j, game_id in enumerate(player_df["game_id"].unique()):
            game_df = player_df[player_df["game_id"] == game_id]
            logger.info(
                f"Analyzing game ID: {game_id} with {len(game_df)} moves")
            for i, (_, row) in enumerate(game_df.iterrows()):
                logger.info(
                    f"Analyzing move {i + 1}/{len(game_df)} for game ID: {game_id}")
                fen = row["fen"]
                move_uci = row["move"]
                cpl = analyse_with_stockfish(fen, move_uci)
                if cpl is not None:
                    result.append({
                        "player": player,
                        "move_number": i,
                        "cpl": cpl
                    })

    df_result = pd.DataFrame(result)
    df_result.to_csv(STOCKFISH_CPL_ANALYSE_PATH, index=False)
