"""Compute and persist per-player blunder statistics using model predictions.

This module contains two approaches:

- compute_blunder_rates: a lightweight, probability-threshold-based method that
  treats a low model probability on the true move as a proxy for a blunder.
- compute_blunder_rates_stockfish: a Stockfish-based method that uses an
  engine to determine whether the true move (and/or model predictions) are
  blunders by comparing centipawn differences.

The Stockfish-based routine is the recommended, more accurate approach. It is
configurable (engine path, depth/time, parallel workers) via the project's
configuration and writes both a per-move detailed parquet and a per-player
aggregated parquet for downstream reporting.
"""

import json
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import chess
import chess.engine
import numpy as np
import polars as pl
from tqdm import tqdm

from src.core.config import Config
from src.core.utils import getLogger

logger = getLogger()

# Map logical variant -> column storing the serialized probability dict / prediction column
VARIANT_TO_PROBS_COL: Dict[str, str] = {
    "maia2": "probs_baseline",
    "maia2_ft": "probs_custom",
    "maia2_ft_mcts": "probs_mcts",
}

VARIANT_TO_PRED_COL: Dict[str, str] = {
    "maia2": "pred_baseline",
    "maia2_ft": "pred_custom",
    "maia2_ft_mcts": "pred_mcts",
}


def _ensure_predictions(config: Config) -> pl.DataFrame:
    """Ensure the predictions parquet exists; generate it if missing."""
    path = config.paths.predictions_path
    if not Path(path).exists():
        from src.evaluation.evaluate_players import generate_predictions_parquet

        logger.info("Predictions parquet not found — generating predictions...")
        # Use default number of MCTS sims (function will choose its default)
        generate_predictions_parquet(config)

    return pl.read_parquet(path)


def _score_to_cp(score_obj: Any, color: chess.Color) -> Optional[int]:
    """Convert a python-chess Score object into a centipawn integer oriented to `color`.

    Returns a large sentinel for mate (±100000) and None if the score cannot be
    interpreted.
    """
    try:
        # Some python-chess versions return a PovScore from analyse(); call pov()
        # to orient it to the mover's color and then extract centipawns via score().
        oriented = score_obj.pov(color)
        cp = oriented.score(mate_score=100000)
        if cp is None:
            return None
        return int(cp)
    except Exception:
        # Fallbacks for alternative score representations
        try:
            if hasattr(score_obj, "cp") and getattr(score_obj, "cp") is not None:
                return int(getattr(score_obj, "cp"))
        except Exception:
            pass
        try:
            mate = getattr(score_obj, "mate", None)
            if mate is not None:
                return 100000 if mate > 0 else -100000
        except Exception:
            pass
    return None


def _eval_move_cp(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    move_uci: str,
    mover_color: chess.Color,
    limit: chess.engine.Limit,
) -> Optional[int]:
    """Evaluate the position after applying `move_uci` and return centipawn from mover_color perspective.

    Returns None if the move is illegal or an error occurs.
    """
    if not move_uci:
        return None
    try:
        move = chess.Move.from_uci(move_uci)
    except Exception:
        return None

    if move not in board.legal_moves:
        return None

    try:
        b2 = board.copy()
        b2.push(move)
        info = engine.analyse(b2, limit)
        score_obj = info.get("score")
        if score_obj is None:
            return None
        return _score_to_cp(score_obj, mover_color)
    except Exception:
        return None


def _stockfish_worker(
    fens: List[str],
    moves_by_fen: Dict[str, List[str]],
    stockfish_cfg: Dict,
    cp_threshold: int,
    depth: Optional[int],
    worker_id: int,
):
    """Worker that runs a Stockfish engine and evaluates a set of unique fens.

    For each assigned fen this worker computes:
      - best move (via engine.play)
      - cp of position after best move
      - cp of position after each move in moves_by_fen[fen]

    Returns a list of per-fen dictionaries to be merged by the caller.
    """
    results: List[Dict] = []

    engine_path = stockfish_cfg.get("path", "stockfish")
    engine_depth = depth or stockfish_cfg.get("depth", 12)
    analysis_time = stockfish_cfg.get("analysis_time")
    threads = stockfish_cfg.get("num_threads", 1)

    # Build limit object
    if analysis_time and analysis_time > 0:
        limit = chess.engine.Limit(time=float(analysis_time))
    else:
        limit = chess.engine.Limit(depth=int(engine_depth))

    try:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        # Try to set thread count if engine accepts it
        try:
            engine.configure({"Threads": int(threads)})
        except Exception:
            pass
    except Exception as e:
        logger.error(
            "Worker %s: Failed to start Stockfish at %s: %s", worker_id, engine_path, e
        )
        return results

    for fen in tqdm(
        fens, desc=f"Worker {worker_id + 1}", position=worker_id, leave=True
    ):
        try:
            board = chess.Board(fen)
        except Exception:
            # Skip invalid FEN
            continue

        mover_color = board.turn

        # Best move
        try:
            play_res = engine.play(board, limit)
            best_move = (
                play_res.move.uci() if play_res and play_res.move is not None else None
            )
        except Exception:
            best_move = None

        cp_best = None
        if best_move is not None:
            try:
                b_best = board.copy()
                b_best.push(chess.Move.from_uci(best_move))
                info_best = engine.analyse(b_best, limit)
                cp_best = _score_to_cp(info_best.get("score"), mover_color)
            except Exception:
                cp_best = None

        # Evaluate requested moves for this fen
        cp_moves: Dict[str, Optional[int]] = {}
        moves = moves_by_fen.get(fen, []) or []
        for mv in moves:
            if mv is None:
                cp_moves[mv] = None
                continue
            try:
                cp_mv = _eval_move_cp(engine, board, mv, mover_color, limit)
            except Exception:
                cp_mv = None
            cp_moves[mv] = cp_mv

        results.append(
            {
                "fen": fen,
                "best_move": best_move,
                "cp_best": cp_best,
                "cp_moves": cp_moves,
            }
        )

    try:
        engine.quit()
    except Exception:
        pass

    return results


def compute_blunder_rates_stockfish(
    config: Config,
    cp_threshold: int = 200,
    depth: Optional[int] = None,
    subsample_frac: float = 1.0,
    num_workers: Optional[int] = None,
) -> None:
    """Compute Stockfish-based blunder labels and per-player aggregated rates.

    This implementation first deduplicates positions globally: it builds the
    set of unique FENs and the set of moves (true + model predictions) that need
    to be evaluated for each FEN. Each worker then evaluates its assigned unique
    FENs once (best move + requested move evaluations). Results are merged and
    expanded back to a per-row detailed table before aggregation.
    """
    df = _ensure_predictions(config)

    n_total = len(df)
    if n_total == 0:
        logger.warning("No prediction rows found in %s", config.paths.predictions_path)
        return

    # Optional subsample
    if subsample_frac < 1.0 and subsample_frac > 0.0:
        n_sub = max(1, int(math.ceil(n_total * subsample_frac)))
        logger.info("Subsampling %d/%d positions for blunder analysis", n_sub, n_total)
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(n_total, size=n_sub, replace=False)
        df_idx = df.with_row_index("__row_idx")
        df = df_idx.filter(pl.col("__row_idx").is_in(indices.tolist())).drop(
            "__row_idx"
        )

    # Extract lists
    fens = df["fen"].to_list()
    true_moves = df["true_move"].to_list()
    players = df["player_name"].to_list()

    preds_by_variant: Dict[str, List[Optional[str]]] = {}
    for variant, col in VARIANT_TO_PRED_COL.items():
        preds_by_variant[variant] = (
            df[col].to_list() if col in df.columns else [None] * len(fens)
        )

    # Build mapping: fen -> list(row indices)
    fen_rows: Dict[str, List[int]] = {}
    for idx, fen in enumerate(fens):
        fen_rows.setdefault(fen, []).append(idx)

    # Build mapping: fen -> set of moves to evaluate (true move + model preds)
    moves_by_fen: Dict[str, List[str]] = {}
    for fen, idxs in fen_rows.items():
        moves = set()
        for i in idxs:
            tm = true_moves[i]
            if tm is not None:
                moves.add(tm)
            for variant in VARIANT_TO_PRED_COL.keys():
                pm = preds_by_variant[variant][i]
                if pm is not None:
                    moves.add(pm)
        moves_by_fen[fen] = list(moves)

    unique_fens = list(moves_by_fen.keys())
    n_unique = len(unique_fens)
    logger.info(
        "Unique positions to evaluate: %d (original rows: %d)", n_unique, len(fens)
    )

    # Determine worker count (bounded by number of unique positions)
    workers = num_workers or int(config.stockfish.num_workers or 1)
    workers = max(1, min(workers, n_unique))

    # Chunk unique fen list
    chunk_size = math.ceil(n_unique / workers)
    fen_chunks: List[List[str]] = [
        unique_fens[i : i + chunk_size] for i in range(0, n_unique, chunk_size)
    ]

    # Prepare per-chunk moves_by_fen subsets to avoid passing the full map each time
    chunks: List[tuple] = []
    for chunk in fen_chunks:
        subset = {fen: moves_by_fen[fen] for fen in chunk}
        chunks.append((chunk, subset))

    # Dispatch workers
    all_per_fen_results: List[Dict] = []

    sf_cfg = config.stockfish.dict()

    if workers == 1:
        fens_chunk, moves_subset = chunks[0]
        res = _stockfish_worker(
            fens_chunk, moves_subset, sf_cfg, cp_threshold, depth, 0
        )
        all_per_fen_results.extend(res)
    else:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = []
            for w_id, (fens_c, moves_c) in enumerate(chunks):
                futures.append(
                    ex.submit(
                        _stockfish_worker,
                        fens_c,
                        moves_c,
                        sf_cfg,
                        cp_threshold,
                        depth,
                        w_id,
                    )
                )

            # Show a progress bar for worker completion
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc="Workers finished"
            ):
                try:
                    r = fut.result()
                    all_per_fen_results.extend(r)
                except Exception as e:
                    logger.error("A worker failed: %s", e)

    if not all_per_fen_results:
        logger.warning(
            "No Stockfish evaluation results produced; aborting blunder aggregation."
        )
        return

    # Merge per-fen results into a lookup
    fen_eval: Dict[str, Dict] = {r["fen"]: r for r in all_per_fen_results}

    # Expand per-row results using fen_eval
    row_results: List[Dict] = []
    for i, fen in enumerate(fens):
        per = fen_eval.get(fen, {})
        cp_best = per.get("cp_best") if per else None
        cp_actual = None
        if per:
            cp_actual = per.get("cp_moves", {}).get(true_moves[i])
        delta_actual = (
            cp_best - cp_actual
            if cp_best is not None and cp_actual is not None
            else None
        )
        is_blunder_actual = (
            delta_actual >= cp_threshold if delta_actual is not None else None
        )

        row_res: Dict = {
            "player_name": players[i],
            "fen": fen,
            "true_move": true_moves[i],
            "cp_best": cp_best,
            "cp_actual": cp_actual,
            "delta_actual": delta_actual,
            "is_blunder_actual": is_blunder_actual,
        }

        for variant in VARIANT_TO_PRED_COL.keys():
            pred_move = (
                preds_by_variant[variant][i] if variant in preds_by_variant else None
            )
            cp_pred = per.get("cp_moves", {}).get(pred_move) if per else None
            delta_pred = (
                cp_best - cp_pred
                if cp_best is not None and cp_pred is not None
                else None
            )
            is_blunder_pred = (
                delta_pred >= cp_threshold if delta_pred is not None else None
            )
            row_res[f"cp_pred_{variant}"] = cp_pred
            row_res[f"delta_pred_{variant}"] = delta_pred
            row_res[f"is_blunder_pred_{variant}"] = is_blunder_pred

        row_results.append(row_res)

    # Create a detailed per-move DataFrame
    df_moves = pl.DataFrame(row_results)

    # Persist detailed results if configured
    detailed_path = (
        config.paths.player_blunders_detailed_path
        if hasattr(config.paths, "player_blunders_detailed_path")
        else None
    )
    if detailed_path:
        try:
            df_moves.write_parquet(detailed_path)
            logger.info(
                "Wrote detailed per-move Stockfish evaluations to %s", detailed_path
            )
        except Exception as e:
            logger.warning("Failed to write detailed Stockfish results: %s", e)

    # Aggregate per-player statistics
    agg_exprs = [pl.count().alias("n_moves")]

    # Actual (human) move statistics
    agg_exprs += [
        pl.col("is_blunder_actual").sum().alias("actual_blunder_count"),
        pl.col("is_blunder_actual").mean().alias("actual_blunder_rate"),
        pl.col("delta_actual").mean().alias("actual_mean_delta_cp"),
    ]

    # Per-variant stats
    for variant in VARIANT_TO_PRED_COL.keys():
        agg_exprs += [
            pl.col(f"is_blunder_pred_{variant}")
            .sum()
            .alias(f"{variant}_blunder_count"),
            pl.col(f"is_blunder_pred_{variant}")
            .mean()
            .alias(f"{variant}_blunder_rate"),
            pl.col(f"delta_pred_{variant}").mean().alias(f"{variant}_mean_delta_cp"),
        ]

    df_players = df_moves.group_by("player_name").agg(agg_exprs).sort("player_name")

    out_path = config.paths.player_blunders_path
    df_players.write_parquet(out_path)

    logger.info("Player blunder statistics (Stockfish) written to %s", out_path)


# Keep the lightweight probability-based function for backward compatibility
def compute_blunder_rates(config: Config, prob_threshold: float = 0.01) -> None:
    """Compute and persist per-player blunder statistics using model probabilities.

    (This function was the original quick heuristic and is preserved for
    compatibility. Prefer `compute_blunder_rates_stockfish` for more reliable
    blunder detection.)
    """
    # Reuse the previously-implemented probability-based routine
    df = _ensure_predictions(config)

    # We'll add per-row helper columns for each variant: true_prob_{variant}, true_rank_{variant}
    # Convert needed columns to python lists for fast iteration once
    true_moves = df["true_move"].to_list()

    # Work on a local copy of the DataFrame to append new columns
    working_df = df

    for variant, probs_col in VARIANT_TO_PROBS_COL.items():
        probs_list = working_df[probs_col].to_list()

        true_prob_list = []
        true_rank_list = []

        for pjson, true in zip(probs_list, true_moves):
            try:
                probs = json.loads(pjson) if pjson is not None else {}
                # If the stored structure is not a dict (unexpected), coerce if possible
                if not isinstance(probs, dict):
                    try:
                        probs = dict(probs)
                    except Exception:
                        probs = {}
            except Exception:
                probs = {}

            # Probability assigned to the actual move (0.0 if missing)
            prob_true = float(probs.get(true, 0.0)) if probs else 0.0
            true_prob_list.append(prob_true)

            # Rank of the true move (1 = highest probability). If no probs available -> NaN
            if probs:
                try:
                    # rank = 1 + number of moves with strictly greater probability
                    rank = 1 + sum(1 for v in probs.values() if v > prob_true)
                    true_rank_list.append(float(rank))
                except Exception:
                    true_rank_list.append(float("nan"))
            else:
                true_rank_list.append(float("nan"))

        # Attach columns
        working_df = working_df.with_columns(
            [
                pl.Series(f"true_prob_{variant}", true_prob_list),
                pl.Series(f"true_rank_{variant}", true_rank_list),
            ]
        )

    # Build aggregation expressions per player
    agg_exprs = [pl.count().alias("n_moves")]

    for variant in VARIANT_TO_PROBS_COL.keys():
        tp = f"true_prob_{variant}"
        tr = f"true_rank_{variant}"

        agg_exprs += [
            pl.col(tp).mean().alias(f"{variant}_mean_true_prob"),
            pl.col(tp).median().alias(f"{variant}_median_true_prob"),
            (pl.col(tp) <= prob_threshold).sum().alias(f"{variant}_blunder_count"),
            (pl.col(tp) <= prob_threshold).mean().alias(f"{variant}_blunder_rate"),
            pl.col(tr).mean().alias(f"{variant}_mean_true_rank"),
        ]

    df_results = working_df.groupby("player_name").agg(agg_exprs).sort("player_name")

    out_path = config.paths.player_blunders_path
    df_results.write_parquet(out_path)

    logger.info(
        "Player blunder statistics written to %s (prob_threshold=%s) [prob-based]",
        out_path,
        prob_threshold,
    )
