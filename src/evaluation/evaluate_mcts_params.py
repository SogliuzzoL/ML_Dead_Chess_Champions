"""Grid search and subsampled evaluation for MCTS hyperparameters.

This script provides a function `evaluate_mcts_params` that runs MCTS over a
subsample of the test set for combinations of `num_simulations`, `c_puct` and
`threshold`. For each configuration it saves a parquet with predictions and a
summary parquet with accuracy statistics.
"""

import os

# Force common BLAS/parallel libs to use a single thread to avoid over-subscribing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# Disable jemalloc background threads which can fail to spawn on some clusters
# (jemalloc reads MALLOC_CONF at process start)
os.environ["MALLOC_CONF"] = "background_thread:false"

import json
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple

import numpy as np
import polars as pl
from tqdm import tqdm

from src.core.config import Config
from src.core.utils import getLogger
from src.models.batched_mcts import BatchedMCTSManager
from src.models.maia import MaiaEngine

logger = getLogger()


def mcts_worker_params(
    fens_chunk: List[str],
    players_chunk: List[str],
    config: Config,
    num_simulations: int,
    batch_size: int,
    worker_id: int,
    c_puct: float,
    threshold: float,
    use_player_embeddings: bool = True,
) -> Tuple[List[str], List[str]]:
    """Worker executed in a separate process to run batched MCTS on its chunk.

    Returns a tuple of (best_moves_list, root_probs_json_list).
    """

    engine = MaiaEngine(config, use_player_embeddings=use_player_embeddings)
    mcts_manager = BatchedMCTSManager(engine, c_puct=c_puct, threshold=threshold)

    all_best_moves = []
    all_probs = []

    for i in tqdm(
        range(0, len(fens_chunk), batch_size),
        desc=f"Worker {worker_id + 1}",
        position=worker_id,
        leave=True,
    ):
        batch_fens = fens_chunk[i : i + batch_size]
        batch_players = players_chunk[i : i + batch_size]

        best_moves, root_probs_list = mcts_manager.run_batch(
            fens=batch_fens,
            active_elos=batch_players,
            num_simulations=num_simulations,
            temperature=0.0,  # deterministic selection by majority visits
        )

        all_best_moves.extend(best_moves)
        all_probs.extend([json.dumps(p) for p in root_probs_list])

    return all_best_moves, all_probs


def evaluate_mcts_params(
    config: Config,
    subsample_frac: float = 1.0,
    num_workers: int = 2,
    batch_size: int = 256,
    param_grid: dict | None = None,
    output_prefix: str = "mcts_grid",
    use_player_embeddings: bool = True,
    single_num_sim: int | None = None,
    single_c_puct: float | None = None,
    single_threshold: float | None = None,
) -> None:
    """Run a grid search over MCTS parameters on (optionally) a subsample of the test set.

    Args:
        config: Project configuration object.
        subsample_frac: Fraction of the test set to sample for each evaluation run.
                       Use 1.0 (default) to run on the full test set.
        num_workers: Number of worker processes to spawn for MCTS.
        batch_size: Batch size used by each worker when calling run_batch.
        param_grid: Dictionary with lists for keys `num_simulations`, `c_puct`, and `threshold`.
        output_prefix: Prefix for output files written to `config.paths.evaluation_dir`.
        use_player_embeddings: Whether to use project-specific per-player embeddings
                               when initializing the Maia engine. Set to False to run
                               MCTS with the base Maia model (no fine-tuned embeddings).
    """

    # Allow overriding a single run via environment variables so each grid cell
    # can be launched independently (useful when scheduling many jobs on a cluster).
    env_num = os.environ.get("MCTS_NUM_SIM")
    env_c = os.environ.get("MCTS_C_PUCT")
    env_thr = os.environ.get("MCTS_THRESHOLD")
    env_use_emb = os.environ.get("MCTS_USE_PLAYER_EMBEDDINGS")

    if env_use_emb is not None:
        use_player_embeddings = env_use_emb not in ("0", "false", "False")

    if param_grid is None:
        param_grid = {
            "num_simulations": [50, 100, 200, 1000],
            "c_puct": [0.5, 1.0, 1.5, 2.5],
            "threshold": [0.0, 0.01, 0.05, 0.1],
        }

    # If single-run environment variables are provided, restrict the grid to that single combo.
    if env_num is not None and env_c is not None and env_thr is not None:
        try:
            num_val = int(env_num)
            c_val = float(env_c)
            thr_val = float(env_thr)
            param_grid = {
                "num_simulations": [num_val],
                "c_puct": [c_val],
                "threshold": [thr_val],
            }
            # include the run identifier in the prefix for clarity
            output_prefix = f"{output_prefix}_sim{num_val}_c{str(c_val).replace('.', '_')}_thr{str(thr_val).replace('.', '_')}"
            logger.info(
                "Detected single-run environment variables; will execute only that configuration."
            )
        except Exception:
            logger.warning(
                "Failed to parse single-run MCTS environment variables; proceeding with full grid."
            )

    eval_dir = Path(config.paths.evaluation_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    df_test = pl.read_parquet(config.paths.test_set_path)
    n_total = df_test.height

    # If subsample_frac >= 1.0, use the full set.
    if subsample_frac >= 1.0:
        df_sub = df_test
        n_sub = n_total
    else:
        n_sub = max(1, int(math.ceil(n_total * subsample_frac)))
        logger.info(f"Total test examples: {n_total}. Using subsample of {n_sub}.")

        # For reproducible subsampling
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(n_total, size=n_sub, replace=False)

        # Polars DataFrame: use with_row_index (replacement for deprecated with_row_count)
        # Default index column name is 'index', but provide '__row_idx' for clarity and compatibility.
        df_test_idx = df_test.with_row_index("__row_idx")
        df_sub = df_test_idx.filter(pl.col("__row_idx").is_in(indices.tolist())).drop(
            "__row_idx"
        )

    fens_list = df_sub["fen"].to_list()
    players_list = df_sub["player_name"].to_list()
    true_moves = df_sub["move"].to_list()

    # Cap num_workers to reasonable values to avoid hitting system thread/process limits
    available_cpus = mp.cpu_count()
    num_workers = max(1, min(int(num_workers), available_cpus, 4))

    # Build chunks for workers
    chunk_size = math.ceil(len(fens_list) / num_workers) if len(fens_list) > 0 else 0
    chunks = []
    w_id = 0
    for i in range(0, len(fens_list), chunk_size):
        chunks.append(
            (fens_list[i : i + chunk_size], players_list[i : i + chunk_size], w_id)
        )
        w_id += 1

    results = []

    # Path for summary (will be created/overwritten during the grid search)
    summary_path = eval_dir / f"{output_prefix}_summary.parquet"

    total_runs = (
        len(param_grid["num_simulations"])
        * len(param_grid["c_puct"])
        * len(param_grid["threshold"])  # type: ignore
    )
    run_idx = 0

    for num_sim in param_grid["num_simulations"]:
        for c in param_grid["c_puct"]:
            for thr in param_grid["threshold"]:
                run_idx += 1
                logger.info(
                    f"Starting run {run_idx}/{total_runs}: sim={num_sim}, c={c}, thr={thr}"
                )
                start_time = time.time()

                mcts_preds = []
                mcts_probs = []

                # Ensure multiprocessing uses the 'spawn' start method so CUDA can be initialized in child processes
                try:
                    mp.set_start_method("spawn", force=True)
                except RuntimeError:
                    pass

                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = []
                    for fens_chunk, players_chunk, worker_id in chunks:
                        futures.append(
                            executor.submit(
                                mcts_worker_params,
                                fens_chunk,
                                players_chunk,
                                config,
                                num_sim,
                                batch_size,
                                worker_id,
                                c,
                                thr,
                                use_player_embeddings,
                            )
                        )

                    for future in futures:
                        b_moves, b_probs = future.result()
                        mcts_preds.extend(b_moves)
                        mcts_probs.extend(b_probs)

                # Ensure length matches (sub)dataset
                if len(mcts_preds) != len(true_moves):
                    logger.warning(
                        f"Produced {len(mcts_preds)} preds but expected {len(true_moves)}. Truncating or padding with None."
                    )

                # Truncate or pad
                if len(mcts_preds) > len(true_moves):
                    mcts_preds = mcts_preds[: len(true_moves)]
                    mcts_probs = mcts_probs[: len(true_moves)]
                else:
                    mcts_preds.extend([None] * (len(true_moves) - len(mcts_preds)))
                    mcts_probs.extend(
                        [json.dumps({})] * (len(true_moves) - len(mcts_probs))
                    )

                # Compute top-1 accuracy (ignores None)
                correct = 0
                total = 0
                for t, p in zip(true_moves, mcts_preds):
                    if p is None:
                        continue
                    total += 1
                    if p == t:
                        correct += 1
                accuracy = float(correct / total) if total > 0 else 0.0

                elapsed = time.time() - start_time
                logger.info(
                    f"Run finished in {elapsed:.1f}s - accuracy={accuracy:.4f} (on {total} evaluated)"
                )

                # Save predictions parquet for this configuration
                out_df = df_sub.select(
                    ["game_id", "fen", "player_name", "move"]
                ).rename({"move": "true_move"})
                out_df = out_df.with_columns(
                    [
                        pl.Series("pred_mcts", mcts_preds),
                        pl.Series("probs_mcts", mcts_probs),
                    ]
                )

                sanitized_c = str(c).replace(".", "_")
                sanitized_thr = str(thr).replace(".", "_")
                out_path = (
                    eval_dir
                    / f"{output_prefix}_sim{num_sim}_c{sanitized_c}_thr{sanitized_thr}.parquet"
                )
                out_df.write_parquet(str(out_path))

                results.append(
                    {
                        "num_simulations": num_sim,
                        "c_puct": c,
                        "threshold": thr,
                        "accuracy": accuracy,
                        "evaluated": total,
                        "elapsed_s": elapsed,
                        "predictions_path": str(out_path),
                    }
                )

                # Save intermediate summary (parquet)
                summary_df = pl.DataFrame(results)
                summary_path = (
                    Path(config.paths.evaluation_dir)
                    / f"{output_prefix}_summary.parquet"
                )
                summary_df.write_parquet(str(summary_path))

    logger.info(f"Grid search completed. Summary saved to {summary_path}")
