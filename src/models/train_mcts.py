import logging
import os

import optuna
import pandas as pd
from chess.engine import SimpleEngine
from tqdm import tqdm

from core.config import ProjectConfig
from core.engine import MaiaEngine

logger = logging.getLogger(__name__)


def create_objective(config: ProjectConfig, df_player: pd.DataFrame, player_name: str):
    """
    Constructs the Optuna objective function, utilizing the dynamic configuration
    for engine instantiation and Stockfish path resolution.
    """

    def objective(trial):
        # Instantiate the modernized engine requiring the config object
        engine = MaiaEngine(config=config)
        stockfish = SimpleEngine.popen_uci(config.stockfish_model_path)

        c_puct = trial.suggest_float("c_puct", 0.1, 10.0, log=True)
        scale = trial.suggest_float("scale", 100.0, 1000.0, log=True)
        threshold = trial.suggest_float("threshold", 0.001, 0.1, log=True)
        num_simulations = trial.suggest_int("num_simulations", 30, 500, log=True)

        correct_predictions = 0

        for _, row in df_player.iterrows():
            best_move, _, _ = engine.predict_mcts(
                fen=row["fen"],
                pgn="",
                stockfish=stockfish,
                num_simulations=num_simulations,
                threshold=threshold,
                active_elo=player_name,
                c_puct=c_puct,
                scale=scale,
            )
            if best_move == row["move"]:
                correct_predictions += 1

        stockfish.quit()
        return correct_predictions / len(df_player)

    return objective


def optimize_mcts(config: ProjectConfig):
    """
    Orchestrates the MCTS hyperparameter optimization across all defined champions.
    """
    logger.info(f"Loading training data from {config.train_set_path}")
    df = pd.read_parquet(config.train_set_path)
    results_list = []

    for player_id, player_name in config.base_player_dict.items():
        df_player = df[df["player_name"] == player_name].copy()

        if df_player.empty:
            logger.warning(
                f"No data found for player {player_name} in the evaluation sequence."
            )
            continue

        study = optuna.create_study(
            study_name=f"mcts_optim_{player_name}",
            storage=f"sqlite:///{config.mcts_optimization_db_path}",
            direction="maximize",
            load_if_exists=True,
        )

        objective_function = create_objective(config, df_player, player_name)

        with tqdm(
            total=50, desc=f"Trials {player_name}", leave=False, position=1
        ) as pbar:

            def optuna_callback(study, trial):
                pbar.update(1)

            study.optimize(
                objective_function, n_trials=50, n_jobs=4, callbacks=[optuna_callback]
            )

        best_params = study.best_params

        results_list.append(
            {
                "player_id": player_id,
                "player_name": player_name,
                "num_simulations": best_params["num_simulations"],
                "c_puct": best_params["c_puct"],
                "scale": best_params["scale"],
                "threshold": best_params["threshold"],
                "best_accuracy": study.best_value,
            }
        )

    final_df = pd.DataFrame(results_list)
    final_df.to_parquet(config.mcts_params_result_path, index=False)
    logger.info(
        f"Successfully saved optimized MCTS parameters to {config.mcts_params_result_path}"
    )
