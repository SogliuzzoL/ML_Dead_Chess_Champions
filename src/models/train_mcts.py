import os

import optuna
import pandas as pd
from chess.engine import SimpleEngine
from tqdm import tqdm

from core.config import (
    MCTS_OPTIMIZATION_DB_PATH,
    RESULT_FOLDER,
    STOCKFISH_MODEL_PATH,
    TRAIN_SET_PATH,
    base_player_dict,
    logger,
)
from core.engine import MaiaEngine

stockfish = SimpleEngine.popen_uci(STOCKFISH_MODEL_PATH)


def create_objective(df_player, player_name):
    def objective(trial):
        engine = MaiaEngine()
        c_puct = trial.suggest_float("c_puct", 0.1, 10.0, log=True)
        scale = trial.suggest_float("scale", 100.0, 1000.0, log=True)
        threshold = trial.suggest_float("threshold", 0.001, 0.1, log=True)
        num_simulations = trial.suggest_int(
            "num_simulations", 30, 500, log=True)

        correct_predictions = 0

        for _, row in df_player.iterrows():
            best_move, _ = engine.predict_mcts(
                fen=row["fen"],
                pgn="",
                stockfish=stockfish,
                num_simulations=num_simulations,
                threshold=threshold,
                active_elo=player_name,
                c_puct=c_puct,
                scale=scale
            )
            if best_move == row["move"]:
                correct_predictions += 1

        return correct_predictions / len(df_player)

    return objective


def train_all_players():
    logger.info(
        "Initiating comprehensive optimization protocol across all designated subjects.")

    df_full = pd.read_parquet(TRAIN_SET_PATH)
    results_list = []

    # optuna.logging.set_verbosity(optuna.logging.WARNING)

    for player_id, player_name in tqdm(base_player_dict.items(), desc="Global Optimization", position=0):
        logger.info(
            f"Commencing Optuna hyperparameter study for entity: {player_name}.")

        df_player = df_full[df_full["player_name"] == player_name]

        if df_player.empty:
            logger.warning(
                f"Insufficient empirical data retrieved for {player_name}, bypassing optimization sequence."
            )
            continue

        df_player = df_player.sample(n=100)

        study = optuna.create_study(
            study_name=f"mcts_optim_{player_name}",
            storage=f"sqlite:///{MCTS_OPTIMIZATION_DB_PATH}",
            direction="maximize",
            load_if_exists=True
        )

        objective_function = create_objective(df_player, player_name)

        with tqdm(total=50, desc=f"Trials {player_name}", leave=False, position=1) as pbar:
            def optuna_callback(study, trial):
                pbar.update(1)

            study.optimize(objective_function, n_trials=50,
                           n_jobs=1, callbacks=[optuna_callback])

        best_params = study.best_params

        results_list.append({
            "player_id": player_id,
            "player_name": player_name,
            "num_simulations": best_params["num_simulations"],
            "c_puct": best_params["c_puct"],
            "scale": best_params["scale"],
            "threshold": best_params["threshold"],
            "best_accuracy": study.best_value
        })

    final_df = pd.DataFrame(results_list)
    output_path = os.path.join(RESULT_FOLDER, "players_mcts_params.parquet")
    final_df.to_parquet(output_path, index=False)
    stockfish.quit()

    logger.info(
        f"Optimization framework concluded successfully, resulting parameters systematically exported to {output_path}."
    )
