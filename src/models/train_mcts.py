import os

import optuna
import pandas as pd

from core.config import RESULT_FOLDER, TRAIN_SET_PATH, base_player_dict, logger
from core.engine import MaiaEngine


def create_objective(df_player, player_name):
    def objective(trial):
        engine = MaiaEngine()
        c_puct = trial.suggest_float("c_puct", 0.5, 4.0)
        scale = trial.suggest_float("scale", 100.0, 1000.0)
        threshold = trial.suggest_float("threshold", 0.001, 0.05, log=True)
        num_simulations = trial.suggest_int("num_simulations", 10, 1000)
        max_depth = trial.suggest_int("max_depth", 1, 10)

        correct_predictions = 0

        for _, row in df_player.iterrows():
            best_move, _ = engine.predict_mcts(
                fen=row["fen"],
                pgn="",
                num_simulations=num_simulations,
                max_depth=max_depth,
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
    logger.info("Démarrage de l'optimisation globale pour tous les champions")

    df_full = pd.read_parquet(TRAIN_SET_PATH)

    results_list = []

    for player_id, player_name in base_player_dict.items():
        logger.info(f"Lancement de l'étude Optuna pour : {player_name}")

        df_player = df_full[df_full["player_name"] == player_name]

        if df_player.empty:
            logger.warning(
                f"Aucune donnée trouvée pour {player_name}, saut de l'optimisation.")
            continue
        df_player = df_player.sample(n=10, random_state=42)

        study = optuna.create_study(direction="maximize")
        objective_function = create_objective(df_player, player_name)

        study.optimize(objective_function, n_trials=5)

        best_params = study.best_params

        results_list.append({
            "player_id": player_id,
            "player_name": player_name,
            "num_simulations": best_params["num_simulations"],
            "max_depth": best_params["max_depth"],
            "c_puct": best_params["c_puct"],
            "scale": best_params["scale"],
            "threshold": best_params["threshold"],
            "best_accuracy": study.best_value
        })

    final_df = pd.DataFrame(results_list)
    output_path = os.path.join(RESULT_FOLDER, "players_mcts_params.parquet")
    final_df.to_parquet(output_path, index=False)

    logger.info(
        f"Optimisation terminée, tous les paramètres sont disponibles dans {output_path}")
