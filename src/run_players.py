import argparse

import pandas as pd
from maia2.inference import inference_batch
from maia2.model import from_pretrained
from maia2.utils import get_all_possible_moves
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import ProjectConfig
from core.engine import MaiaEngine
from data.evaluation_dataset import EvaluationDataset
from models.train_players import run_training
from visualization.visualize_accuracies import visualize_player_accuracies

if __name__ == "__main__":
    # Command-line interface configuration
    parser = argparse.ArgumentParser(description="Player evaluation execution script.")
    parser.add_argument(
        "--train", type=int, default=0, help="Set to 1 to enable training mode."
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default="data",
        help="Target pathway for the data directory.",
    )
    parser.add_argument(
        "--result_folder",
        type=str,
        default="results",
        help="Target pathway for the results directory.",
    )
    args = parser.parse_args()

    # Dynamic instantiation of the project configuration
    config = ProjectConfig(
        data_folder=args.data_folder, result_folder=args.result_folder
    )
    config.create_directories()

    if bool(args.train):
        run_training()

    engine = MaiaEngine()
    baseline_model = from_pretrained("rapid", device=engine.device)

    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    base_elo_idx = engine._get_style_idx(2500)

    # Utilizing the injected data module with dynamic directory pathways
    test_dataset = EvaluationDataset(
        config.test_set_path, engine.player_to_idx, all_moves_dict, base_elo_idx
    )
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)

    correct_preds_custom, player_ids = engine.evaluate_batch(test_loader)

    idx_to_player = {idx: player for player, idx in engine.player_to_idx.items()}
    metrics = {
        player: {"n_positions": 0, "custom_correct": 0, "baseline_accuracy": 0.0}
        for player in config.base_player_dict.values()
    }

    for i in range(len(player_ids)):
        if player_ids[i] in idx_to_player:
            player_name = idx_to_player[player_ids[i]]
            metrics[player_name]["n_positions"] += 1
            if correct_preds_custom[i]:
                metrics[player_name]["custom_correct"] += 1

    df_full = pd.read_parquet(config.test_set_path)
    df_full["active_elo"] = 2500
    df_full["opponent_elo"] = 2500

    for player_name in tqdm(config.base_player_dict.values()):
        player_mask = df_full["player_name"] == player_name
        if not player_mask.any():
            continue

        df_player = df_full.loc[player_mask, config.maia_col_order].copy()
        _, baseline_acc = inference_batch(
            df_player, baseline_model, batch_size=512, num_workers=4, verbose=False
        )
        metrics[player_name]["baseline_accuracy"] = baseline_acc

    results = []
    for p, stats in metrics.items():
        if stats["n_positions"] > 0:
            custom_acc = stats["custom_correct"] / stats["n_positions"]
            baseline_acc = stats["baseline_accuracy"]
            results.append(
                {
                    "player": p,
                    "n_positions": stats["n_positions"],
                    "baseline_accuracy": baseline_acc,
                    "custom_accuracy": custom_acc,
                    "absolute_improvement": custom_acc - baseline_acc,
                }
            )

    df_results = pd.DataFrame(results).sort_values(
        "absolute_improvement", ascending=False
    )
    print(df_results)

    # Saving artifacts utilizing the dynamic configuration
    output_path = config.result_folder + "/player_accuracies_comparison.parquet"
    df_results.to_parquet(output_path, index=False)

    visualize_player_accuracies()
