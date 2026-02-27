import pandas as pd
import torch
from maia2 import inference, model

from core.config import DATASET_PATH, MAIA_COL_ORDER, MAIA_RESULT_PATH, logger

if __name__ == "__main__":
    data = []
    df = pd.read_parquet(DATASET_PATH)
    logger.info("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rapid_model = model.from_pretrained("rapid", device)

    df["active_elo"] = 2500
    df["opponent_elo"] = 2500

    for player in df["player_name"].unique():
        player_mask = df["player_name"] == player

        df_ready = df.loc[player_mask, MAIA_COL_ORDER].copy()

        logger.info(
            f"Running inference for player {player} with {len(df_ready)} samples.")

        df_result, accuracy = inference.inference_batch(
            df_ready,
            rapid_model,
            verbose=True,
            batch_size=128,
            num_workers=4
        )

        logger.info(
            f"Inference complete for player {player}. Accuracy: {accuracy:.2%}")

        data.append({
            "player": player,
            "maia_accuracy": accuracy
        })

    df_result = pd.DataFrame(data)
    df_result.to_parquet(MAIA_RESULT_PATH)
