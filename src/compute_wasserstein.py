import pandas as pd
from scipy.stats import wasserstein_distance_nd

from core.config import DATA_FOLDER, UMAP_RESULT_PATH, logger

if __name__ == "__main__":
    logger.info("Loading UMAP results...")
    df = pd.read_parquet(UMAP_RESULT_PATH)

    logger.info("Calculating Wasserstein distances between players...")
    distance_df = []
    for player1 in df["player_name"].unique():
        for player2 in df["player_name"].unique():
            if player1 < player2:
                logger.info(
                    f"Calculating distance between {player1} and {player2}...")
                embeddings1 = df[df["player_name"]
                                 == player1][["UMAP1", "UMAP2"]]
                embeddings2 = df[df["player_name"]
                                 == player2][["UMAP1", "UMAP2"]]
                distance = wasserstein_distance_nd(embeddings1, embeddings2)
                distance_df.append(
                    {"Player1": player1, "Player2": player2, "WassersteinDistance": distance})

    logger.info("Saving Wasserstein distances...")
    distance_df = pd.DataFrame(distance_df)
    distance_df.to_parquet(
        f"{DATA_FOLDER}/wasserstein_distances.parquet", index=False)
