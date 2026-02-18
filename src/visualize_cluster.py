import pandas as pd
from scipy.stats import wasserstein_distance_nd

from core.config import DATA_FOLDER, UMAP_RESULT_PATH, logger
from visualization.utils_plot import plot_player_centroids, plot_style_comparison

if __name__ == "__main__":
    df = pd.read_parquet(UMAP_RESULT_PATH)
    plot_player_centroids(df,
                          output_filename=f"{DATA_FOLDER}/centroids_map.pdf")
    plot_style_comparison(df,
                          ["Kasparov", "Timman", "Morphy"],
                          output_filename=f"{DATA_FOLDER}/style_comparison.pdf")

    kasparov_embeddings = df[df["player_name"]
                             == "Kasparov"][["UMAP1", "UMAP2"]].values
    timman_embeddings = df[df["player_name"]
                           == "Timman"][["UMAP1", "UMAP2"]].values
    morphy_embeddings = df[df["player_name"]
                           == "Morphy"][["UMAP1", "UMAP2"]].values

    kasparov_embeddings = kasparov_embeddings[:100]
    timman_embeddings = timman_embeddings[:100]
    morphy_embeddings = morphy_embeddings[:100]

    distance = wasserstein_distance_nd(kasparov_embeddings, timman_embeddings)
    logger.info(
        "Wasserstein distance between Kasparov and Timman: %.4f", distance)
    distance = wasserstein_distance_nd(kasparov_embeddings, morphy_embeddings)
    logger.info(
        "Wasserstein distance between Kasparov and Morphy: %.4f", distance)
    distance = wasserstein_distance_nd(timman_embeddings, morphy_embeddings)
    logger.info(
        "Wasserstein distance between Timman and Morphy: %.4f", distance)
