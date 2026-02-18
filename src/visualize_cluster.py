import pandas as pd

from core.config import DATA_FOLDER, UMAP_RESULT_PATH, logger
from visualization.utils_plot import plot_player_centroids, plot_style_comparison

if __name__ == "__main__":
    df = pd.read_parquet(UMAP_RESULT_PATH)
    plot_player_centroids(df,
                          output_filename=f"{DATA_FOLDER}/centroids_map.pdf")
    plot_style_comparison(df,
                          ["Kasparov", "Timman"],
                          output_filename=f"{DATA_FOLDER}/style_comparison.pdf")
