import pandas as pd

from core.config import DATA_FOLDER, UMAP_RESULT_PATH, UMAP_STATE_RESULT_PATH, logger
from visualization.utils_plot import plot_player_centroids, plot_style_comparison


def visualize_umap(player_list, state_mode=False):
    result_path = UMAP_STATE_RESULT_PATH if state_mode else UMAP_RESULT_PATH
    centroids_path = f"{DATA_FOLDER}/centroids_map.pdf" if not state_mode else f"{DATA_FOLDER}/centroids_map_state.pdf"
    style_comparison_path = f"{DATA_FOLDER}/style_comparison.pdf" if not state_mode else f"{DATA_FOLDER}/style_comparison_state.pdf"

    logger.info("Loading UMAP results...")
    df = pd.read_parquet(result_path)

    logger.info("Plotting player centroids and style comparisons...")
    plot_player_centroids(df, output_filename=centroids_path)
    plot_style_comparison(
        df, player_list, output_filename=style_comparison_path)
