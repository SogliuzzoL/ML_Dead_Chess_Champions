import logging
import os

import pandas as pd

from core.config import ProjectConfig
from visualization.utils_plot import plot_player_centroids, plot_style_comparison

logger = logging.getLogger(__name__)


def visualize_umap(
    config: ProjectConfig, player_list: list, state_mode=False, is_test=False
):
    """
    Generates academic scatter plots of the UMAP latent space using injected configuration pathways.
    """
    if is_test:
        result_path = (
            config.test_umap_state_result_path
            if state_mode
            else config.test_umap_result_path
        )
    else:
        result_path = (
            config.train_umap_state_result_path
            if state_mode
            else config.train_umap_result_path
        )

    suffix = "_test" if is_test else "_train"
    state_suffix = "_state" if state_mode else ""

    centroids_path = os.path.join(
        config.result_folder, f"centroids_map{state_suffix}{suffix}.pdf"
    )
    style_comparison_path = os.path.join(
        config.result_folder, f"style_comparison{state_suffix}{suffix}.pdf"
    )

    logger.info(f"Loading UMAP results from {result_path}")
    df = pd.read_parquet(result_path)

    logger.info("Plotting player centroids and style comparisons...")
    plot_player_centroids(df, output_filename=centroids_path)
    plot_style_comparison(df, player_list, output_filename=style_comparison_path)
