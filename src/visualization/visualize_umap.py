import pandas as pd

from core.config import (
    RESULT_FOLDER,
    TEST_UMAP_RESULT_PATH,
    TEST_UMAP_STATE_RESULT_PATH,
    TRAIN_UMAP_RESULT_PATH,
    TRAIN_UMAP_STATE_RESULT_PATH,
    logger,
)
from visualization.utils_plot import plot_player_centroids, plot_style_comparison


def visualize_umap(player_list, state_mode=False, is_test=False):
    if is_test:
        result_path = (
            TEST_UMAP_STATE_RESULT_PATH if state_mode else TEST_UMAP_RESULT_PATH
        )
    else:
        result_path = (
            TRAIN_UMAP_STATE_RESULT_PATH if state_mode else TRAIN_UMAP_RESULT_PATH
        )

    suffix = "_test" if is_test else "_train"
    state_suffix = "_state" if state_mode else ""

    centroids_path = f"{RESULT_FOLDER}/centroids_map{state_suffix}{suffix}.pdf"
    style_comparison_path = (
        f"{RESULT_FOLDER}/style_comparison{state_suffix}{suffix}.pdf"
    )

    logger.info("Loading UMAP results...")
    df = pd.read_parquet(result_path)

    logger.info("Plotting player centroids and style comparisons...")
    plot_player_centroids(df, output_filename=centroids_path)
    plot_style_comparison(df, player_list, output_filename=style_comparison_path)
