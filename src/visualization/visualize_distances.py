import os

import pandas as pd

from core.config import ProjectConfig
from visualization.utils_plot import plot_bar_distribution, plot_distance_heatmap


def visualize_distances(config: ProjectConfig, state_mode=False, is_test=False):
    """
    Renders the Jensen-Shannon distance matrix utilizing dynamically configured output pathways.
    """
    result_path = (
        config.distances_test_result_path
        if is_test
        else config.distances_train_result_path
    )
    suffix = "_test" if is_test else "_train"
    state_suffix = "_state" if state_mode else ""

    output_filename = os.path.join(
        config.result_folder, f"heatmap_js_distances{state_suffix}{suffix}.pdf"
    )

    distance_df = pd.read_parquet(result_path)
    plot_distance_heatmap(
        df=distance_df,
        title="Jensen-Shannon Distance Matrix Between Champions",
        output_filename=output_filename,
    )


def visualize_train_test_distances(config: ProjectConfig, state_mode=False):
    """
    Visualizes the intra-player divergence between training and testing sets.
    """
    state_suffix = "_state" if state_mode else ""
    output_filename = os.path.join(
        config.result_folder, f"bar_js_distances_train_test{state_suffix}.pdf"
    )

    distance_df = pd.read_parquet(config.cross_distances_train_test_result_path)

    # Transform the DataFrame into a pd.Series to interface with the plotting utility
    series_data = distance_df.set_index("Player")["JSDistance_Train_Test"]

    plot_bar_distribution(
        data=series_data,
        title="Train vs Test Jensen-Shannon Divergence",
        output_filename=output_filename,
    )
