import pandas as pd

from core.config import (
    CROSS_DISTANCES_TRAIN_TEST_RESULT_PATH,
    DISTANCES_TEST_RESULT_PATH,
    DISTANCES_TRAIN_RESULT_PATH,
    RESULT_FOLDER,
)
from visualization.utils_plot import plot_bar_distribution, plot_distance_heatmap


def visualize_distances(state_mode=False, is_test=False):
    result_path = DISTANCES_TEST_RESULT_PATH if is_test else DISTANCES_TRAIN_RESULT_PATH
    suffix = "_test" if is_test else "_train"
    state_suffix = "_state" if state_mode else ""

    output_filename = f"{RESULT_FOLDER}/heatmap_js_distances{state_suffix}{suffix}.pdf"

    distance_df = pd.read_parquet(result_path)
    plot_distance_heatmap(
        df=distance_df,
        title="Jensen,Shannon Distance Matrix Between Champions",
        output_filename=output_filename,
    )


def visualize_train_test_distances(state_mode=False):
    state_suffix = "_state" if state_mode else ""
    output_filename = f"{RESULT_FOLDER}/bar_js_distances_train_test{state_suffix}.pdf"

    distance_df = pd.read_parquet(CROSS_DISTANCES_TRAIN_TEST_RESULT_PATH)

    # Transform the DataFrame into a pd.Series to interface with the plotting utility
    series_data = distance_df.set_index("Player")["JSDistance_Train_Test"]

    plot_bar_distribution(
        data=series_data,
        title="Stylistic Divergence: Training vs. Testing Sets",
        xlabel="Historical Champions",
        ylabel="Jensen, Shannon Distance",
        output_filename=output_filename,
        color="#2E86C1",
    )
