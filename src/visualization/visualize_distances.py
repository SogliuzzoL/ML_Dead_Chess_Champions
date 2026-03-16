import pandas as pd

from core.config import (
    DISTANCES_TEST_RESULT_PATH,
    DISTANCES_TRAIN_RESULT_PATH,
    RESULT_FOLDER,
)
from visualization.utils_plot import plot_distance_heatmap


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
