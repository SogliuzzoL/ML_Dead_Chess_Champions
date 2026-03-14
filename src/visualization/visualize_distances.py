import pandas as pd

from core.config import (
    DISTANCES_RESULT_PATH,
    DISTANCES_STATE_RESULT_PATH,
    RESULT_FOLDER,
)
from visualization.utils_plot import plot_distance_heatmap


def visualize_distances(state_mode=False):
    result_path = DISTANCES_STATE_RESULT_PATH if state_mode else DISTANCES_RESULT_PATH
    output_filename = (
        f"{RESULT_FOLDER}/heatmap_js_distances_state.pdf"
        if state_mode
        else f"{RESULT_FOLDER}/heatmap_js_distances.pdf"
    )

    distance_df = pd.read_parquet(result_path)
    plot_distance_heatmap(
        df=distance_df,
        title="Jensen,Shannon Distance Matrix Between Champions",
        output_filename=output_filename,
    )
