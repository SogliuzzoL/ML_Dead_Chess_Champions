import pandas as pd

from core.config import DATA_FOLDER
from visualization.utils_plot import plot_distance_heatmap

if __name__ == "__main__":
    distance_df = pd.read_parquet(f"{DATA_FOLDER}/distances.parquet")
    plot_distance_heatmap(
        df=distance_df,
        title="Jensen,Shannon Distance Matrix Between Champions",
        output_filename=f"{DATA_FOLDER}/heatmap_js_distances.pdf"
    )
