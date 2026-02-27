import numpy as np
import pandas as pd
import tqdm

from core.config import DATASET_PATH, UMAP_VECTORS_PATH
from core.umap import position_to_vector

if __name__ == "__main__":
    df = pd.read_parquet(DATASET_PATH)
    vectors = []
    progress_bar = tqdm.tqdm(df.iterrows(), total=len(df))
    for _, row in progress_bar:
        vectors.append(position_to_vector(row["fen"], row["move"]))

    vectors = np.array(vectors)
    np.save(UMAP_VECTORS_PATH, vectors)
