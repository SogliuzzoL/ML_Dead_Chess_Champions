import numpy as np
import pandas as pd
import tqdm

from core.config import DATASET_PATH, VECTORS_PATH
from core.umap import position_to_vector


def compute_vectors():
    df = pd.read_parquet(DATASET_PATH)
    vectors = []
    progress_bar = tqdm.tqdm(df.iterrows(), total=len(df))
    for _, row in progress_bar:
        vectors.append(position_to_vector(row["fen"], row["move"]))

    vectors = np.array(vectors)
    np.save(VECTORS_PATH, vectors)
