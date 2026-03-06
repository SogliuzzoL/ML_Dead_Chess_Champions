import numpy as np
import pandas as pd
import tqdm

from core.umap import position_to_vector


def compute_vectors(input_path: str, output_path: str) -> None:
    df = pd.read_parquet(input_path)
    vectors = []
    progress_bar = tqdm.tqdm(df.iterrows(), total=len(df))
    for _, row in progress_bar:
        vectors.append(position_to_vector(row["fen"], row["move"]))

    vectors = np.array(vectors)
    np.save(output_path, vectors)
