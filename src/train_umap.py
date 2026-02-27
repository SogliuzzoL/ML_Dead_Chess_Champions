import os

import numpy as np
import pandas as pd

from core.config import (
    DATASET_PATH,
    UMAP_MODEL_PATH,
    UMAP_RESULT_PATH,
    UMAP_VECTORS_PATH,
    logger,
)
from core.umap import StyleUMAP

if __name__ == "__main__":
    logger.info("Loading vectors from %s", UMAP_VECTORS_PATH)
    vectors = np.load(UMAP_VECTORS_PATH, mmap_mode="r")

    logger.info("Performing UMAP dimensionality reduction")
    model_umap = StyleUMAP(n_components=2)
    result_umap = model_umap.fit_transform(vectors)

    logger.info("Saving UMAP results to %s", UMAP_RESULT_PATH)
    df = pd.read_parquet(DATASET_PATH)

    result_df = pd.DataFrame(result_umap, columns=["UMAP1", "UMAP2"])
    result_df["player_name"] = df["player_name"].values

    result_df.to_parquet(UMAP_RESULT_PATH, index=False)
    model_umap.save_model(UMAP_MODEL_PATH)
