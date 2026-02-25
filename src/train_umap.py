import numpy as np
import pandas as pd

from core.cluster import StyleUMAP
from core.config import (
    DATASET_PATH,
    MAIA_EMBEDDINGS_PATH,
    UMAP_MODEL_PATH,
    UMAP_RESULT_PATH,
    logger,
)

if __name__ == "__main__":
    logger.info("Loading embeddings and dataset...")
    embeddings = np.load(MAIA_EMBEDDINGS_PATH, mmap_mode="r")
    df = pd.read_parquet(DATASET_PATH)

    logger.info("Training UMAP model...")
    model = StyleUMAP(n_components=2, n_neighbors=80, verbose=True)
    result = model.fit_transform(embeddings)

    logger.info("Saving UMAP results...")
    result_df = pd.DataFrame(result, columns=["UMAP1", "UMAP2"])
    result_df["player_name"] = df["player_name"].values
    result_df.to_parquet(UMAP_RESULT_PATH, index=False)
    model.save_model(UMAP_MODEL_PATH)

    logger.info("Saving UMAP results...")
    result_df = pd.DataFrame(result, columns=["UMAP1", "UMAP2"])
    result_df["player_name"] = df["player_name"].values
    result_df.to_parquet(UMAP_RESULT_PATH, index=False)
    model.save_model(UMAP_MODEL_PATH)
