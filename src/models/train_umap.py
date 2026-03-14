import numpy as np
import pandas as pd

from core.config import (
    DATASET_PATH,
    TRAIN_MAIA_EMBEDDINGS_PATH,
    TRAIN_UMAP_RESULT_PATH,
    TRAIN_UMAP_STATE_RESULT_PATH,
    TRAIN_UMAP_VECTORS_PATH,
    UMAP_MODEL_PATH,
    UMAP_STATE_MODEL_PATH,
    logger,
)
from core.umap import StyleUMAP


def train_umap(state_mode=False):
    vectors_path = (
        TRAIN_UMAP_VECTORS_PATH if not state_mode else TRAIN_MAIA_EMBEDDINGS_PATH
    )
    result_path = (
        TRAIN_UMAP_RESULT_PATH if not state_mode else TRAIN_UMAP_STATE_RESULT_PATH
    )
    model_path = UMAP_MODEL_PATH if not state_mode else UMAP_STATE_MODEL_PATH

    logger.info("Loading vectors from %s", vectors_path)
    vectors = np.load(vectors_path, mmap_mode="r")

    logger.info("Performing UMAP dimensionality reduction")
    model_umap = StyleUMAP(n_components=2)
    result_umap = model_umap.fit_transform(vectors)

    logger.info("Saving UMAP results to %s", result_path)
    df = pd.read_parquet(DATASET_PATH)

    result_df = pd.DataFrame(result_umap, columns=["UMAP1", "UMAP2"])
    result_df["player_name"] = df["player_name"].values

    result_df.to_parquet(result_path, index=False)
    model_umap.save_model(model_path)
