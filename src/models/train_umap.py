import numpy as np
import pandas as pd

from core.config import (
    TEST_MAIA_EMBEDDINGS_PATH,
    TEST_SET_PATH,
    TEST_UMAP_RESULT_PATH,
    TEST_UMAP_STATE_RESULT_PATH,
    TEST_UMAP_VECTORS_PATH,
    TRAIN_MAIA_EMBEDDINGS_PATH,
    TRAIN_SET_PATH,
    TRAIN_UMAP_RESULT_PATH,
    TRAIN_UMAP_STATE_RESULT_PATH,
    TRAIN_UMAP_VECTORS_PATH,
    UMAP_MODEL_PATH,
    UMAP_STATE_MODEL_PATH,
    logger,
)
from core.umap import StyleUMAP


def train_umap(state_mode=False):
    vectors_path = TRAIN_MAIA_EMBEDDINGS_PATH if state_mode else TRAIN_UMAP_VECTORS_PATH
    result_path = TRAIN_UMAP_STATE_RESULT_PATH if state_mode else TRAIN_UMAP_RESULT_PATH
    model_path = UMAP_STATE_MODEL_PATH if state_mode else UMAP_MODEL_PATH

    logger.info("Loading training vectors from %s", vectors_path)
    vectors = np.load(vectors_path, mmap_mode="r")

    logger.info("Performing UMAP dimensionality reduction fitting")
    model_umap = StyleUMAP(n_components=2)
    result_umap = model_umap.fit_transform(vectors)

    df = pd.read_parquet(TRAIN_SET_PATH)
    result_df = pd.DataFrame(result_umap, columns=["UMAP1", "UMAP2"])
    result_df["player_name"] = df["player_name"].values

    logger.info("Saving UMAP training results to %s", result_path)
    result_df.to_parquet(result_path, index=False)
    model_umap.save_model(model_path)


def infer_umap(state_mode=False, is_test=True):
    vectors_path = TEST_MAIA_EMBEDDINGS_PATH if state_mode else TEST_UMAP_VECTORS_PATH
    result_path = TEST_UMAP_STATE_RESULT_PATH if state_mode else TEST_UMAP_RESULT_PATH
    dataset_path = TEST_SET_PATH
    model_path = UMAP_STATE_MODEL_PATH if state_mode else UMAP_MODEL_PATH

    logger.info("Loading inference vectors from %s", vectors_path)
    vectors = np.load(vectors_path, mmap_mode="r")

    logger.info("Loading pre-trained UMAP model from %s", model_path)
    model_umap = StyleUMAP(n_components=2).load_model(model_path)

    logger.info("Applying UMAP transformation to unseen data")
    result_umap = model_umap.transform(vectors)

    df = pd.read_parquet(dataset_path)
    result_df = pd.DataFrame(result_umap, columns=["UMAP1", "UMAP2"])
    result_df["player_name"] = df["player_name"].values

    result_df.to_parquet(result_path, index=False)
