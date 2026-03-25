import logging

import numpy as np
import pandas as pd

from core.config import ProjectConfig
from core.umap import StyleUMAP

logger = logging.getLogger(__name__)


def train_umap(config: ProjectConfig, state_mode=False):
    """
    Fits a UMAP dimensionality reduction model using the dynamically configured data pathways.
    """
    vectors_path = (
        config.train_maia_embeddings_path
        if state_mode
        else config.train_umap_vectors_path
    )
    result_path = (
        config.train_umap_state_result_path
        if state_mode
        else config.train_umap_result_path
    )
    model_path = config.umap_state_model_path if state_mode else config.umap_model_path

    logger.info("Loading training vectors from %s", vectors_path)
    vectors = np.load(vectors_path, mmap_mode="r")

    logger.info("Performing UMAP dimensionality reduction fitting")
    model_umap = StyleUMAP(n_components=2)
    result_umap = model_umap.fit_transform(vectors)

    df = pd.read_parquet(config.train_set_path)
    result_df = pd.DataFrame(result_umap, columns=["UMAP1", "UMAP2"])
    result_df["player_name"] = df["player_name"].values

    logger.info("Saving UMAP training results to %s", result_path)
    result_df.to_parquet(result_path, index=False)
    model_umap.save_model(model_path)


def infer_umap(config: ProjectConfig, state_mode=False, is_test=True):
    """
    Applies a pre-trained UMAP transformation to the test dataset leveraging the configuration object.
    """
    vectors_path = (
        config.test_maia_embeddings_path
        if state_mode
        else config.test_umap_vectors_path
    )
    result_path = (
        config.test_umap_state_result_path
        if state_mode
        else config.test_umap_result_path
    )
    dataset_path = config.test_set_path
    model_path = config.umap_state_model_path if state_mode else config.umap_model_path

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
