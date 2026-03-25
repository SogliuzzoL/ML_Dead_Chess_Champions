import logging

import numpy as np

from core.config import ProjectConfig
from core.pca import StylePCA

logger = logging.getLogger(__name__)


def train_pca(config: ProjectConfig, input_path: str, output_path: str):
    pca_vectors = np.load(input_path, mmap_mode="r")
    logger.info(f"Loaded PCA vectors from {input_path} with shape {pca_vectors.shape}")

    logger.info("Training PCA model...")
    model = StylePCA(n_components=128)
    result = model.fit_transform(pca_vectors)

    logger.info(
        f"Saving PCA model to {config.pca_model_path} and transformed vectors to {output_path}"
    )
    model.save_model(config.pca_model_path)
    np.save(output_path, result)


def infer_pca(config: ProjectConfig, input_path: str, output_path: str):
    pca_vectors = np.load(input_path, mmap_mode="r")
    logger.info(f"Loading pre-trained PCA model from {config.pca_model_path}")

    model = StylePCA(n_components=128).load_model(config.pca_model_path)
    result = model.transform(pca_vectors)

    np.save(output_path, result)
