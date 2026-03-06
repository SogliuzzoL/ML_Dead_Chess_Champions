import numpy as np

from core.config import PCA_MODEL_PATH, logger
from core.pca import StylePCA


def train_pca(input_path: str, output_path: str):
    pca_vectors = np.load(input_path, mmap_mode='r')
    logger.info(
        f"Loaded PCA vectors from {input_path} with shape {pca_vectors.shape}")

    logger.info("Training PCA model...")
    model = StylePCA(n_components=128)
    result = model.fit_transform(pca_vectors)

    logger.info(
        f"Saving PCA model to {PCA_MODEL_PATH} and transformed vectors to {output_path}")
    model.save_model(PCA_MODEL_PATH)
    np.save(output_path, result)
