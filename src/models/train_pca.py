import numpy as np

from core.config import PCA_MODEL_PATH, UMAP_VECTORS_PATH, VECTORS_PATH, logger
from core.pca import StylePCA


def train_pca():
    pca_vectors = np.load(VECTORS_PATH, mmap_mode='r')
    logger.info(
        f"Loaded PCA vectors from {VECTORS_PATH} with shape {pca_vectors.shape}")

    logger.info("Training PCA model...")
    model = StylePCA(n_components=128)
    result = model.fit_transform(pca_vectors)

    logger.info(
        f"Saving PCA model to {PCA_MODEL_PATH} and transformed vectors to {UMAP_VECTORS_PATH}")
    model.save_model(PCA_MODEL_PATH)
    np.save(UMAP_VECTORS_PATH, result)
