import numpy as np

from core.config import PCA_MODEL_PATH, PCA_VECTORS_PATH, UMAP_VECTORS_PATH
from core.pca import StylePCA


def train_pca():
    maia_embeddings = np.load(PCA_VECTORS_PATH, mmap_mode='r')
    print(f"Loaded Maia embeddings with shape: {maia_embeddings.shape}")
    model = StylePCA(n_components=maia_embeddings.shape[1] // 10)
    result = model.fit_transform(maia_embeddings)
    model.save_model(PCA_MODEL_PATH)

    np.save(UMAP_VECTORS_PATH, result)
