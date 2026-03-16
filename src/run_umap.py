import argparse

from core.config import (
    PLAYER_REFERENCE,
    TEST_MAIA_EMBEDDINGS_PATH,
    TEST_SET_PATH,
    TEST_UMAP_VECTORS_PATH,
    TEST_VECTORS_PATH,
    TRAIN_MAIA_EMBEDDINGS_PATH,
    TRAIN_SET_PATH,
    TRAIN_UMAP_VECTORS_PATH,
    TRAIN_VECTORS_PATH,
)
from features.compute_vectors import compute_vectors
from features.style_extractor import extract_styles
from models.compute_distances import compute_distances
from models.train_autoencoder import infer_autoencoder, run_autoencoder
from models.train_pca import infer_pca, train_pca
from models.train_umap import infer_umap, train_umap
from visualization.visualize_distances import visualize_distances
from visualization.visualize_umap import visualize_umap

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state", type=int, default=0, help="Set to 1 to enable STATE_MODE."
    )
    parser.add_argument(
        "--train", type=int, default=0, help="Set to 1 to enable TRAIN_MODE."
    )
    parser.add_argument(
        "--compute", type=int, default=0, help="Set to 1 to enable COMPUTE_MODE."
    )
    parser.add_argument(
        "--pca", type=int, default=0, help="Set to 1 to enable PCA_MODE."
    )

    args = parser.parse_args()
    STATE_MODE = bool(args.state)
    TRAIN_MODE = bool(args.train)
    COMPUTE_MODE = bool(args.compute)
    PCA_MODE = bool(args.pca)

    if TRAIN_MODE:
        if STATE_MODE:
            if COMPUTE_MODE:
                extract_styles(TRAIN_SET_PATH, TRAIN_MAIA_EMBEDDINGS_PATH)
        else:
            if COMPUTE_MODE:
                compute_vectors(TRAIN_SET_PATH, TRAIN_VECTORS_PATH)
            if PCA_MODE:
                train_pca(TRAIN_VECTORS_PATH, TRAIN_UMAP_VECTORS_PATH)
            else:
                run_autoencoder(TRAIN_VECTORS_PATH, TRAIN_UMAP_VECTORS_PATH)

        train_umap(STATE_MODE)

    if STATE_MODE:
        if COMPUTE_MODE:
            extract_styles(TEST_SET_PATH, TEST_MAIA_EMBEDDINGS_PATH)
        infer_umap(STATE_MODE, is_test=True)
    else:
        if COMPUTE_MODE:
            compute_vectors(TEST_SET_PATH, TEST_VECTORS_PATH)
        if PCA_MODE:
            infer_pca(TEST_VECTORS_PATH, TEST_UMAP_VECTORS_PATH)
        else:
            infer_autoencoder(TEST_VECTORS_PATH, TEST_UMAP_VECTORS_PATH)

        infer_umap(STATE_MODE, is_test=True)

    compute_distances(STATE_MODE, is_test=True)
    visualize_distances(STATE_MODE, is_test=True)
    visualize_umap(PLAYER_REFERENCE, STATE_MODE, is_test=True)
