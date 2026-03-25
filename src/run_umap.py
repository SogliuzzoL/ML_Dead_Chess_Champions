import argparse

from core.config import ProjectConfig
from features.compute_vectors import compute_vectors
from features.style_extractor import extract_styles
from models.compute_distances import compute_distances, compute_train_test_distances
from models.train_autoencoder import infer_autoencoder, run_autoencoder
from models.train_pca import infer_pca, train_pca
from models.train_umap import infer_umap, train_umap
from visualization.visualize_distances import (
    visualize_distances,
    visualize_train_test_distances,
)
from visualization.visualize_umap import visualize_umap

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UMAP dimensionality reduction pipeline orchestrator."
    )
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
    parser.add_argument(
        "--data_folder",
        type=str,
        default="data",
        help="Target pathway for the data directory.",
    )
    parser.add_argument(
        "--result_folder",
        type=str,
        default="results",
        help="Target pathway for the results directory.",
    )

    args = parser.parse_args()
    STATE_MODE = bool(args.state)
    TRAIN_MODE = bool(args.train)
    COMPUTE_MODE = bool(args.compute)
    PCA_MODE = bool(args.pca)

    # Dynamic instantiation of the project configuration
    config = ProjectConfig(
        data_folder=args.data_folder, result_folder=args.result_folder
    )
    config.create_directories()

    if TRAIN_MODE:
        if STATE_MODE:
            if COMPUTE_MODE:
                extract_styles(config)
        else:
            if COMPUTE_MODE:
                compute_vectors(config)
            if PCA_MODE:
                train_pca(config)
            else:
                run_autoencoder(config)

        train_umap(config, state_mode=STATE_MODE)

    if STATE_MODE:
        if COMPUTE_MODE:
            extract_styles(config)
        infer_umap(config, state_mode=STATE_MODE, is_test=True)
    else:
        if COMPUTE_MODE:
            compute_vectors(config)
        if PCA_MODE:
            infer_pca(config)
        else:
            infer_autoencoder(config)

        infer_umap(config, state_mode=STATE_MODE, is_test=True)

    compute_distances(config, state_mode=STATE_MODE, is_test=True)
    visualize_distances(config, state_mode=STATE_MODE, is_test=True)
    visualize_umap(config, state_mode=STATE_MODE, is_test=True)

    compute_train_test_distances(config, state_mode=STATE_MODE)
    visualize_train_test_distances(config, state_mode=STATE_MODE)
