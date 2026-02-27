import argparse

from features.compute_vectors import compute_vectors
from features.style_extractor import extract_styles
from models.compute_distances import compute_distances
from models.train_umap import train_umap
from visualization.visualize_distances import visualize_distances
from visualization.visualize_umap import visualize_umap

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state",
        type=int,
        default=0,
        help="Définir sur 1 pour activer le STATE_MODE, ou 0 pour le désactiver."
    )

    args = parser.parse_args()
    STATE_MODE = bool(args.state)

    if STATE_MODE:
        extract_styles()
    else:
        compute_vectors()
    train_umap(STATE_MODE)
    compute_distances(STATE_MODE)
    visualize_umap(["Timman"], STATE_MODE)
    visualize_distances(STATE_MODE)
    visualize_distances(STATE_MODE)
