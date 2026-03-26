import argparse

from src.core.config import Config
from src.core.utils import getLogger

logger = getLogger()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "step",
        choices=[
            "fetch",
            "build",
            "stats",
            "vectors",
            "autoencoder",
            "umap",
            "evaluate",
        ],
        help="The step of the pipeline to run",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yml",
        help="Path to the YAML configuration file (default: config/default.yml)",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="umap",
        choices=["umap", "vae", "contrastif"],
        help="The dimensionality reduction method to evaluate (default: umap)",
    )

    args = parser.parse_args()
    config = Config.from_yaml(args.config)

    if args.step == "fetch":
        from src.data.fetch_games import fetch_all_games

        logger.info("Starting data fetching step...")
        fetch_all_games(config)

    if args.step == "build":
        from src.data.build_dataset import build_dataset

        logger.info("Starting dataset building step...")
        build_dataset(config)

    if args.step == "stats":
        from src.data.opening_stats import extract_opening_stats

        logger.info("Starting opening stats extraction step...")
        extract_opening_stats(config)

    if args.step == "vectors":
        from src.features.compute_vectors import compute_vectors

        logger.info("Starting vector computation step...")
        compute_vectors(config)

    if args.step == "autoencoder":
        from src.training.train_autoencoder import run_autoencoder_pipeline

        logger.info("Starting autoencoder training step...")
        run_autoencoder_pipeline(config)

    if args.step == "umap":
        from src.training.train_umap import run_umap_pipeline

        logger.info("Starting UMAP training step...")
        run_umap_pipeline(config)

    if args.step == "evaluate":
        from src.evaluation.compute_distances import (
            compute_distances,
            compute_train_test_distances,
        )

        logger.info(f"Starting evaluation step for method: {args.method.upper()}...")
        compute_distances(config, method=args.method, is_test=False)
        compute_distances(config, method=args.method, is_test=True)
        compute_train_test_distances(config, method=args.method)


if __name__ == "__main__":
    main()
