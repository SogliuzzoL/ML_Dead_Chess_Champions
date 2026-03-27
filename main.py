import argparse

from src.core.config import Config
from src.core.utils import getLogger

logger = getLogger()


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate discrete stages of the ML_Dead_Chess_Champions pipeline."
    )

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
            "train_players",
            "evaluate_players",
        ],
        help="Pipeline stage to execute",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yml",
        help="Filesystem path to the YAML configuration file (default: config/default.yml)",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="umap",
        choices=["umap", "vae", "contrastif"],
        help="Dimensionality-reduction method to evaluate (default: umap)",
    )

    args = parser.parse_args()
    config = Config.from_yaml(args.config)

    if args.step == "fetch":
        from src.data.fetch_games import fetch_all_games

        logger.info("Commencing data acquisition stage...")
        fetch_all_games(config)

    if args.step == "build":
        from src.data.build_dataset import build_dataset

        logger.info("Commencing dataset construction stage...")
        build_dataset(config)

    if args.step == "stats":
        from src.data.opening_stats import extract_opening_stats

        logger.info("Commencing extraction of opening statistics...")
        extract_opening_stats(config)

    if args.step == "vectors":
        from src.features.compute_vectors import compute_vectors

        logger.info("Commencing computation of position vectors...")
        compute_vectors(config)

    if args.step == "autoencoder":
        from src.training.train_autoencoder import run_autoencoder_pipeline

        logger.info("Commencing autoencoder training routine...")
        run_autoencoder_pipeline(config)

    if args.step == "umap":
        from src.training.train_umap import run_umap_pipeline

        logger.info("Commencing UMAP training and transformation routine...")
        run_umap_pipeline(config)

    if args.step == "evaluate":
        from src.evaluation.compute_distances import (
            compute_distances,
            compute_train_test_distances,
        )

        logger.info(f"Commencing evaluation for method: {args.method.upper()}...")
        compute_distances(config, method=args.method, is_test=False)
        compute_distances(config, method=args.method, is_test=True)
        compute_train_test_distances(config, method=args.method)

    if args.step == "train_players":
        from src.training.train_players import run_training

        logger.info("Commencing player embedding training routine...")
        run_training(config)

    if args.step == "evaluate_players":
        from src.evaluation.evaluate_players import evaluate_players

        logger.info("Commencing player evaluation and comparison routine...")
        evaluate_players(config, force_train=False)


if __name__ == "__main__":
    main()
