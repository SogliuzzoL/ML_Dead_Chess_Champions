import argparse

from core.config import logger
from data.oppening_stats import extract_opening_stats
from visualization.visualize_openings import run_individual_profiles

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extract",
        type=int,
        default=0,
        help="Set to 1 to enable EXTRACT_MODE, or 0 to disable it.",
    )

    args = parser.parse_args()
    EXTRACT_MODE = bool(args.extract)

    if EXTRACT_MODE:
        logger.info("Extracting opening statistics from PGN files...")
        extract_opening_stats()
        logger.info("Opening statistics extraction completed.")

    logger.info("Running opening visualizations...")
    run_individual_profiles()
    logger.info("Opening visualizations completed.")
