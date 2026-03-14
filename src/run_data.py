import argparse

from data.build_dataset import build_dataset
from data.fetch_games import fetch_all_games
from visualization.stats_dataset import plot_games_distributions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download",
        type=int,
        default=0,
        help="Set to 1 to enable DOWNLOAD_MODE, or 0 to disable it.",
    )

    args = parser.parse_args()
    DOWNLOAD_MODE = bool(args.download)

    if DOWNLOAD_MODE:
        fetch_all_games()

    build_dataset()
    plot_games_distributions()
