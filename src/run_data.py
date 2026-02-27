from data.build_dataset import build_dataset
from data.fetch_games import fetch_all_games
from visualization.stats_dataset import plot_games_distributions

if __name__ == "__main__":
    fetch_all_games()
    build_dataset()
    plot_games_distributions()
