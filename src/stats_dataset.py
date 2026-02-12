import pandas as pd

from config import DATASET_PATH

if __name__ == "__main__":
    df = pd.read_parquet(DATASET_PATH)
    players = df["player_name"].unique()
    for player in players:
        player_df = df[df["player_name"] == player]
        print(f"Player: {player}")
        print(f"Total moves: {len(player_df)}")
