import polars as pl

df = pl.read_parquet("data/processed/chess_champion_distances.parquet")

print(df.head())
