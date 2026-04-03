from pathlib import Path

import chess.pgn as pgn
import polars as pl
import tqdm

from src.core.config import Config
from src.core.utils import getLogger

logger = getLogger()


def extract_players_stats(config: Config) -> None:
    data = []
    raw_data_dir = Path(config.paths.raw_data)

    for player_id, player_name in config.data.players.items():
        logger.info(f"Extracting statistics for {player_name} (ID: {player_id})")

        pgn_folder = raw_data_dir / player_id

        if not pgn_folder.exists():
            logger.warning(
                f"No PGN directory found for {player_name} (ID: {player_id}). Skipping."
            )
            continue

        pgn_files = list(pgn_folder.glob("*.pgn"))
        progress_bar = tqdm.tqdm(pgn_files, desc=f"Processing {player_name}")

        n_games = len(pgn_files)
        n_plys = 0
        mean_year = 0
        n_year = 0

        for pgn_path in progress_bar:
            with open(pgn_path, encoding="utf-8") as f:
                game = pgn.read_game(f)

            if game is None:
                continue

            n_plys += int(int(game.headers.get("PlyCount", 0)) / 2)
            year = (
                int(game.headers.get("Date", "????.??.??")[:4])
                if "Date" in game.headers
                else 0
            )
            if year != 0:
                mean_year += year
                n_year += 1

        mean_year /= n_year if n_year > 0 else 1

        data.append(
            {
                "player_id": player_id,
                "player_name": player_name,
                "n_games": n_games,
                "n_plys": n_plys,
                "mean_year": int(mean_year),
            }
        )

    logger.info("Generating the Polars DataFrame...")
    df = pl.DataFrame(data)
    print(df)

    logger.info("Saving opening statistics...")
    df.write_parquet(config.paths.player_stats_path)

    logger.info("Players statistics saved successfully.")
