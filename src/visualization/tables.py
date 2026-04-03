import polars as pl

from src.core.config import Config
from src.core.utils import getLogger

logger = getLogger()


def generate_latex_table(config: Config) -> None:
    df = pl.read_parquet(config.paths.player_stats_path)

    df_sorted = df.select(
        [pl.col("player_name"), pl.col("mean_year"), pl.col("n_games")]
    ).sort("player_name")

    latex_rows = []
    for row in df_sorted.iter_rows():
        name, year, games = row
        latex_rows.append(f"{name} & {year} & {games} \\\\")

    table_body = "\n".join(latex_rows)

    latex_template = f"""\\begin{{table}}[!t]
                         \\renewcommand{{\\arraystretch}}{{1.3}}
                         \\caption{{Overview of selected chess champions}}
                         \\label{{tab:dataset}}
                         \\centering
                         \\begin{{tabular}}{{l c c}}
                         \\hline
                         \\bfseries Player & \\bfseries Average game's year & \\bfseries Games \\\\
                         \\hline\\hline
                         {table_body}
                         \\hline
                         \\end{{tabular}}
                         \\end{{table}}"""
    with open(config.paths.table_latex_path, "w", encoding="utf-8") as f:
        f.write(latex_template)

    logger.info(f"LaTeX table generated and saved to {config.paths.table_latex_path}")
