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


def generate_ae_latex_table(config: Config) -> None:
    """Génère le tableau LaTeX détaillant l'architecture et l'entraînement de l'AutoEncoder."""

    # Extraction des paramètres
    lr = config.autoencoder.learning_rate
    batch_size = config.autoencoder.batch_size
    epochs = config.autoencoder.epochs
    latent_dim = config.autoencoder.latent_dim

    out_path = config.paths.ae_table_latex_path

    # Utilisation d'une notation compacte pour les couches
    latex_template = f"""\\begin{{table}}[!t]
                         \\renewcommand{{\\arraystretch}}{{1.3}}
                         \\caption{{Architecture and training parameters of the AutoEncoder}}
                         \\label{{tab:ae}}
                         \\centering
                         \\begin{{tabular}}{{@{{}}l l@{{}}}}
                         \\hline
                         \\bfseries Parameter & \\bfseries Value \\\\
                         \\hline\\hline
                         Learning Rate & {lr} \\\\
                         Batch Size & {batch_size} \\\\
                         Epochs & {epochs} \\\\
                         Latent Dimension & {latent_dim} \\\\
                         \\hline
                         Encoder layers & $2304 \\rightarrow 1024 \\rightarrow 512 \\rightarrow 256 \\rightarrow {latent_dim}$ \\\\
                         Decoder layers & ${latent_dim} \\rightarrow 256 \\rightarrow 512 \\rightarrow 1024 \\rightarrow 2304$ \\\\
                         Hidden Activations & ReLU \\\\
                         Output Activation & Sigmoid \\\\
                         \\hline
                         \\end{{tabular}}
                         \\end{{table}}"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex_template)

    logger.info(f"AutoEncoder LaTeX table generated and saved to {out_path}")


def generate_all_tables(config: Config) -> None:
    """Generate all LaTeX tables for the paper."""
    generate_ae_latex_table(config)
    generate_latex_table(config)
