from pathlib import Path

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


def generate_accuracy_latex_table(config: Config) -> None:
    """Génère le tableau LaTeX des précisions (move-accuracy) par joueur et par modèle."""

    # Chargement des prédictions générées par evaluate_players.py
    df = pl.read_parquet(config.paths.predictions_path)

    # Calcul de la précision (move-accuracy) par joueur
    df_acc = (
        df.group_by("player_name")
        .agg(
            [
                (pl.col("true_move") == pl.col("pred_baseline"))
                .mean()
                .alias("acc_baseline"),
                (pl.col("true_move") == pl.col("pred_custom"))
                .mean()
                .alias("acc_custom"),
                (pl.col("true_move") == pl.col("pred_mcts")).mean().alias("acc_mcts"),
            ]
        )
        .sort("player_name")
    )

    # Calcul de la précision globale (moyenne totale) pour la dernière ligne
    overall_baseline = df.select(
        (pl.col("true_move") == pl.col("pred_baseline")).mean()
    ).item()
    overall_custom = df.select(
        (pl.col("true_move") == pl.col("pred_custom")).mean()
    ).item()
    overall_mcts = df.select((pl.col("true_move") == pl.col("pred_mcts")).mean()).item()

    # Formatage des lignes du tableau
    latex_rows = []
    for row in df_acc.iter_rows(named=True):
        name = row["player_name"]
        acc_b = f"{row['acc_baseline'] * 100:.1f}\\%"
        acc_c = f"{row['acc_custom'] * 100:.1f}\\%"
        acc_m = f"{row['acc_mcts'] * 100:.1f}\\%"
        latex_rows.append(
            f"                         {name} & {acc_b} & {acc_c} & {acc_m} \\\\"
        )

    # Ajout de la ligne avec les moyennes globales
    latex_rows.append("                         \\hline")
    latex_rows.append(
        f"                         \\bfseries Average & \\bfseries {overall_baseline * 100:.1f}\\% & \\bfseries {overall_custom * 100:.1f}\\% & \\bfseries {overall_mcts * 100:.1f}\\% \\\\"
    )

    table_body = "\n".join(latex_rows)

    latex_template = f"""\\begin{{table}}[!t]
                         \\renewcommand{{\\arraystretch}}{{1.3}}
                         \\caption{{Move-accuracy of the different models on the test set}}
                         \\label{{tab:move_accuracy}}
                         \\centering
                         \\begin{{tabular}}{{l c c c}}
                         \\hline
                         \\bfseries Player & \\bfseries Maia-2 & \\bfseries Maia-2 FT & \\bfseries Maia-2 FT + MCTS \\\\
                         \\hline\\hline
{table_body}
                         \\hline
                         \\end{{tabular}}
                         \\end{{table}}"""

    # Enregistrement. Utilise un chemin sécurisé si accuracy_table_latex_path n'est pas dans config.yaml
    out_path = getattr(
        config.paths,
        "accuracy_table_latex_path",
        str(Path(config.paths.table_latex_path).parent / "accuracy_table.tex"),
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex_template)

    logger.info(f"Accuracy LaTeX table generated and saved to {out_path}")


def generate_training_hyperparameters_latex_table(config: Config) -> None:
    """Génère le tableau LaTeX détaillant les hyperparamètres d'entraînement des embeddings."""

    # Extraction des paramètres depuis config.training (avec des valeurs par défaut basées sur ton papier)
    optimizer = getattr(config.player_training, "optimizer", "Adam")
    lr = getattr(config.player_training, "learning_rate", 1e-4)
    batch_size = getattr(config.player_training, "batch_size", 512)
    epochs = getattr(config.player_training, "epochs", 10)
    loss = getattr(config.player_training, "loss_function", "Cross-Entropy")

    # Formatage spécifique pour le Learning Rate en notation scientifique LaTeX (ex: 1 \times 10^{-4})
    if isinstance(lr, (float, int)) and lr < 0.01:
        # Transforme 0.0001 ou 1e-4 en notation scientifique propre pour LaTeX
        base, exp = f"{lr:.0e}".split("e")
        lr_str = f"${base} \\times 10^{{{int(exp)}}}$"
    else:
        lr_str = str(lr)

    # Chemin de sortie (sécurisé si hyperparameters_table_latex_path n'est pas dans config.yaml)
    out_path = getattr(
        config.paths,
        "hyperparameters_table_latex_path",
        str(Path(config.paths.table_latex_path).parent / "hyperparameters_table.tex"),
    )

    latex_template = f"""\\begin{{table}}[!t]
                         \\renewcommand{{\\arraystretch}}{{1.3}}
                         \\caption{{Training Hyperparameters}}
                         \\label{{tab:hyperparameters}}
                         \\centering
                         \\begin{{tabular}}{{l c}}
                         \\hline
                         \\bfseries Hyperparameter & \\bfseries Value \\\\
                         \\hline\\hline
                         Optimizer & {optimizer} \\\\
                         Learning Rate & {lr_str} \\\\
                         Batch Size & {batch_size} \\\\
                         Epochs & {epochs} \\\\
                         Loss Function & {loss} \\\\
                         \\hline
                         \\end{{tabular}}
                         \\end{{table}}"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex_template)

    logger.info(
        f"Training hyperparameters LaTeX table generated and saved to {out_path}"
    )


def generate_all_tables(config: Config) -> None:
    """Generate all LaTeX tables for the paper."""
    generate_ae_latex_table(config)
    generate_latex_table(config)
    generate_training_hyperparameters_latex_table(config)
    # generate_accuracy_latex_table(config)
