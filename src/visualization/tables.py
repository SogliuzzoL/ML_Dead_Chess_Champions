from pathlib import Path

import numpy as np
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
    """Génère le tableau LaTeX avec la précision absolue et le delta entre parenthèses."""

    # Chargement des précisions
    df_acc = pl.read_parquet(config.paths.accuracy_path).sort("player_name")

    # Calcul de la précision globale (moyenne des précisions des joueurs)
    overall_baseline = df_acc.select(pl.col("baseline_accuracy").mean()).item()
    overall_custom = df_acc.select(pl.col("custom_accuracy").mean()).item()
    overall_mcts = df_acc.select(pl.col("mcts_accuracy").mean()).item()

    # Formatage des lignes du tableau
    latex_rows = []
    for row in df_acc.iter_rows(named=True):
        name = row["player_name"]

        # Baseline en valeur absolue
        acc_b = f"{row['baseline_accuracy'] * 100:.1f}\\%"

        # Custom : Absolu + Delta
        abs_c = row["custom_accuracy"] * 100
        delta_c = (row["custom_accuracy"] - row["baseline_accuracy"]) * 100
        acc_c = f"{abs_c:.1f}\\% ({delta_c:+.1f}\\%)"

        # MCTS : Absolu + Delta
        abs_m = row["mcts_accuracy"] * 100
        delta_m = (row["mcts_accuracy"] - row["baseline_accuracy"]) * 100
        acc_m = f"{abs_m:.1f}\\% ({delta_m:+.1f}\\%)"

        latex_rows.append(
            f"                         {name} & {acc_b} & {acc_c} & {acc_m} \\\\"
        )

    # Ajout de la ligne avec les moyennes globales
    latex_rows.append("                         \\hline")

    # Deltas pour la moyenne globale
    avg_abs_c = overall_custom * 100
    avg_delta_c = (overall_custom - overall_baseline) * 100
    avg_c_str = f"{avg_abs_c:.1f}\\% ({avg_delta_c:+.1f}\\%)"

    avg_abs_m = overall_mcts * 100
    avg_delta_m = (overall_mcts - overall_baseline) * 100
    avg_m_str = f"{avg_abs_m:.1f}\\% ({avg_delta_m:+.1f}\\%)"

    latex_rows.append(
        f"                         \\bfseries Average & \\bfseries {overall_baseline * 100:.1f}\\% & \\bfseries {avg_c_str} & \\bfseries {avg_m_str} \\\\"
    )

    table_body = "\n".join(latex_rows)

    # J'ai retiré le (\Delta) des en-têtes puisque le format (+-%) le rend explicite
    latex_template = f"""\\begin{{table}}[!t]
                         \\renewcommand{{\\arraystretch}}{{1.3}}
                         \\caption{{Move-accuracy of the different models on the test set. Values in parentheses indicate the difference relative to Maia-2.}}
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

    # Enregistrement
    out_path = config.paths.accuracy_table_latex_path

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


def generate_jsd_stability_table(config: Config) -> None:
    """Generate a LaTeX table with per-player diagonal JSD and deltas vs TEST.

    Matches the style of `generate_accuracy_latex_table` so the JSD table is IEEE-ready:
    - Per-player rows with TEST (reference) and three Maia columns
    - An "Average" row at the bottom with bolded mean values and deltas
    """

    methods = ["maia2", "maia2_ft", "maia2_ft_mcts"]
    reference = "umap"

    # Human-readable, title-cased labels to match the accuracy table
    method_labels = ["Maia-2", "Maia-2 FT", "Maia-2 FT + MCTS"]

    def _load_diag(method_name: str) -> dict:
        # Prefer cross distances; fall back to full cross matrix diagonal
        try:
            p = config.paths.get_cross_distances_path(method_name, config.jsd.kde)
            df = pl.read_parquet(p)
            if df is not None and df.height > 0:
                return {r["player"]: r["distance"] for r in df.iter_rows(named=True)}
        except Exception:
            pass
        try:
            p = config.paths.get_full_cross_matrix_path(method_name, config.jsd.kde)
            df = pl.read_parquet(p)
            if df is not None and df.height > 0:
                diag = df.filter(pl.col("p_train") == pl.col("p_test"))
                return {r["p_train"]: r["distance"] for r in diag.iter_rows(named=True)}
        except Exception:
            pass
        return {}

    ref_vals = _load_diag(reference)
    method_vals = {m: _load_diag(m) for m in methods}

    # Determine player ordering: alphabetical (case-insensitive) for reproducible display
    if ref_vals:
        players_order = sorted(list(ref_vals.keys()), key=lambda s: s.lower())
    else:
        union_players = set()
        for d in method_vals.values():
            union_players.update(d.keys())
        if union_players:
            players_order = sorted(union_players, key=lambda s: s.lower())
        else:
            players_order = sorted(
                list(config.data.players.values()), key=lambda s: s.lower()
            )

    # Build table rows
    latex_rows = []
    for player in players_order:
        ref = ref_vals.get(player, float("nan"))
        # Use two decimals to reduce width for IEEE two-column figures
        ref_str = f"{ref:.2f}" if ref == ref else "N/A"

        cols = [ref_str]
        for m in methods:
            v = method_vals[m].get(player, float("nan"))
            if v == v and ref == ref:
                delta = v - ref
                cols.append(f"{v:.2f} ({delta:+.2f})")
            elif v == v:
                cols.append(f"{v:.2f} (N/A)")
            else:
                cols.append("N/A (N/A)")

        row = "                         " + player + " & " + " & ".join(cols) + " \\\\"
        latex_rows.append(row)

    # Compute averages across the same player order (ignore NaNs)
    def _mean_for_players(d: dict) -> float:
        vals = [d.get(p) for p in players_order]
        nums = [v for v in vals if isinstance(v, (int, float)) and not np.isnan(v)]
        return float(np.mean(nums)) if nums else float("nan")

    overall_ref = _mean_for_players(ref_vals)
    overall_methods = {m: _mean_for_players(method_vals[m]) for m in methods}

    # Average row values (formatted) — use 2 decimals to save horizontal space
    avg_ref_str = f"{overall_ref:.2f}" if overall_ref == overall_ref else "N/A"
    avg_method_strs = []
    for m in methods:
        mv = overall_methods.get(m, float("nan"))
        if mv == mv and overall_ref == overall_ref:
            avg_method_strs.append(f"{mv:.2f} ({(mv - overall_ref):+.2f})")
        elif mv == mv:
            avg_method_strs.append(f"{mv:.2f} (N/A)")
        else:
            avg_method_strs.append("N/A (N/A)")

    # Append average row in bold, matching the accuracy table style
    latex_rows.append("                         \\hline")
    avg_row = (
        "                         \\bfseries Average & \\bfseries "
        + avg_ref_str
        + " & \\bfseries "
        + avg_method_strs[0]
        + " & \\bfseries "
        + avg_method_strs[1]
        + " & \\bfseries "
        + avg_method_strs[2]
        + " \\\\"
    )
    latex_rows.append(avg_row)

    # Use human-readable labels for the header
    header = " & ".join(["Player", "TEST"] + method_labels)
    table_body = "\n".join(latex_rows)

    # Build LaTeX table using a raw f-string to avoid escape confusion
    latex_template = rf"""\begin{{table}}[!t]
\renewcommand{{\arraystretch}}{{1.3}}
\caption{{Train/Test Jensen-Shannon distances per player (diagonal) and deltas vs TEST.}}
\label{{tab:jsd_stability}}
\centering
\scriptsize
\begin{{tabular}}{{l c c c c}}
\hline
\bfseries {header} \\
\hline\hline
{table_body}
\hline
\end{{tabular}}
\end{{table}}"""

    out_path = config.paths.jsd_table_latex_path
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex_template)

    logger.info("JSD stability LaTeX table saved to %s", out_path)


def generate_all_tables(config: Config) -> None:
    """Generate all LaTeX tables for the paper."""
    generate_ae_latex_table(config)
    generate_latex_table(config)
    generate_training_hyperparameters_latex_table(config)
    generate_accuracy_latex_table(config)
    generate_jsd_stability_table(config)
