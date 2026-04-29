# ML_Dead_Chess_Champions

## Abstract
ML_Dead_Chess_Champions is a reproducible data-processing and representation-learning pipeline developed to extract and quantify stylistic signatures from historical chess games. The pipeline acquires games in Portable Game Notation (PGN), constructs a move-level dataset of board states and moves, computes per-move vector representations, compresses these representations via an autoencoder, produces two-dimensional embeddings with UMAP, and evaluates inter-player and cross-split similarities using the Jensen–Shannon divergence. The codebase emphasises modularity, configurability (via Pydantic-validated settings), and clear instrumentation for reproducible research.

## Project Objectives
- Acquire and standardise historical chess game data for a curated set of players.
- Construct a move-level dataset that records board FENs and the player's UCI moves.
- Convert (FEN, move) pairs into fixed-size vectors suitable for representation learning.
- Learn compact latent representations via a feed-forward autoencoder; derive 2D embeddings with UMAP for visualization and analysis.
- Quantify stylistic distances between players and evaluate the stability of embedding methods across train/test splits.

## High-level Pipeline and Steps
The pipeline is orchestrated through the command-line entry point `main.py`. Individual stages may be executed independently to support iterative experimentation:

- `fetch` — Retrieve raw PGN files for configured players from chessgames.com.
- `build` — Parse PGN files and produce a consolidated, move-level Parquet dataset.
- `stats` — Extract opening (ECO) statistics from PGN headers and persist them.
- `vectors` — Convert (FEN, move) pairs into fixed-size vectors and save train/test `.npy` arrays.
- `autoencoder` — Train the autoencoder on training vectors and encode train/test into latent vectors.
- `umap` — Fit UMAP on training latent vectors and transform test latent vectors (or apply an existing UMAP model).
- `evaluate` — Compute pairwise and cross-split Jensen–Shannon distances for a given method.
- `results` — Convenience pipeline that builds AE -> UMAP -> JSD for model-generated embeddings (Maia variants) and writes evaluation artifacts.
- `visualize` — Generate publication-ready graphics and LaTeX tables (includes per-method heatmaps and a JSD stability table).

## Quick start: example invocations
Run an individual pipeline step via the CLI. The following examples illustrate standard usages.

Create position vectors (FEN->vector):

    python main.py vectors --config config/default.yml

Train the autoencoder, derive UMAP embeddings, and evaluate UMAP-based JSD:

    python main.py autoencoder --config config/default.yml
    python main.py umap --config config/default.yml
    python main.py evaluate --config config/default.yml --method umap

If you already have a trained UMAP model (serialized) and only want to transform the test latents, use the `umap` step which will load the serialized model and apply it to the test latents.

To compute the AE -> UMAP -> JSD pipeline for the Maia variants (Maia-2, Maia-2 FT, Maia-2 FT + MCTS) and save the evaluation artifacts, run:

    python main.py results --config config/default.yml

To generate all visuals and LaTeX tables (including a per-method JSD heatmap and a per-player JSD stability table):

    python main.py visualize --config config/default.yml

Notes:
- Generating predictions with MCTS (if required) can be computationally expensive; adjust `num_simulations` when calling the evaluation utilities if you want a quicker run for testing.

## Configuration
All runtime parameters and filesystem paths are centralised in a Pydantic `Config` model defined within `src/core/config.py`. The pipeline reads configuration overrides from a YAML file (default path: `config/default.yml`) when present; otherwise, it employs sensible defaults. The `Config.from_yaml(path)` helper performs validation and creates the required directory layout.

## Input and Output Artifacts
- Input
  - Raw PGN files for each configured player are stored under `paths.raw_data/<player_id>/`.
- Primary intermediate artifacts
  - Consolidated move-level dataset: `paths.dataset_path` (Parquet).
  - Train/test splits: `paths.train_set_path`, `paths.test_set_path` (Parquet).
  - Per-split raw vectors: `paths.train_vectors_path`, `paths.test_vectors_path` (.npy).
  - Autoencoder latent vectors: `paths.train_encoded_vectors_path`, `paths.test_encoded_vectors_path` (.npy).
- Representation-learning artifacts
  - Autoencoder state dict: `paths.autoencoder_model_path` (.pth).
  - Serialized UMAP instance: `paths.umap_model_path` (.pkl).
  - UMAP 2D coordinates: `paths.train_umap_result_path`, `paths.test_umap_result_path` (Parquet).
  - Method-specific embeddings (generated from model predictions): `paths.method_train_embeddings_template` / `paths.method_test_embeddings_template` (Parquet).
- Evaluation artifacts
  - Pairwise and cross-split Jensen–Shannon distance tables: written under `paths.evaluation_dir` as Parquet. Filenames follow the convention `distances_{split}_{method}.parquet`, `cross_distances_{method}.parquet`, and `full_cross_distances_{method}.parquet`.

## Visualization and LaTeX tables
- `python main.py visualize` now generates:
  - Learning curves and moves distribution graphs.
  - JSD heatmaps and asymmetric stability heatmaps for each configured method (UMAP and Maia variants) using the templates defined in `config`.
  - A LaTeX table summarising per-player train-vs-test JSD (diagonal), with deltas of each Maia method relative to the TEST reference (UMAP). The table is saved to `paths.jsd_table_latex_path`.

## Implementation notes and assumptions
- Position-to-vector conversion
  - Each (FEN, move) pair is converted by encoding the board state immediately before and after the move and concatenating these encodings into a single flattened vector. See `src/features/umap.py` for the deterministic conversion routine.
- Autoencoder
  - The autoencoder decoder concludes with a `Sigmoid` activation and the training loss is currently binary cross-entropy (BCE). This design presumes input vectors are normalised to the [0, 1] interval. If the input vectors are not scaled to [0, 1], the current architecture and loss may be inappropriate; consider using an identity output with mean-squared error (MSE) as an alternative; or normalise inputs before training/inference.
- UMAP
  - A thin `StyleUMAP` wrapper around `umap.UMAP` is provided with `save_model` and `load_model` helpers. The loader is implemented to allow deserialization without requiring a pre-existing instance.
- Evaluation
  - Jensen–Shannon distances between players are estimated by binning 2D embeddings into histograms on a common grid derived from global bounds; this ensures consistent binning for pairwise and cross-split comparisons.

## Reproducibility and logging
- The pipeline uses a central logger provided by `src.core.utils.getLogger()` to emit structured informative messages throughout execution. These messages are intended to form an audit trail for experimental runs.
- The `PathsConfig.make_directories()` helper ensures that required directories exist prior to artifact persistence; it is invoked automatically when a configuration is loaded via `Config.from_yaml`.

## Project structure (selected)
- `main.py` — CLI entry point orchestrating discrete pipeline steps.
- `src/core/config.py` — Pydantic configuration models and filesystem helpers.
- `src/core/utils.py` — Centralised logging and auxiliary utilities.
- `src/data/fetch_games.py` — Download utilities for obtaining PGN files from chessgames.com.
- `src/data/build_dataset.py` — PGN parsing and move-level dataset construction.
- `src/data/opening_stats.py` — Extraction of opening (ECO) statistics.
- `src/features/compute_vectors.py` — Conversion of (FEN, move) pairs to vectors and persistence.
- `src/features/umap.py` — Board encoding utilities and UMAP persistence helpers.
- `src/models/autoencoder.py` — Feed-forward autoencoder implementation.
- `src/training/train_autoencoder.py` — Training and encoding utilities for the autoencoder.
- `src/training/train_umap.py` — UMAP fitting and transformation workflow utilities.
- `src/evaluation/compute_distances.py` — Jensen–Shannon-based evaluation routines.
- `src/evaluation/compute_model_jsd.py` — AE->UMAP->JSD pipeline for model-generated embeddings (Maia variants).
- `src/visualization/graphics.py` — Graphics generation (heatmaps, stability plots, distribution plots).
- `src/visualization/tables.py` — LaTeX table generation (dataset, AE, accuracies, JSD stability table).
