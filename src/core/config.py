"""Application configuration models and filesystem helpers.

This module centralizes project-wide configuration using Pydantic models. It
defines structured containers for filesystem paths, data acquisition settings,
and modelling hyperparameters. Helper methods are provided to create the
required directory layout and to load configuration overrides from a YAML file.

All user-visible documentation is presented in formal academic English to
support reproducible and well-documented data processing workflows.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    """Filesystem layout and canonical paths used by the project.

    Attributes
    ----------
    data, raw_data, model, result, evaluation_dir : str
        Base directories used to organize the project's artifacts.
    dataset_path, train_set_path, test_set_path, opening_stats_path : str
        Canonical locations for processed dataset artifacts.
    train_vectors_path, test_vectors_path : str
        Paths to the NumPy arrays containing position vectors.
    autoencoder_model_path : str
        Location where the trained autoencoder state dict is persisted.
    train_encoded_vectors_path, test_encoded_vectors_path : str
        Paths for latent vectors produced by the autoencoder.
    train_umap_result_path, test_umap_result_path, umap_model_path : str
        Paths for UMAP outputs and serialized model.
    """

    data: str = "data/"
    raw_data: str = "data/raw/"
    model: str = "models/"
    result: str = "results/"
    evaluation_dir: str = "results/evaluation/"

    dataset_path: str = "data/processed/dataset.parquet"
    train_set_path: str = "data/processed/train.parquet"
    test_set_path: str = "data/processed/test.parquet"
    opening_stats_path: str = "data/processed/opening_stats.parquet"

    train_vectors_path: str = "data/processed/train_vectors.npy"
    test_vectors_path: str = "data/processed/test_vectors.npy"

    autoencoder_model_path: str = "models/saved/autoencoder.pth"

    train_encoded_vectors_path: str = "data/processed/train_encoded_vectors.npy"
    test_encoded_vectors_path: str = "data/processed/test_encoded_vectors.npy"

    train_umap_result_path: str = "data/processed/train_umap.parquet"
    test_umap_result_path: str = "data/processed/test_umap.parquet"
    umap_model_path: str = "models/saved/style_umap.pkl"

    def make_directories(self):
        """Ensure all configured filesystem paths exist on disk.

        For each configured path, this helper creates parent directories as
        required. If a configured value represents a file (i.e., has a suffix),
        its parent directory is created; otherwise the path itself is treated
        as a directory and created.
        """
        for path_str in self.model_dump().values():
            path_obj = Path(path_str)

            if path_obj.suffix:
                path_obj.parent.mkdir(parents=True, exist_ok=True)
            else:
                path_obj.mkdir(parents=True, exist_ok=True)

    def get_embeddings_path(self, method: str, is_test: bool) -> str:
        """Return a canonical path for embeddings produced by `method`.

        Parameters
        ----------
        method : str
            Short identifier of the embedding method (e.g., 'umap', 'pca').
        is_test : bool
            Whether the requested path is for the test split.

        Returns
        -------
        str
            Filesystem path where embeddings for the requested split and method
            should be stored.
        """
        split = "test" if is_test else "train"
        return f"data/processed/{split}_{method}.parquet"

    def get_distances_path(self, method: str, is_test: bool) -> str:
        """Return a canonical path for distance tables for a given method."""
        split = "test" if is_test else "train"
        return f"{self.evaluation_dir}distances_{split}_{method}.parquet"

    def get_cross_distances_path(self, method: str) -> str:
        """Return the canonical path for cross-split distances for `method`."""
        return f"{self.evaluation_dir}cross_distances_{method}.parquet"


class UMAPConfig(BaseModel):
    """Configuration for UMAP dimensionality reduction."""

    n_components: int = 2


class AutoencoderConfig(BaseModel):
    """Hyperparameters for the autoencoder training routine."""

    latent_dim: int = 128
    epochs: int = 10
    batch_size: int = 1024
    learning_rate: float = 1e-3
    num_workers: int = 0


class DataConfig(BaseModel):
    """Data acquisition and dataset-related configuration.

    Attributes
    ----------
    max_workers : int
        Maximum number of worker threads used for concurrent downloads.
    headers : dict
        HTTP headers used for web requests.
    players : dict
        Mapping from remote player identifier to human-readable player name.
    dataset_col_order : list
        Preferred column ordering for the produced dataset DataFrame.
    """

    max_workers: int = 10
    headers: dict = Field(
        default_factory=lambda: {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://www.google.com/",
        }
    )
    players: dict = Field(
        default_factory=lambda: {
            "10240": "Alekhine",
            "12112": "Andersson",
            "12088": "Anand",
            "13755": "Beliavsky",
            "47544": "Capablanca",
            "19233": "Fischer",
            "12183": "Ivanchuk",
            "20719": "Karpov",
            "15940": "Kasparov",
            "15866": "Korchnoi",
            "11227": "Larsen",
            "16149": "Petrosian",
            "14568": "Portisch",
            "12181": "Short",
            "14380": "Tal",
            "14220": "Timman",
        }
    )
    dataset_col_order: list = Field(
        default_factory=lambda: [
            "game_id",
            "round",
            "player_name",
            "player_color",
            "fen",
            "move",
            "repetition",
            "result",
        ]
    )


class Config(BaseModel):
    """Top-level application configuration model.

    Instances of this class aggregate `PathsConfig`, `DataConfig`,
    `AutoencoderConfig` and `UMAPConfig`. The class method `from_yaml` permits
    loading overrides from a YAML file; when invoked it also ensures the
    configured filesystem layout exists on disk.
    """

    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    autoencoder: AutoencoderConfig = Field(default_factory=AutoencoderConfig)
    umap: UMAPConfig = Field(default_factory=UMAPConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Instantiate a `Config` optionally overriding defaults with a YAML file.

        If `yaml_path` does not exist an instance with default values is returned.
        When a YAML file is present it is parsed and used to construct the Pydantic
        model. After instantiation the required directories referenced by the
        `PathsConfig` are created on disk to ensure downstream code can write
        artifacts reliably.

        Parameters
        ----------
        yaml_path : str
            Path to a YAML file containing configuration overrides.

        Returns
        -------
        Config
            A fully-initialized configuration instance.
        """
        path = Path(yaml_path)
        if not path.exists():
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            yaml_dict = yaml.safe_load(f) or {}

        config_instance = cls(**yaml_dict)
        config_instance.paths.make_directories()
        return config_instance
