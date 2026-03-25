import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List

# ==========================================
# GLOBAL CONFIGURATION
# ==========================================
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.google.com/",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class ProjectConfig:
    # ==========================================
    # CORE DIRECTORIES
    # ==========================================
    data_folder: str = "data"
    model_folder: str = "models"
    result_folder: str = "results"

    # ==========================================
    # PLAYER DEFINITIONS & DATA STRUCTURES
    # ==========================================
    base_player_dict: Dict[str, str] = field(
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

    player_reference: List[str] = field(
        default_factory=lambda: ["Korchnoi", "Ivanchuk", "Anand", "Karpov"]
    )

    maia_col_order: List[str] = field(
        default_factory=lambda: ["fen", "move", "active_elo", "opponent_elo"]
    )

    dataset_col_order: List[str] = field(
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

    def create_directories(self):
        """Generates foundational directories if they are currently absent."""
        os.makedirs(self.data_folder, exist_ok=True)
        os.makedirs(self.model_folder, exist_ok=True)
        os.makedirs(self.result_folder, exist_ok=True)

    # ==========================================
    # MODEL ARTIFACT PATHS
    # ==========================================
    @property
    def stockfish_model_path(self) -> str:
        return os.path.join(self.model_folder, "stockfish")

    @property
    def champions_embeddings_path(self) -> str:
        return os.path.join(self.model_folder, "champions_style_embeddings.pth")

    @property
    def mcts_optimization_db_path(self) -> str:
        return os.path.join(self.data_folder, "mcts_optimization.db")

    @property
    def pca_model_path(self) -> str:
        return os.path.join(self.model_folder, "pca_model.pkl")

    @property
    def autoencoder_model_path(self) -> str:
        return os.path.join(self.model_folder, "autoencoder_model.pth")

    @property
    def umap_model_path(self) -> str:
        return os.path.join(self.model_folder, "umap_model.pkl")

    @property
    def umap_state_model_path(self) -> str:
        return os.path.join(self.model_folder, "umap_state_model.pkl")

    # ==========================================
    # AGGREGATE DATASETS
    # ==========================================
    @property
    def dataset_path(self) -> str:
        return os.path.join(self.data_folder, "chess_positions.parquet")

    @property
    def train_set_path(self) -> str:
        return os.path.join(self.data_folder, "train_set.parquet")

    @property
    def test_set_path(self) -> str:
        return os.path.join(self.data_folder, "test_set.parquet")

    @property
    def generated_set_path(self) -> str:
        return os.path.join(self.data_folder, "generated_set.parquet")

    @property
    def opening_stats_path(self) -> str:
        return os.path.join(self.data_folder, "opening_stats.parquet")

    # ==========================================
    # TRAINING PIPELINE (MODEL FITTING)
    # ==========================================
    @property
    def train_vectors_path(self) -> str:
        return os.path.join(self.data_folder, "train_vectors.npy")

    @property
    def train_umap_vectors_path(self) -> str:
        return os.path.join(self.data_folder, "train_umap_vectors.npy")

    @property
    def train_maia_embeddings_path(self) -> str:
        return os.path.join(self.data_folder, "train_maia_style_embeddings.npy")

    @property
    def train_umap_result_path(self) -> str:
        return os.path.join(self.data_folder, "train_umap_result.parquet")

    @property
    def train_umap_state_result_path(self) -> str:
        return os.path.join(self.data_folder, "train_umap_state_result.parquet")

    # ==========================================
    # INFERENCE PIPELINE (TEST EVALUATION)
    # ==========================================
    @property
    def test_vectors_path(self) -> str:
        return os.path.join(self.data_folder, "test_vectors.npy")

    @property
    def test_umap_vectors_path(self) -> str:
        return os.path.join(self.data_folder, "test_umap_vectors.npy")

    @property
    def test_maia_embeddings_path(self) -> str:
        return os.path.join(self.data_folder, "test_maia_style_embeddings.npy")

    @property
    def test_umap_result_path(self) -> str:
        return os.path.join(self.data_folder, "test_umap_result.parquet")

    @property
    def test_umap_state_result_path(self) -> str:
        return os.path.join(self.data_folder, "test_umap_state_result.parquet")

    # ==========================================
    # GENERATION PIPELINE (STYLE COMPARISON)
    # ==========================================
    @property
    def generated_vectors_path(self) -> str:
        return os.path.join(self.data_folder, "generated_vectors.npy")

    @property
    def generated_umap_vectors_path(self) -> str:
        return os.path.join(self.data_folder, "generated_umap_vectors.npy")

    @property
    def generated_maia_embeddings_path(self) -> str:
        return os.path.join(self.data_folder, "generated_maia_style_embeddings.npy")

    @property
    def generated_umap_result_path(self) -> str:
        return os.path.join(self.data_folder, "generated_umap_result.parquet")

    @property
    def generated_umap_state_result_path(self) -> str:
        return os.path.join(self.data_folder, "generated_umap_state_result.parquet")

    # ==========================================
    # EVALUATION METRICS & DISTANCE MATRICES
    # ==========================================
    @property
    def maia_result_path(self) -> str:
        return os.path.join(self.data_folder, "maia_result.parquet")

    @property
    def test_evaluation_result_path(self) -> str:
        return os.path.join(self.result_folder, "test_set_evaluation.parquet")

    @property
    def mcts_params_result_path(self) -> str:
        return os.path.join(self.result_folder, "players_mcts_params.parquet")

    @property
    def distances_train_result_path(self) -> str:
        return os.path.join(self.data_folder, "player_distances_train.parquet")

    @property
    def distances_test_result_path(self) -> str:
        return os.path.join(self.data_folder, "player_distances_test.parquet")

    @property
    def cross_distances_result_path(self) -> str:
        return os.path.join(self.data_folder, "cross_distances_test_vs_gen.parquet")

    @property
    def cross_distances_train_test_result_path(self) -> str:
        return os.path.join(self.data_folder, "cross_distances_train_test.parquet")


# Default instance instantiation to maintain backward compatibility during the transitional phase.
default_config = ProjectConfig()
