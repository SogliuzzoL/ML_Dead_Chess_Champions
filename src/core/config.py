import logging
import os
from token import OP

# ==========================================
# CORE DIRECTORIES
# ==========================================
DATA_FOLDER = "data"
MODEL_FOLDER = "models"
RESULT_FOLDER = "results"

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

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

# ==========================================
# PLAYER DEFINITIONS & DATA STRUCTURES
# ==========================================
base_player_dict = {
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

PLAYER_REFERENCE = ["Korchnoi", "Ivanchuk", "Anand", "Karpov"]

MAIA_COL_ORDER = ["fen", "move", "active_elo", "opponent_elo"]
DATASET_COL_ORDER = [
    "game_id",
    "round",
    "player_name",
    "player_color",
    "fen",
    "move",
    "repetition",
    "result",
]

# ==========================================
# MODEL ARTIFACT PATHS
# ==========================================
STOCKFISH_MODEL_PATH = os.path.join(MODEL_FOLDER, "stockfish")
CHAMPIONS_EMBEDDINGS_PATH = os.path.join(MODEL_FOLDER, "champions_style_embeddings.pth")
MCTS_OPTIMIZATION_DB_PATH = os.path.join(DATA_FOLDER, "mcts_optimization.db")
PCA_MODEL_PATH = os.path.join(MODEL_FOLDER, "pca_model.pkl")
AUTOENCODER_MODEL_PATH = os.path.join(MODEL_FOLDER, "autoencoder_model.pth")
UMAP_MODEL_PATH = os.path.join(MODEL_FOLDER, "umap_model.pkl")
UMAP_STATE_MODEL_PATH = os.path.join(MODEL_FOLDER, "umap_state_model.pkl")

# ==========================================
# AGGREGATE DATASETS
# ==========================================
DATASET_PATH = os.path.join(DATA_FOLDER, "chess_positions.parquet")
TRAIN_SET_PATH = os.path.join(DATA_FOLDER, "train_set.parquet")
TEST_SET_PATH = os.path.join(DATA_FOLDER, "test_set.parquet")
GENERATED_SET_PATH = os.path.join(DATA_FOLDER, "generated_set.parquet")
OPENING_STATS_PATH = os.path.join(DATA_FOLDER, "opening_stats.parquet")

# ==========================================
# TRAINING PIPELINE (MODEL FITTING)
# ==========================================
TRAIN_VECTORS_PATH = os.path.join(DATA_FOLDER, "train_vectors.npy")
TRAIN_UMAP_VECTORS_PATH = os.path.join(DATA_FOLDER, "train_umap_vectors.npy")
TRAIN_MAIA_EMBEDDINGS_PATH = os.path.join(
    DATA_FOLDER, "train_maia_style_embeddings.npy"
)
TRAIN_UMAP_RESULT_PATH = os.path.join(DATA_FOLDER, "train_umap_result.parquet")
TRAIN_UMAP_STATE_RESULT_PATH = os.path.join(
    DATA_FOLDER, "train_umap_state_result.parquet"
)

# ==========================================
# INFERENCE PIPELINE (TEST EVALUATION)
# ==========================================
TEST_VECTORS_PATH = os.path.join(DATA_FOLDER, "test_vectors.npy")
TEST_UMAP_VECTORS_PATH = os.path.join(DATA_FOLDER, "test_umap_vectors.npy")
TEST_MAIA_EMBEDDINGS_PATH = os.path.join(DATA_FOLDER, "test_maia_style_embeddings.npy")
TEST_UMAP_RESULT_PATH = os.path.join(DATA_FOLDER, "test_umap_result.parquet")
TEST_UMAP_STATE_RESULT_PATH = os.path.join(
    DATA_FOLDER, "test_umap_state_result.parquet"
)

# ==========================================
# GENERATION PIPELINE (STYLE COMPARISON)
# ==========================================
GENERATED_VECTORS_PATH = os.path.join(DATA_FOLDER, "generated_vectors.npy")
GENERATED_UMAP_VECTORS_PATH = os.path.join(DATA_FOLDER, "generated_umap_vectors.npy")
GENERATED_MAIA_EMBEDDINGS_PATH = os.path.join(
    DATA_FOLDER, "generated_maia_style_embeddings.npy"
)
GENERATED_UMAP_RESULT_PATH = os.path.join(DATA_FOLDER, "generated_umap_result.parquet")
GENERATED_UMAP_STATE_RESULT_PATH = os.path.join(
    DATA_FOLDER, "generated_umap_state_result.parquet"
)

# ==========================================
# EVALUATION METRICS & DISTANCE MATRICES
# ==========================================
MAIA_RESULT_PATH = os.path.join(DATA_FOLDER, "maia_result.parquet")
TEST_EVALUATION_RESULT_PATH = os.path.join(RESULT_FOLDER, "test_set_evaluation.parquet")
MCTS_PARAMS_RESULT_PATH = os.path.join(RESULT_FOLDER, "players_mcts_params.parquet")

DISTANCES_TRAIN_RESULT_PATH = os.path.join(
    DATA_FOLDER, "player_distances_train.parquet"
)
DISTANCES_TEST_RESULT_PATH = os.path.join(DATA_FOLDER, "player_distances_test.parquet")
CROSS_DISTANCES_RESULT_PATH = os.path.join(
    DATA_FOLDER, "cross_distances_test_vs_gen.parquet"
)
CROSS_DISTANCES_TRAIN_TEST_RESULT_PATH = os.path.join(
    DATA_FOLDER, "cross_distances_train_test.parquet"
)
