import logging
import os

DATA_FOLDER = "data"
MODEL_FOLDER = "models"
RESULT_FOLDER = "results"

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7',
    'Referer': 'https://www.google.com/'
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

player_dict = {
    "10240": "Alekhine",
    "12112": "Andersson",
    "12088": "Anand",
    "13755": "Beliavsky",
    "47544": "Capablanca",
    # "79773": "Deschapelles",
    "19233": "Fischer",
    "12183": "Ivanchuk",
    "20719": "Karpov",
    "15940": "Kasparov",
    "15866": "Korchnoi",
    "11227": "Larsen",
    # "16002": "Morphy",
    # "31576": "Philidor",
    "16149": "Petrosian",
    "14568": "Portisch",
    "12181": "Short",
    "14380": "Tal",
    "14220": "Timman",
}

# Train variables

TRAIN_SET_PATH = os.path.join(DATA_FOLDER, "train_set.parquet")
TRAIN_VECTORS_PATH = os.path.join(DATA_FOLDER, "train_vectors.npy")
TRAIN_MAIA_EMBEDDINGS_PATH = os.path.join(
    DATA_FOLDER, "train_maia_style_embeddings.npy")
TRAIN_UMAP_VECTORS_PATH = os.path.join(DATA_FOLDER, "train_umap_vectors.npy")
TRAIN_UMAP_STATE_RESULT_PATH = os.path.join(
    DATA_FOLDER, "train_umap_state_result.parquet")
TRAIN_UMAP_RESULT_PATH = os.path.join(DATA_FOLDER, "train_umap_result.parquet")

# ===================

DATASET_PATH = os.path.join(DATA_FOLDER, "chess_positions.parquet")
MAIA_RESULT_PATH = os.path.join(DATA_FOLDER, "maia_result.parquet")

TEST_SET_PATH = os.path.join(DATA_FOLDER, "test_set.parquet")

MAIA_COL_ORDER = ["fen", "move", "active_elo", "opponent_elo"]
DATASET_COL_ORDER = ["game_id", "round", "player_name",
                     "player_color", "fen", "move", "repetition", "result"]

STOCKFISH_MODEL_PATH = os.path.join(MODEL_FOLDER, "stockfish")
STOCKFISH_CPL_ANALYSE_PATH = os.path.join(
    DATA_FOLDER, "stockfish_cpl_analysis.parquet")

VECTORS_PATH = os.path.join(DATA_FOLDER, "vectors.npy")
PCA_MODEL_PATH = os.path.join(MODEL_FOLDER, "pca_model.pkl")
AUTOENCODER_MODEL_PATH = os.path.join(MODEL_FOLDER, "autoencoder_model.pth")
UMAP_VECTORS_PATH = os.path.join(DATA_FOLDER, "umap_vectors.npy")
UMAP_MODEL_PATH = os.path.join(MODEL_FOLDER, "umap_model.pkl")
UMAP_RESULT_PATH = os.path.join(DATA_FOLDER, "umap_result.parquet")
DISTANCES_RESULT_PATH = os.path.join(DATA_FOLDER, "player_distances.parquet")

PLAYER_REFERENCE = ["Korchnoi", "Ivanchuk", "Anand", "Karpov"]

MAIA_EMBEDDINGS_PATH = os.path.join(DATA_FOLDER, "maia_style_embeddings.npy")
UMAP_STATE_MODEL_PATH = os.path.join(MODEL_FOLDER, "umap_state_model.pkl")
UMAP_STATE_RESULT_PATH = os.path.join(DATA_FOLDER, "umap_state_result.parquet")
DISTANCES_STATE_RESULT_PATH = os.path.join(
    DATA_FOLDER, "player_state_distances.parquet")
