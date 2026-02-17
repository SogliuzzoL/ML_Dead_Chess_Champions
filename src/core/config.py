import logging
import os

DATA_FOLDER = "data"
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
    "12183": "Ivanchuk",
    "14220": "Timman",
    "11227": "Larsen",
    "14568": "Portisch",
    "13755": "Beliavsky",
    "12112": "Andersson",
    "12181": "Short",
    "79773": "Deschapelles",
    "31576": "Philidor",
    "16002": "Morphy",
    "15866": "Korchnoi",
    "12088": "Anand",
    "10240": "Alekhine",
    "47544": "Capablanca",
    "16149": "Petrosian",
    "20719": "Karpov",
    "19233": "Fischer",
    "14380": "Tal",
    "15940": "Kasparov",
}

DATASET_PATH = os.path.join(DATA_FOLDER, "chess_positions.parquet")
MAIA_RESULT_PATH = os.path.join(DATA_FOLDER, "maia_result.parquet")
