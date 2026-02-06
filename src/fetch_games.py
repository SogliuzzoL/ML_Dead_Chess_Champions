import logging
import os

import requests
from lxml import html
from tenacity import retry, stop_after_attempt, wait_exponential

DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

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


def log_retry_attempt(retry_state):
    """Logue chaque tentative de retry avec le temps d'attente."""
    wait_time = retry_state.next_action.sleep
    attempt = retry_state.attempt_number
    error = retry_state.outcome.exception()
    logger.warning(
        f"Tentative n°{attempt} échouée ({error}). Attente de {wait_time:.1f}s...")


@retry(stop=stop_after_attempt(7), wait=wait_exponential(max=10), before_sleep=log_retry_attempt, reraise=True)
def fetch_url(url: str) -> html.HtmlElement:
    """
    Fetches the content of a URL and returns an lxml tree.
    Retries up to 7 times with exponential backoff if the request fails.

    Args:
        url (str): The URL to fetch.
    Returns:
        lxml.html.HtmlElement: The parsed HTML tree of the response content.
    Raises:
        Exception: If the request fails after 7 attempts.
    """
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch URL: {url} with status code: {response.status_code}")

    tree = html.fromstring(response.content)
    return tree


@retry(stop=stop_after_attempt(7), wait=wait_exponential(max=10), before_sleep=log_retry_attempt, reraise=True)
def download_pgn(pid: str, gid: str, download_url: str) -> None:
    """
    Downloads a PGN file for a given game and saves it to the data folder.
    Retries up to 7 times with exponential backoff if the download fails.

    Args:
        pid (str): Player ID.
        gid (str): Game ID.
        download_url (str): URL to download the PGN file from.

    Raises:
        Exception: If the download fails after 7 attempts.
    """
    os.makedirs(os.path.join(DATA_FOLDER, pid), exist_ok=True)
    if os.path.exists(os.path.join(DATA_FOLDER, pid, f"{gid}.pgn")):
        logger.info(f"PGN for game {gid} already exists. Skipping download.")
        return
    response = requests.get(download_url, headers=HEADERS)
    if response.status_code != 200:
        raise Exception(
            f"Failed to download PGN for game {gid} with status code: {response.status_code}")
    file_path = os.path.join(DATA_FOLDER, pid, f"{gid}.pgn")
    with open(file_path, "wb") as f:
        f.write(response.content)
    logger.info(f"Downloaded game {gid} for player {player_id}")


def fetch_chessgames(player_id: str) -> None:
    """
    Fetches chess games for a given player ID from chessgames.com and downloads the PGN files.

    Args:
        player_id (str): The player ID to fetch games for.
    """
    page_id = 1
    while True:
        url = f"https://www.chessgames.com/perl/chess.pl?page={page_id}&pid={player_id}"
        XPath_table = "//table[@cellpadding='3']"

        tree = fetch_url(url)
        table = tree.xpath(XPath_table)

        if not table:
            logger.info("No more games found.")
            break

        table = table[0]
        rows = table.xpath(".//tr")[1:]

        for row in rows:
            cells = row.xpath(".//td")
            link = cells[0].xpath(".//a/@href")
            gid = link[0].split("gid=")[-1]
            download_url = f"https://www.chessgames.com/njs/api/game/downloadPGN/{gid}"
            download_pgn(player_id, gid, download_url)
        page_id += 1


if __name__ == "__main__":
    player_id = "15940"  # Kasparov's player ID
    fetch_chessgames(player_id)
