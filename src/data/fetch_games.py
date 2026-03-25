import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from lxml import html
from tenacity import retry, stop_after_delay, wait_exponential
from tqdm import tqdm

from core.config import (  # HEADERS remains global for network protocol consistency
    HEADERS,
    ProjectConfig,
)


@retry(stop=stop_after_delay(360), wait=wait_exponential(max=60), reraise=True)
def fetch_url(url: str) -> html.HtmlElement:
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch URL: {url}")
    return html.fromstring(response.content)


@retry(stop=stop_after_delay(360), wait=wait_exponential(max=60), reraise=True)
def download_pgn(config: ProjectConfig, pid: str, gid: str, download_url: str) -> None:
    player_dir = os.path.join(config.data_folder, pid)
    os.makedirs(player_dir, exist_ok=True)
    file_path = os.path.join(player_dir, f"{gid}.pgn")

    if os.path.exists(file_path):
        return

    response = requests.get(download_url, headers=HEADERS)
    if response.status_code != 200:
        raise Exception(f"Failed to download PGN {gid}")

    with open(file_path, "wb") as f:
        f.write(response.content)

    time.sleep(random.uniform(2, 4))


def fetch_chessgames(
    config: ProjectConfig,
    player_id: str,
    player_name: str,
    executor: ThreadPoolExecutor,
) -> None:
    page_id = 1
    with tqdm(desc=f"Downloading {player_name}", unit=" pgn") as pbar:
        while True:
            url = f"https://www.chessgames.com/perl/chess.pl?page={page_id}&pid={player_id}"
            tree = fetch_url(url)
            table = tree.xpath("//table[@cellpadding='3']")

            if not table:
                break

            rows = table[0].xpath(".//tr")[1:]
            futures = []

            for row in rows:
                cells = row.xpath(".//td")
                link = cells[0].xpath(".//a/@href")
                gid = link[0].split("gid=")[-1]
                download_url = (
                    f"https://www.chessgames.com/njs/api/game/downloadPGN/{gid}"
                )

                futures.append(
                    executor.submit(download_pgn, config, player_id, gid, download_url)
                )

            for future in as_completed(futures):
                try:
                    future.result()
                    pbar.update(1)
                except Exception:
                    pass
            page_id += 1


def fetch_all_games(config: ProjectConfig):
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Starting fetching all games using dynamic configuration...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        for player_id, player_name in config.base_player_dict.items():
            fetch_chessgames(config, player_id, player_name, executor)
