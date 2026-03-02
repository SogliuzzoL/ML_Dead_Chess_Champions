import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from lxml import html
from tenacity import retry, stop_after_delay, wait_exponential
from tqdm import tqdm

from core.config import DATA_FOLDER, HEADERS, logger, player_dict

os.makedirs(DATA_FOLDER, exist_ok=True)


@retry(
    stop=stop_after_delay(360),
    wait=wait_exponential(max=60),
    reraise=True
)
def fetch_url(url: str) -> html.HtmlElement:
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch URL: {url}")

    return html.fromstring(response.content)


@retry(
    stop=stop_after_delay(360),
    wait=wait_exponential(max=60),
    reraise=True
)
def download_pgn(pid: str, gid: str, download_url: str) -> None:
    os.makedirs(os.path.join(DATA_FOLDER, pid), exist_ok=True)
    file_path = os.path.join(DATA_FOLDER, pid, f"{gid}.pgn")

    if os.path.exists(file_path):
        return

    response = requests.get(download_url, headers=HEADERS)
    if response.status_code != 200:
        raise Exception(f"Failed to download PGN {gid}")

    with open(file_path, "wb") as f:
        f.write(response.content)

    time.sleep(random.uniform(0.5, 1.5))


def fetch_chessgames(player_id: str, player_name: str, executor: ThreadPoolExecutor) -> None:
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
                download_url = f"https://www.chessgames.com/njs/api/game/downloadPGN/{gid}"

                futures.append(executor.submit(
                    download_pgn, player_id, gid, download_url))

            for future in as_completed(futures):
                try:
                    future.result()
                    pbar.update(1)
                except Exception:
                    pass

            page_id += 1


def fetch_all_games():
    logger.info("Starting fetching all games...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        for player_id, player_name in player_dict.items():
            fetch_chessgames(player_id, player_name, executor)
