import requests
from bs4 import BeautifulSoup


def fetch_chessgames(player_id):
    url = f"https://www.chessgames.com/perl/chessplayer?pid={player_id}"
