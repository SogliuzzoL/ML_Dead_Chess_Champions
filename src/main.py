import pandas as pd
from maia2 import model

if __name__ == "__main__":
    all_moves = model.get_all_possible_moves()
    maia = model.from_pretrained("rapid", "cpu")
