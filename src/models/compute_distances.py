import os
from itertools import combinations

import numpy as np
import pandas as pd
import tqdm
from scipy.spatial.distance import jensenshannon

from core.config import (
    CROSS_DISTANCES_TRAIN_TEST_RESULT_PATH,
    DISTANCES_TEST_RESULT_PATH,
    DISTANCES_TRAIN_RESULT_PATH,
    TEST_UMAP_RESULT_PATH,
    TEST_UMAP_STATE_RESULT_PATH,
    TRAIN_UMAP_RESULT_PATH,
    TRAIN_UMAP_STATE_RESULT_PATH,
    logger,
)


def compute_js_distance(emb1, emb2, bins=15, bounds=None):
    hist1, _, _ = np.histogram2d(emb1[:, 0], emb1[:, 1], bins=bins, range=bounds)
    hist2, _, _ = np.histogram2d(emb2[:, 0], emb2[:, 1], bins=bins, range=bounds)

    p = hist1.flatten()
    q = hist2.flatten()

    return jensenshannon(p, q, base=2)


def compute_distances(state_mode=False, is_test=False):
    if is_test:
        result_path = (
            TEST_UMAP_STATE_RESULT_PATH if state_mode else TEST_UMAP_RESULT_PATH
        )
        distances_path = DISTANCES_TEST_RESULT_PATH
    else:
        result_path = (
            TRAIN_UMAP_STATE_RESULT_PATH if state_mode else TRAIN_UMAP_RESULT_PATH
        )
        distances_path = DISTANCES_TRAIN_RESULT_PATH

    logger.info("Loading UMAP results...")
    df = pd.read_parquet(result_path)

    logger.info("Calculating distances between players...")
    distance_df = []
    player_names = df["player_name"].unique()

    umap1_min, umap1_max = df["UMAP1"].min(), df["UMAP1"].max()
    umap2_min, umap2_max = df["UMAP2"].min(), df["UMAP2"].max()
    global_bounds = [[umap1_min, umap1_max], [umap2_min, umap2_max]]

    player_combinations = list(combinations(player_names, 2))
    progress_bar = tqdm.tqdm(player_combinations)

    for player1, player2 in progress_bar:
        progress_bar.set_description(
            f"Calculating distance between {player1} and {player2}"
        )
        embeddings1 = df[df["player_name"] == player1][["UMAP1", "UMAP2"]].values
        embeddings2 = df[df["player_name"] == player2][["UMAP1", "UMAP2"]].values

        distance_js = compute_js_distance(
            embeddings1, embeddings2, bins=50, bounds=global_bounds
        )
        distance_df.append(
            {"Player1": player1, "Player2": player2, "JSDistance": distance_js}
        )

    logger.info("Saving distances...")
    distance_df = pd.DataFrame(distance_df)
    distance_df.sort_values("JSDistance", inplace=True)
    print(distance_df.head())
    print(distance_df.tail())
    distance_df.to_parquet(distances_path, index=False)
    logger.info("Saving distances...")


def compute_train_test_distances(state_mode=False):
    train_path = TRAIN_UMAP_STATE_RESULT_PATH if state_mode else TRAIN_UMAP_RESULT_PATH
    test_path = TEST_UMAP_STATE_RESULT_PATH if state_mode else TEST_UMAP_RESULT_PATH

    logger.info("Loading Train and Test UMAP latent representations...")
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    distance_df = []
    player_names = df_train["player_name"].unique()

    umap1_min = min(df_train["UMAP1"].min(), df_test["UMAP1"].min())
    umap1_max = max(df_train["UMAP1"].max(), df_test["UMAP1"].max())
    umap2_min = min(df_train["UMAP2"].min(), df_test["UMAP2"].min())
    umap2_max = max(df_train["UMAP2"].max(), df_test["UMAP2"].max())
    global_bounds = [[umap1_min, umap1_max], [umap2_min, umap2_max]]

    progress_bar = tqdm.tqdm(player_names)
    for player in progress_bar:
        progress_bar.set_description(f"Computing Train vs Test divergence for {player}")

        emb_train = df_train[df_train["player_name"] == player][
            ["UMAP1", "UMAP2"]
        ].values
        emb_test = df_test[df_test["player_name"] == player][["UMAP1", "UMAP2"]].values

        if len(emb_train) == 0 or len(emb_test) == 0:
            continue

        distance_js = compute_js_distance(
            emb_train, emb_test, bins=50, bounds=global_bounds
        )
        distance_df.append({"Player": player, "JSDistance_Train_Test": distance_js})

    logger.info("Exporting cross-distance metrics...")
    distance_df = pd.DataFrame(distance_df)
    distance_df.sort_values("JSDistance_Train_Test", inplace=True)
    distance_df.to_parquet(CROSS_DISTANCES_TRAIN_TEST_RESULT_PATH, index=False)
