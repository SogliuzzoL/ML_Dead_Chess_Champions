import logging
from itertools import combinations

import numpy as np
import pandas as pd
import tqdm
from scipy.spatial.distance import jensenshannon

from core.config import ProjectConfig

logger = logging.getLogger(__name__)


def compute_js_distance(emb1, emb2, bins=15, bounds=None):
    """
    Computes the Jensen-Shannon divergence between two empirical distributions.
    """
    hist1, _, _ = np.histogram2d(emb1[:, 0], emb1[:, 1], bins=bins, range=bounds)
    hist2, _, _ = np.histogram2d(emb2[:, 0], emb2[:, 1], bins=bins, range=bounds)

    p = hist1.flatten()
    q = hist2.flatten()

    return jensenshannon(p, q, base=2)


def compute_distances(config: ProjectConfig, state_mode=False, is_test=False):
    """
    Computes pairwise stylistic distances between players utilizing dynamic configuration pathways.
    """
    if is_test:
        result_path = (
            config.test_umap_state_result_path
            if state_mode
            else config.test_umap_result_path
        )
        distances_path = config.distances_test_result_path
    else:
        result_path = (
            config.train_umap_state_result_path
            if state_mode
            else config.train_umap_result_path
        )
        distances_path = config.distances_train_result_path

    logger.info(f"Loading UMAP representations from {result_path}...")
    df = pd.read_parquet(result_path)

    distance_df = []
    player_names = df["player_name"].unique()

    umap1_min = df["UMAP1"].min()
    umap1_max = df["UMAP1"].max()
    umap2_min = df["UMAP2"].min()
    umap2_max = df["UMAP2"].max()
    global_bounds = [[umap1_min, umap1_max], [umap2_min, umap2_max]]

    progress_bar = tqdm.tqdm(list(combinations(player_names, 2)))
    for p1, p2 in progress_bar:
        progress_bar.set_description(f"Computing JS divergence between {p1} and {p2}")

        emb1 = df[df["player_name"] == p1][["UMAP1", "UMAP2"]].values
        emb2 = df[df["player_name"] == p2][["UMAP1", "UMAP2"]].values

        if len(emb1) == 0 or len(emb2) == 0:
            continue

        distance_js = compute_js_distance(emb1, emb2, bounds=global_bounds)
        distance_df.append({"p1": p1, "p2": p2, "distance": distance_js})

    distance_df = pd.DataFrame(distance_df)
    logger.info(f"Saving computed distances to {distances_path}")
    distance_df.to_parquet(distances_path, index=False)


def compute_train_test_distances(config: ProjectConfig, state_mode=False):
    """
    Computes the intra-player divergence between training and testing sets to evaluate embedding stability.
    """
    train_path = (
        config.train_umap_state_result_path
        if state_mode
        else config.train_umap_result_path
    )
    test_path = (
        config.test_umap_state_result_path
        if state_mode
        else config.test_umap_result_path
    )
    distances_path = config.cross_distances_train_test_result_path

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

        distance_js = compute_js_distance(emb_train, emb_test, bounds=global_bounds)
        distance_df.append({"player": player, "distance": distance_js})

    distance_df = pd.DataFrame(distance_df)
    logger.info(f"Saving cross distances to {distances_path}")
    distance_df.to_parquet(distances_path, index=False)
