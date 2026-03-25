import logging

import numpy as np
import pandas as pd
import tqdm

from core.config import ProjectConfig
from core.umap import position_to_vector

logger = logging.getLogger(__name__)


def compute_vectors(config: ProjectConfig, is_test: bool = False) -> None:
    """
    Computes numerical vectors for chess positions using the dynamic configuration pathways.
    """
    input_path = config.test_set_path if is_test else config.train_set_path
    output_path = config.test_vectors_path if is_test else config.train_vectors_path

    logger.info(f"Computing vectors from {input_path}")
    df = pd.read_parquet(input_path)
    vectors = []

    progress_bar = tqdm.tqdm(df.iterrows(), total=len(df), desc="Vectorizing Positions")
    for _, row in progress_bar:
        vectors.append(position_to_vector(row["fen"], row["move"]))

    vectors = np.array(vectors)
    logger.info(f"Saving computed vectors with shape {vectors.shape} to {output_path}")
    np.save(output_path, vectors)
