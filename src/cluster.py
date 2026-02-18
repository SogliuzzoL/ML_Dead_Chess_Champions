import os

import numpy as np
import pandas as pd
from cuml import UMAP
from sklearn.model_selection import train_test_split

from core.config import DATA_FOLDER, DATASET_PATH, MAIA_EMBEDDINGS_PATH, logger

if __name__ == "__main__":
    embeddings = np.load(MAIA_EMBEDDINGS_PATH, mmap_mode='r')
    df = pd.read_parquet(DATASET_PATH)
    logger.info("Loaded %d embeddings", len(embeddings))

    assert len(embeddings) == len(
        df), "Number of embeddings does not match number of records in the dataset"

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, df["player_name"], test_size=0.2)

    umap_model = UMAP(n_components=2)
    logger.info("Fitting UMAP model to training data")
    umap_model.fit(X_train)

    logger.info("Transforming test data using UMAP model")
    X_test_umap = umap_model.transform(X_test)

    test_umap_df = pd.DataFrame(X_test_umap, columns=["UMAP1", "UMAP2"])
    test_umap_df["player_name"] = y_test.values

    test_umap_df.to_parquet(os.path.join(DATA_FOLDER, "test_umap.parquet"))
