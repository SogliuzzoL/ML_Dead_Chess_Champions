import numpy as np
import umap
from sklearn.model_selection import train_test_split

from core.config import MAIA_EMBEDDINGS_PATH, logger

embeddings = np.load(MAIA_EMBEDDINGS_PATH)

train, test = train_test_split(embeddings, test_size=0.2, random_state=42)

umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1,
                       n_components=2, random_state=42)

umap_embeddings = umap_model.fit_transform(train)
