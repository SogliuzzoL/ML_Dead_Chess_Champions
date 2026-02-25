import pickle

import cuml.accel as accel
from umap import UMAP

accel.install(log_level="debug")


class StyleUMAP(UMAP):
    def save_model(self, path):
        pickle.dump(self, open(path, "wb"))

    def load_model(self, path):
        return pickle.load(open(path, "rb"))
