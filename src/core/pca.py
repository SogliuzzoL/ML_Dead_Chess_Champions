import pickle

import cuml.accel as accel
from sklearn.decomposition import PCA

accel.install(log_level="debug")


class StylePCA(PCA):
    def save_model(self, path):
        pickle.dump(self, open(path, "wb"))

    def load_model(self, path):
        return pickle.load(open(path, "rb"))
