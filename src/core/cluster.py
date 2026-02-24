import cuml.accel as accel
from umap import UMAP

accel.install(log_level="debug")


class StyleUMAP(UMAP):
    def save_model(self, path):
        pass

    def load_model(self, path):
        pass


if __name__ == "__main__":
    model = StyleUMAP(n_components=2, n_neighbors=80)
    model.save_model("umap_model.pkl")
