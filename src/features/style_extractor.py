import numpy as np
import pandas as pd
import torch
from maia2 import inference, model
from maia2 import main as maia

from core.config import MAIA_COL_ORDER, logger


class MAIA2StyleExtractor:
    def __init__(self, maia_model: maia.MAIA2Model):
        self.model = maia_model
        self.hook = self.model.last_ln.register_forward_hook(self._hook_fn)
        self.embeddings = []

    def _hook_fn(self, module, input, output: torch.Tensor):
        self.embeddings.append(output.detach().cpu().numpy())

    def clear(self):
        self.embeddings = []

    def remove_hook(self):
        self.hook.remove()

    def get_embeddings(self):
        return np.concatenate(self.embeddings, axis=0)


def extract_styles(input_path: str, output_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    maia_model = model.from_pretrained("rapid", device=device)
    style_extractor = MAIA2StyleExtractor(maia_model)

    df = pd.read_parquet(input_path)
    df["active_elo"] = 2500
    df["opponent_elo"] = 2500

    df_ready: pd.DataFrame = df.loc[:, MAIA_COL_ORDER].copy()

    inference.inference_batch(df_ready, maia_model, True, 128, 4)

    logger.info("Embeddings shape: {}".format(style_extractor.get_embeddings().shape))

    np.save(output_path, style_extractor.get_embeddings())
