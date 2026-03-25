import logging

import numpy as np
import pandas as pd
import torch
from maia2 import inference, model
from maia2 import main as maia

from core.config import ProjectConfig

logger = logging.getLogger(__name__)


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


def extract_styles(config: ProjectConfig, is_test: bool = False):
    """
    Extracts underlying style embeddings from the MAIA model leveraging the injected configuration.
    """
    input_path = config.test_set_path if is_test else config.train_set_path
    output_path = (
        config.test_maia_embeddings_path
        if is_test
        else config.train_maia_embeddings_path
    )

    logger.info(f"Extracting styles from {input_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    maia_model = model.from_pretrained("rapid", device=device)
    style_extractor = MAIA2StyleExtractor(maia_model)

    df = pd.read_parquet(input_path)
    df["active_elo"] = 2500
    df["opponent_elo"] = 2500

    # Dynamically calling the column order from the configuration instance
    df_ready: pd.DataFrame = df.loc[:, config.maia_col_order].copy()

    inference.inference_batch(df_ready, maia_model, True, 128, 4)

    embeddings = style_extractor.get_embeddings()
    logger.info(
        f"Saving extracted embeddings with shape {embeddings.shape} to {output_path}"
    )
    np.save(output_path, embeddings)

    style_extractor.remove_hook()
