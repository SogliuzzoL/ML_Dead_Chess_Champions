import torch
from maia2 import inference, model


class MaiaEngine:
    def __init__(self, model_type="rapid"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.from_pretrained(model_type, self.device)
        self.prepare = inference.prepare()

    def predict_move(self, fen, active_elo=2500, opponent_elo=2500):
        result, _ = inference.inference_each(
            self.model, self.prepare, fen, active_elo, opponent_elo)
        best_move = list(result.keys())[0]
        return best_move, result
