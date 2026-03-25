from chess.engine import SimpleEngine
from flask import Flask, jsonify, render_template, request

from core.config import STOCKFISH_MODEL_PATH, base_player_dict
from core.engine import MaiaEngine

app = Flask(__name__, template_folder="../templates")


engine = MaiaEngine()

stockfish = SimpleEngine.popen_uci(STOCKFISH_MODEL_PATH)


@app.route("/")
def index():
    players = list(base_player_dict.values())
    _, elo_dict, _ = engine.prepare
    standard_elos = list(elo_dict.keys())

    return render_template("index.html", players=players, standard_elos=standard_elos)


@app.route("/get-move", methods=["POST"])
def get_move():
    data = request.get_json()
    try:
        default_params = {
            "c_puct": 1.5,
            "scale": 400.0,
            "threshold": 0,
            "num_simulations": 500,
        }

        move_uci, move_dict, tree_data = engine.predict_mcts(
            data["fen"],
            data["pgn"],
            stockfish,
            **default_params,
            active_elo=data["active_elo"],
            opponent_elo=data["opponent_elo"],
        )

        # move_uci, move_dict = engine.predict_move(
        #     data["fen"],
        #     data["active_elo"],
        #     data["opponent_elo"]
        # )

        return jsonify(
            {"move": move_uci, "probabilities": move_dict, "tree": tree_data}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
