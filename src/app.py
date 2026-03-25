import logging

from chess.engine import SimpleEngine
from flask import Flask, jsonify, render_template, request

from core.config import ProjectConfig
from core.engine import MaiaEngine

# Configure the Flask application
app = Flask(__name__, template_folder="../templates")
logger = logging.getLogger(__name__)

# Initialize the dynamic configuration instance
# Note: In a production environment, these arguments could be sourced
# from environment variables or a separate server-side config.
config = ProjectConfig()
config.create_directories()

# Instantiate the modernized engine with dependency injection
engine = MaiaEngine(config=config)

# Initialize the external chess engine process utilizing dynamic path resolution
stockfish = SimpleEngine.popen_uci(config.stockfish_model_path)


@app.route("/")
def index():
    """
    Renders the primary landing page, dynamically populating player
    references and ELO categories from the active configuration.
    """
    players = list(config.base_player_dict.values())
    _, elo_dict, _ = engine.prepare
    standard_elos = list(elo_dict.keys())

    return render_template("index.html", players=players, standard_elos=standard_elos)


@app.route("/get-move", methods=["POST"])
def get_move():
    """
    Handles asynchronous move prediction requests by orchestrating
    the MCTS algorithm with pre-defined heuristic parameters.
    """
    data = request.get_json()
    try:
        # These parameters could eventually be moved to ProjectConfig for greater modularity
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

        return jsonify(
            {"move": move_uci, "probabilities": move_dict, "tree": tree_data}
        )
    except Exception as e:
        logger.error(f"Inference Failure: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Launch the application in a local development context
    app.run(debug=True, port=5000)
