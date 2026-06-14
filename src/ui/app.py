"""Flask web interface for interactive move prediction and exploration.

This module exposes `create_app` which constructs a Flask application embedding
an instantiated `MaiaEngine`. The application provides endpoints for rendering
an interactive board UI and for requesting model-predicted moves via MCTS.
"""

from pathlib import Path
from typing import Dict

from flask import Flask, jsonify, render_template, request

from src.core.config import Config
from src.core.utils import getLogger
from src.models.maia import MaiaEngine
from src.models.style_engine import StyleMaiaEngine

logger = getLogger()


def create_app(config: Config) -> Flask:
    """Create and configure the Flask application using Maia engines.

    The Flask instance uses a local `templates` directory for rendering the web
    interface. Two engines are instantiated at startup:
      - baseline `MaiaEngine`
      - `StyleMaiaEngine` (if adapter is present)

    The `/get-move` endpoint accepts an optional `model` field in the request
    JSON which may be either "maia" (default) or "style" to select the
    adapter-enhanced variant.
    """
    # Templates directory is relative to this module
    app = Flask(__name__, template_folder="templates")

    logger.info(
        "Initializing baseline Maia engine for web interface (loading model)..."
    )
    baseline = MaiaEngine(config)

    # Prepare lazy-loading for the style engine. We only enable the UI option
    # if the adapter state file exists. The actual StyleMaiaEngine is created on
    # demand and cached so we don't double-load heavy models unnecessarily.
    adapter_path = Path(config.paths.model) / "saved" / "style_moe.pth"
    style_engine_cache: Dict[str, object] = {"engine": None}

    def get_style_engine():
        if style_engine_cache["engine"] is not None:
            return style_engine_cache["engine"]
        if not adapter_path.exists():
            logger.warning("Style adapter file not found at %s", adapter_path)
            return None
        try:
            logger.info("Instantiating StyleMaiaEngine (lazy)...")
            eng = StyleMaiaEngine(
                config, maia_model=baseline.model, adapter_path=str(adapter_path)
            )
            style_engine_cache["engine"] = eng
            return eng
        except Exception as e:
            logger.exception("Failed to instantiate StyleMaiaEngine: %s", e)
            style_engine_cache["engine"] = None
            return None

    @app.route("/")
    def index():
        # Dynamically obtain player and Elo options from the configuration
        players = list(config.data.players.values())
        _, elo_dict, _ = baseline.prepare
        standard_elos = list(elo_dict.keys())

        # Expose whether the style engine is available so the UI can show options
        # Use adapter_path existence as a lightweight availability signal.
        style_available_local = adapter_path.exists()

        return render_template(
            "index.html",
            players=players,
            standard_elos=standard_elos,
            style_available=style_available_local,
        )

    @app.route("/get-move", methods=["POST"])
    def get_move():
        data = request.get_json()
        model_choice = data.get("model", "maia")

        try:
            if model_choice == "style":
                # Lazy instantiate or reuse the cached style engine
                engine = get_style_engine()
                if engine is None:
                    return jsonify({"error": "Style engine not available"}), 400

                move_uci, move_dict = engine.predict_mcts(
                    fen=data["fen"],
                    pgn=data.get("pgn", ""),
                    active_elo=data["active_elo"],
                    opponent_elo=data["opponent_elo"],
                    c_puct=2.5,
                    threshold=0.01,
                    num_simulations=50,
                )
            else:
                move_uci, move_dict = baseline.predict_mcts(
                    fen=data["fen"],
                    pgn=data.get("pgn", ""),
                    active_elo=data["active_elo"],
                    opponent_elo=data["opponent_elo"],
                    c_puct=1.5,
                    threshold=0.01,
                    num_simulations=100,
                )

            return jsonify({"move": move_uci, "move_dict": move_dict})

        except Exception as e:
            logger.error("Error while predicting move for web UI: %s", e)
            return jsonify({"error": str(e)}), 500

    return app


def run_ui(config: Config) -> None:
    """Launch the local Flask web server for interactive exploration.

    The server is launched with `debug=False` by default to avoid double-loading
    heavy PyTorch models during development reloads.
    """
    app = create_app(config)
    logger.info("Starting web interface at http://127.0.0.1:5000")
    # Debug is disabled to prevent double-loading the heavy PyTorch model
    app.run(host="0.0.0.0", port=5000, debug=False)
