from flask import Flask, jsonify, render_template, request

from src.core.config import Config
from src.core.utils import getLogger
from src.models.maia import MaiaEngine

logger = getLogger()


def create_app(config: Config) -> Flask:
    """Crée et configure l'application Flask avec le moteur Maia."""
    # Le dossier templates est maintenant relatif à ce fichier
    app = Flask(__name__, template_folder="templates")

    logger.info(
        "Initialisation du moteur Maia pour l'interface web (Chargement sur GPU)..."
    )
    engine = MaiaEngine(config)

    @app.route("/")
    def index():
        # Récupération dynamique des joueurs depuis la configuration
        players = list(config.data.players.values())
        _, elo_dict, _ = engine.prepare
        standard_elos = list(elo_dict.keys())

        return render_template(
            "index.html", players=players, standard_elos=standard_elos
        )

    @app.route("/get-move", methods=["POST"])
    def get_move():
        data = request.get_json()
        try:
            # Plus aucune trace de Stockfish, appel pur au MCTS de Maia
            move_uci, move_dict = engine.predict_mcts(
                fen=data["fen"],
                pgn=data.get("pgn", ""),
                active_elo=data["active_elo"],
                opponent_elo=data["opponent_elo"],
                c_puct=1.5,
                threshold=0.01,
                num_simulations=1000,
            )

            return jsonify({"move": move_uci, "move_dict": move_dict})

        except Exception as e:
            logger.error(f"Erreur de prédiction dans l'interface web : {e}")
            return jsonify({"error": str(e)}), 500

    return app


def run_ui(config: Config) -> None:
    """Lance le serveur web local."""
    app = create_app(config)
    logger.info("Démarrage de l'interface utilisateur sur http://127.0.0.1:5000")
    # Debug est désactivé par défaut pour éviter de recharger le lourd modèle PyTorch en double
    app.run(host="0.0.0.0", port=5000, debug=False)
