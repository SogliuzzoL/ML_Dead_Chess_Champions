from flask import Flask, jsonify, render_template, request

from core.engine import MaiaEngine

app = Flask(__name__,
            template_folder='../templates')


engine = MaiaEngine()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get-move", methods=["POST"])
def get_move():
    data = request.get_json()
    try:
        move_uci, move_dict = engine.predict_move_without_repetition(
            data["fen"],
            data["pgn"],
            int(data["active_elo"]),
            int(data["opponent_elo"])
        )
        return jsonify({
            "move": move_uci,
            "probabilities": move_dict
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
