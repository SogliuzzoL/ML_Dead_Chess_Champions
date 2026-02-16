let board = null;
let game = new Chess();

function onDrop(source, target) {
    let move = game.move({ from: source, to: target, promotion: 'q' });
    if (move === null) return 'snapback';
    window.setTimeout(makeAIMove, 250);
}

async function makeAIMove() {
    const isWhiteTurn = game.turn() === 'w';
    const activeElo = isWhiteTurn ? $('#eloWhite').val() : $('#eloBlack').val();
    const opponentElo = isWhiteTurn ? $('#eloBlack').val() : $('#eloWhite').val();

    const response = await fetch('/get-move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            fen: game.fen(),
            active_elo: activeElo,
            opponent_elo: opponentElo
        })
    });

    const data = await response.json();
    game.move(data.move, { sloppy: true });
    board.position(game.fen());
}

function initGame() {
    const fen = $('#fenInput').val();
    game = (fen === 'start') ? new Chess() : new Chess(fen);

    board = Chessboard('board', {
        draggable: true,
        position: game.fen(),
        onDrop: onDrop
    });
    if ($('#userColor').val() === 'b' && game.turn() === 'w') {
        makeAIMove();
    }
}