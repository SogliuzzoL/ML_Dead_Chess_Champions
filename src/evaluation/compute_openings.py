import chess
import chess.pgn as pgn
import polars as pl
from tqdm import tqdm

from src.core.config import Config


def compute_openings(config: Config):
    """Compute opening probability densities for each player and Maia variant.

    The function builds the ECO opening tree from `config/eco.pgn` and then
    performs a probabilistic traversal guided by the Maia variants (baseline,
    finetuned, finetuned+MCTS). For each player and variant we compute the
    probability mass assigned to ECO terminal nodes (ECO code + Opening name).

    Outputs a parquet file at `config.paths.generated_data/opening_densities.parquet`
    with columns: player_name, variant, eco, opening, probability.
    """

    import os
    from collections import defaultdict

    from src.core.utils import getLogger
    from src.models.batched_mcts import BatchedMCTSManager
    from src.models.maia import MaiaEngine

    logger = getLogger()

    eco_path = "config/eco.pgn"
    if not os.path.exists(eco_path):
        raise FileNotFoundError(f"ECO PGN not found at {eco_path}")

    # ------------------------------------------------------------------
    # 1) Parse ECO file and build a tree: fen -> {move_uci: next_fen}
    #    Also collect terminal metadata (ECO code, Opening name) at final fens.
    # ------------------------------------------------------------------
    children_map: dict[str, dict[str, str]] = {}
    terminal_meta: dict[str, list[tuple]] = defaultdict(list)

    with open(eco_path, encoding="utf-8") as f:
        while True:
            game = pgn.read_game(f)
            if game is None:
                break
            eco = game.headers.get("ECO", "Unknown")
            opening = game.headers.get("Opening", "Unknown")
            variation = game.headers.get("Variation", "")

            board = game.board()
            for mv in game.mainline_moves():
                cur_fen = board.fen()
                next_board = board.copy()
                next_board.push(mv)
                next_fen = next_board.fen()
                children_map.setdefault(cur_fen, {})[mv.uci()] = next_fen
                board.push(mv)

            final_fen = board.fen()
            terminal_meta[final_fen].append((eco, opening, variation))

    logger.info(
        "Parsed ECO tree: nodes=%d, terminals=%d", len(children_map), len(terminal_meta)
    )

    # ------------------------------------------------------------------
    # 2) Prepare Maia engines for the three variants
    # ------------------------------------------------------------------
    engine_baseline = MaiaEngine(config, use_player_embeddings=False)
    engine_ft = MaiaEngine(config, use_player_embeddings=True)

    # Create batched managers for efficient batch inference and MCTS
    mcts_manager_baseline = BatchedMCTSManager(engine_baseline, threshold=0.01)
    mcts_manager_ft = BatchedMCTSManager(engine_ft, threshold=0.01)

    variants = [
        ("maia2", engine_baseline, mcts_manager_baseline),
        ("maia2_ft", engine_ft, mcts_manager_ft),
        ("maia2_ft_mcts", engine_ft, mcts_manager_ft),
    ]

    players = list(config.data.players.values())

    # MCTS parameter used when approximating the search-improved policy
    MCTS_SIMS = 100
    OPPONENT_ELO = 2000
    # Batch sizes for prediction and MCTS runs
    PRED_BATCH_SIZE = 512
    MCTS_BATCH_SIZE = 64

    all_rows = []

    # ------------------------------------------------------------------
    # 3) For each player+variant traverse the ECO tree using breadth-first batching
    # ------------------------------------------------------------------
    players_pbar = tqdm(
        players, desc="Players", dynamic_ncols=True, position=0, unit="player"
    )
    for player in players_pbar:
        # For fairness, compute densities when the champion plays White and when they play Black
        for player_color in ["White", "Black"]:
            logger.info(
                "Computing opening densities for player: %s (%s)", player, player_color
            )

            variants_pbar = tqdm(
                variants,
                desc=f"{player} {player_color} variants",
                leave=False,
                dynamic_ncols=True,
                position=1,
                unit="variant",
            )
            for variant, engine, mcts_manager in variants_pbar:
                logger.info("  Variant: %s", variant)

                # Cache move distributions per fen to avoid repeated model calls
                dist_cache: dict[str, dict[str, float]] = {}

                # Starting layer: mapping fen -> probability mass
                root_fen = chess.Board().fen()
                current_map: dict[str, float] = {root_fen: 1.0}

                # Accumulate probability mass per (eco, opening)
                opening_mass: dict[tuple, float] = defaultdict(float)

                # Breadth-first traversal
                while current_map:
                    # 1) Distribute mass for terminal fens
                    nonterminal_fens = []
                    for fen, mass in current_map.items():
                        if fen in terminal_meta:
                            metas = terminal_meta[fen]
                            share = mass / len(metas)
                            for eco_code, opening_name, variation in metas:
                                opening_mass[(eco_code, opening_name)] += share

                        # If fen has children, it will be expanded
                        if fen in children_map and children_map[fen]:
                            nonterminal_fens.append(fen)

                    if not nonterminal_fens:
                        break

                    # 2) Evaluate child distributions in batches (unique fens)
                    unique_fens = [f for f in nonterminal_fens if f not in dist_cache]

                    # Process prediction batches for non-MCTS variants
                    if variant != "maia2_ft_mcts" and unique_fens:
                        # Evaluate in chunks to avoid OOM
                        for i in range(0, len(unique_fens), PRED_BATCH_SIZE):
                            batch_fens = unique_fens[i : i + PRED_BATCH_SIZE]
                            boards = [chess.Board(f) for f in batch_fens]

                            # Determine active elos per board
                            active_elos = []
                            for f in batch_fens:
                                is_white_to_move = chess.Board(f).turn == chess.WHITE
                                if player_color == "White":
                                    active_elo_node = (
                                        player if is_white_to_move else OPPONENT_ELO
                                    )
                                else:
                                    active_elo_node = (
                                        OPPONENT_ELO if is_white_to_move else player
                                    )
                                active_elos.append(active_elo_node)

                            probs_list, _ = mcts_manager._predict_batch(
                                boards, active_elos, opponent_elo=OPPONENT_ELO
                            )

                            for f, probs in zip(batch_fens, probs_list):
                                children = children_map.get(f, {})
                                dist = {
                                    m: float(probs.get(m, 0.0)) for m in children.keys()
                                }
                                s = sum(dist.values())
                                if s <= 0 and dist:
                                    n = len(dist)
                                    dist = {m: 1.0 / n for m in dist.keys()}
                                elif s > 0:
                                    dist = {m: v / s for m, v in dist.items()}
                                dist_cache[f] = dist

                    # Process MCTS batches
                    if variant == "maia2_ft_mcts" and unique_fens:
                        for i in range(0, len(unique_fens), MCTS_BATCH_SIZE):
                            batch_fens = unique_fens[i : i + MCTS_BATCH_SIZE]

                            # Determine active elos per board
                            active_elos = []
                            for f in batch_fens:
                                is_white_to_move = chess.Board(f).turn == chess.WHITE
                                if player_color == "White":
                                    active_elo_node = (
                                        player if is_white_to_move else OPPONENT_ELO
                                    )
                                else:
                                    active_elo_node = (
                                        OPPONENT_ELO if is_white_to_move else player
                                    )
                                active_elos.append(active_elo_node)

                            # Run batched MCTS
                            _, root_visit_list = mcts_manager.run_batch(
                                batch_fens,
                                active_elos,
                                num_simulations=MCTS_SIMS,
                                opponent_elo=OPPONENT_ELO,
                            )

                            for f, root_visit in zip(batch_fens, root_visit_list):
                                children = children_map.get(f, {})
                                dist = {
                                    m: float(root_visit.get(m, 0.0))
                                    for m in children.keys()
                                }
                                s = sum(dist.values())
                                if s <= 0 and dist:
                                    n = len(dist)
                                    dist = {m: 1.0 / n for m in dist.keys()}
                                elif s > 0:
                                    dist = {m: v / s for m, v in dist.items()}
                                dist_cache[f] = dist

                    # 3) Build next layer by expanding using computed distributions
                    next_map: dict[str, float] = defaultdict(float)
                    for fen, mass in current_map.items():
                        child_dist = dist_cache.get(fen, {})
                        if not child_dist:
                            continue
                        for mv, mv_p in child_dist.items():
                            if mv_p <= 0.0:
                                continue
                            next_fen = children_map[fen][mv]
                            next_map[next_fen] += mass * mv_p

                    current_map = next_map

                total_mass = sum(opening_mass.values())

                if total_mass <= 0:
                    # No mass computed: mark everything as Unknown with full mass
                    logger.warning(
                        "No opening mass computed for player=%s variant=%s (is the ECO tree empty?).",
                        player,
                        variant,
                    )
                    all_rows.append(
                        {
                            "player_name": player,
                            "player_color": player_color,
                            "variant": variant,
                            "eco": "Unknown",
                            "opening": "Unknown",
                            "probability": 1.0,
                        }
                    )
                else:
                    # If total_mass differs from 1, normalize and log a warning.
                    if abs(total_mass - 1.0) > 1e-6:
                        logger.warning(
                            "Total opening mass for player=%s color=%s variant=%s is %0.6f (renormalizing to 1.0)",
                            player,
                            player_color,
                            variant,
                            total_mass,
                        )

                    for (eco_code, opening_name), mass in opening_mass.items():
                        prob = float(mass) / float(total_mass)
                        all_rows.append(
                            {
                                "player_name": player,
                                "player_color": player_color,
                                "variant": variant,
                                "eco": eco_code,
                                "opening": opening_name,
                                "probability": prob,
                            }
                        )

    # ------------------------------------------------------------------
    # 4) Persist results
    # ------------------------------------------------------------------
    out_dir = config.paths.generated_data
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "opening_densities.parquet")

    df_out = pl.DataFrame(all_rows)
    df_out.write_parquet(out_path)

    logger.info("Saved opening densities to %s", out_path)

    return df_out
