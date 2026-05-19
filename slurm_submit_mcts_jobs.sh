#!/bin/bash
# Submit MCTS grid-search jobs to SLURM using CLI arguments (no env vars).
#
# Usage:
#   1) Edit the GRID arrays below (NUM_SIM_LIST, C_PUCT_LIST, THR_LIST) if needed.
#   2) Make executable: chmod +x slurm_submit_mcts_jobs.sh
#   3) Run: ./slurm_submit_mcts_jobs.sh
#
# This script submits TWO jobs per grid cell by default:
#   - one using the fine-tuned per-player embeddings (FT),
#   - one WITHOUT the player embeddings (base Maia backbone),
# so you don't need to edit the script to compare both variants.
#
# Resource coherence and safety decisions (defaults chosen to be conservative):
# - We set MCTS worker count to 1 by default to avoid multiple processes loading
#   separate copies of the model on the same GPU which would cause GPU OOMs.
# - We allocate a small number of CPU cores per task (4) to allow modest
#   multiprocessing/IO without wasting cores.
# - Default batch size is conservative (128) to limit per-process GPU memory.
# - Adjust these via CLI options below if you know your hardware can handle more.

# -------------------------
# User-configurable grid
# -------------------------
NUM_SIM_LIST=(50 100 200 1000)
C_PUCT_LIST=(0.5 1.0 1.5 2.5)
THR_LIST=(0.0 0.01 0.05 0.1)

# Fraction of test set to use (1.0 = full set)
SUBSAMPLE_FRAC=1.0

# Per-job resources (conservative defaults)
PARTITION="batch"
GPUS=1
CPUS_PER_TASK=4   # CPUs allocated to the job on the node
MEM="16G"
TIME="4-00:00:00"

# MCTS internal params (safe defaults)
MCTS_NUM_WORKERS=1   # number of processes that will call the model (1 avoids multiple GPU copies)
MCTS_BATCH_SIZE=128  # batch size per worker

# Project paths (adjust if your layout differs)
PROJECT_ROOT="/home/sogliuzzol/Projects/ChessBehaviors"
CONFIG_PATH="config/default.yml"
LOG_DIR="$PROJECT_ROOT/logs/mcts"
mkdir -p "$LOG_DIR"

# Optional flags controlling which variants to submit. By default submit both.
ONLY_FT=0
ONLY_NOFT=0

# Optional: allow overriding via script args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only-ft)
      ONLY_FT=1; shift ;;
    --only-noft)
      ONLY_NOFT=1; shift ;;
    --subsample)
      SUBSAMPLE_FRAC="$2"; shift 2 ;;
    --partition)
      PARTITION="$2"; shift 2 ;;
    --gpus)
      GPUS="$2"; shift 2 ;;
    --cpus)
      CPUS_PER_TASK="$2"; shift 2 ;;
    --mem)
      MEM="$2"; shift 2 ;;
    --time)
      TIME="$2"; shift 2 ;;
    --mcts-workers)
      MCTS_NUM_WORKERS="$2"; shift 2 ;;
    --mcts-batch-size)
      MCTS_BATCH_SIZE="$2"; shift 2 ;;
    *)
      echo "Unknown option $1"; exit 1 ;;
  esac
done

if [[ "$ONLY_FT" -eq 1 && "$ONLY_NOFT" -eq 1 ]]; then
  echo "Conflicting options: --only-ft and --only-noft cannot both be set."; exit 1
fi

# Sanity checks / warnings
if [[ $MCTS_NUM_WORKERS -gt 1 && $GPUS -lt $MCTS_NUM_WORKERS ]]; then
  echo "Warning: MCTS_NUM_WORKERS=$MCTS_NUM_WORKERS but allocated GPUS=$GPUS. Multiple workers will share the same GPU and may OOM. Consider increasing --gpus or reducing --mcts-workers." >&2
fi

# Submit jobs
count=0
for sim in "${NUM_SIM_LIST[@]}"; do
  for c in "${C_PUCT_LIST[@]}"; do
    for thr in "${THR_LIST[@]}"; do

      # Compose a safe name suffix for floats
      safe_c=$(echo "$c" | tr '.' '_')
      safe_thr=$(echo "$thr" | tr '.' '_')

      # FT variant (use player embeddings)
      if [[ "$ONLY_NOFT" -eq 0 ]]; then
        count=$((count + 1))
        JOB_NAME="mcts_sim${sim}_c${safe_c}_thr${safe_thr}_ft"
        OUT_LOG="$LOG_DIR/${JOB_NAME}.out"
        ERR_LOG="$LOG_DIR/${JOB_NAME}.err"

        CLI_ARGS=(evaluate_mcts_params --config "$CONFIG_PATH" --mcts-num-sim "$sim" --mcts-c-puct "$c" --mcts-threshold "$thr" --mcts-subsample-frac "$SUBSAMPLE_FRAC" --mcts-num-workers "$MCTS_NUM_WORKERS" --mcts-batch-size "$MCTS_BATCH_SIZE")
        # FT variant: do NOT add --mcts-disable-player-embeddings

        WRAP_CMD="cd $PROJECT_ROOT && uv run main.py --config $CONFIG_PATH ${CLI_ARGS[*]}"

        sbatch --job-name="$JOB_NAME" \
               --output="$OUT_LOG" \
               --error="$ERR_LOG" \
               --partition="$PARTITION" \
               --gpus=$GPUS \
               --cpus-per-task=$CPUS_PER_TASK \
               --mem=$MEM \
               --time=$TIME \
               --wrap="$WRAP_CMD"

        echo "Submitted job #$count: $JOB_NAME"
      fi

      # no-FT variant (disable player embeddings)
      if [[ "$ONLY_FT" -eq 0 ]]; then
        count=$((count + 1))
        JOB_NAME="mcts_sim${sim}_c${safe_c}_thr${safe_thr}_noft"
        OUT_LOG="$LOG_DIR/${JOB_NAME}.out"
        ERR_LOG="$LOG_DIR/${JOB_NAME}.err"

        CLI_ARGS=(evaluate_mcts_params --config "$CONFIG_PATH" --mcts-num-sim "$sim" --mcts-c-puct "$c" --mcts-threshold "$thr" --mcts-subsample-frac "$SUBSAMPLE_FRAC" --mcts-num-workers "$MCTS_NUM_WORKERS" --mcts-batch-size "$MCTS_BATCH_SIZE" --mcts-disable-player-embeddings)

        WRAP_CMD="cd $PROJECT_ROOT && uv run main.py --config $CONFIG_PATH ${CLI_ARGS[*]}"

        sbatch --job-name="$JOB_NAME" \
               --output="$OUT_LOG" \
               --error="$ERR_LOG" \
               --partition="$PARTITION" \
               --gpus=$GPUS \
               --cpus-per-task=$CPUS_PER_TASK \
               --mem=$MEM \
               --time=$TIME \
               --wrap="$WRAP_CMD"

        echo "Submitted job #$count: $JOB_NAME"
      fi

    done
  done
done

echo "Submitted $count jobs. Logs in $LOG_DIR"
