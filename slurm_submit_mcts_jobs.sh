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
#
# Notes about GPUs and SBATCH options:
# - Different clusters accept different SBATCH flags for GPUs. Some accept
#   `--gpus=1`, some `--gres=gpu:1`, others require a specific device name.
# - By default this script does NOT request a partition so the scheduler will
#   use the default partition. You can pass --partition <name> to select one.

# -------------------------
# User-configurable grid
# -------------------------
NUM_SIM_LIST=(50 100 200 1000)
C_PUCT_LIST=(0.5 1.0 1.5 2.5)
THR_LIST=(0.0 0.01 0.05 0.1)

# Fraction of test set to use (1.0 = full set)
SUBSAMPLE_FRAC=1.0

# Per-job resources (conservative defaults)
PARTITION=""   # empty => don't pass --partition, let scheduler use default
# To request GPUs pass --gpu-arg "--gres=gpu:1" when calling the script
GPU_SBATCH_ARG=""
CPUS_PER_TASK=4   # CPUs allocated to the job on the node
MEM="64G"
TIME="4-00:00:00"

# MCTS internal params (safe defaults)
MCTS_NUM_WORKERS=1   # number of processes that will call the model (1 avoids multiple GPU copies)
MCTS_BATCH_SIZE=128  # batch size per worker

# Project paths: use current working directory as project root by default
PROJECT_ROOT="$(pwd)"
CONFIG_PATH="config/default.yml"
LOG_DIR="$PROJECT_ROOT/logs/mcts"
mkdir -p "$LOG_DIR" 2>/dev/null || true

# Optional flags controlling which variants to submit. By default submit both.
ONLY_FT=0
ONLY_NOFT=0
NO_GPU=0

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
    --gpu-arg)
      GPU_SBATCH_ARG="$2"; shift 2 ;;
    --no-gpu)
      NO_GPU=1; shift ;;
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

# If NO_GPU requested, clear GPU SBATCH arg
if [[ "$NO_GPU" -eq 1 ]]; then
  GPU_SBATCH_ARG=""
fi

# Sanity checks / warnings
if [[ -n "$GPU_SBATCH_ARG" && $MCTS_NUM_WORKERS -gt 1 ]]; then
  echo "Warning: MCTS_NUM_WORKERS=$MCTS_NUM_WORKERS and GPU requested. Multiple workers may share the GPU and cause OOM. Consider setting --mcts-workers 1 or request more GPUs via --gpu-arg." >&2
fi

# Prepare SBATCH GPU option tokens (split into array for safe expansion)
IFS=' ' read -r -a GPU_ARG_TOKENS <<< "$GPU_SBATCH_ARG"

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
        JOB_NAME="mcts_sim${sim}_c${safe_c}_thr${safe_thr}_ft"
        OUT_LOG="$LOG_DIR/${JOB_NAME}.out"
        ERR_LOG="$LOG_DIR/${JOB_NAME}.err"

        CLI_ARGS=(evaluate_mcts_params --config "$CONFIG_PATH" --mcts-num-sim "$sim" --mcts-c-puct "$c" --mcts-threshold "$thr" --mcts-subsample-frac "$SUBSAMPLE_FRAC" --mcts-num-workers "$MCTS_NUM_WORKERS" --mcts-batch-size "$MCTS_BATCH_SIZE")
        # FT variant: do NOT add --mcts-disable-player-embeddings

        WRAP_CMD="cd '$PROJECT_ROOT' && uv run main.py --config '$CONFIG_PATH' ${CLI_ARGS[*]}"

        # Build sbatch command as array to safely include optional partition/gpu tokens
        SBATCH_CMD=(sbatch --job-name="$JOB_NAME" --output="$OUT_LOG" --error="$ERR_LOG")
        if [[ -n "$PARTITION" ]]; then
          SBATCH_CMD+=(--partition="$PARTITION")
        fi
        if [[ ${#GPU_ARG_TOKENS[@]} -gt 0 && -n "${GPU_ARG_TOKENS[0]}" ]]; then
          SBATCH_CMD+=("${GPU_ARG_TOKENS[@]}")
        fi
        SBATCH_CMD+=(--cpus-per-task=$CPUS_PER_TASK --mem=$MEM --time=$TIME --wrap "$WRAP_CMD")

        SBATCH_OUT=$("${SBATCH_CMD[@]}" 2>&1)
        SBATCH_RC=$?
        if [[ $SBATCH_RC -eq 0 ]]; then
          echo "Submitted job: $JOB_NAME - sbatch: $SBATCH_OUT"
          count=$((count + 1))
        else
          echo "Failed to submit $JOB_NAME: $SBATCH_OUT" >&2
        fi
      fi

      # no-FT variant (disable player embeddings)
      if [[ "$ONLY_FT" -eq 0 ]]; then
        JOB_NAME="mcts_sim${sim}_c${safe_c}_thr${safe_thr}_noft"
        OUT_LOG="$LOG_DIR/${JOB_NAME}.out"
        ERR_LOG="$LOG_DIR/${JOB_NAME}.err"

        CLI_ARGS=(evaluate_mcts_params --config "$CONFIG_PATH" --mcts-num-sim "$sim" --mcts-c-puct "$c" --mcts-threshold "$thr" --mcts-subsample-frac "$SUBSAMPLE_FRAC" --mcts-num-workers "$MCTS_NUM_WORKERS" --mcts-batch-size "$MCTS_BATCH_SIZE" --mcts-disable-player-embeddings)

        WRAP_CMD="cd '$PROJECT_ROOT' && uv run main.py --config '$CONFIG_PATH' ${CLI_ARGS[*]}"

        SBATCH_CMD=(sbatch --job-name="$JOB_NAME" --output="$OUT_LOG" --error="$ERR_LOG")
        if [[ -n "$PARTITION" ]]; then
          SBATCH_CMD+=(--partition="$PARTITION")
        fi
        if [[ ${#GPU_ARG_TOKENS[@]} -gt 0 && -n "${GPU_ARG_TOKENS[0]}" ]]; then
          SBATCH_CMD+=("${GPU_ARG_TOKENS[@]}")
        fi
        SBATCH_CMD+=(--cpus-per-task=$CPUS_PER_TASK --mem=$MEM --time=$TIME --wrap "$WRAP_CMD")

        SBATCH_OUT=$("${SBATCH_CMD[@]}" 2>&1)
        SBATCH_RC=$?
        if [[ $SBATCH_RC -eq 0 ]]; then
          echo "Submitted job: $JOB_NAME - sbatch: $SBATCH_OUT"
          count=$((count + 1))
        else
          echo "Failed to submit $JOB_NAME: $SBATCH_OUT" >&2
        fi
      fi

    done
  done
done

echo "Submitted $count jobs. Logs in $LOG_DIR"
