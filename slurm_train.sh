#!/bin/bash
#SBATCH --job-name=chess_eval
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --partition=batch          # La partition correcte pour Lyra
#SBATCH --gpus=1                   # Une GPU par nœud sur Lyra
#SBATCH --cpus-per-task=4          # 4 cœurs CPU pour le multi-processing MCTS
#SBATCH --mem=32G                  # 32 Go de RAM
#SBATCH --time=4-00:00:00

# Se placer dans le dossier du projet
cd $SLURM_SUBMIT_DIR

# Exposer uv
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Redirection du cache vers le Global Scratch
export UV_CACHE_DIR="$GLOBALSCRATCH/.cache/uv"

# Set threading/env vars early so child processes inherit them
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# Disable jemalloc background threads
export MALLOC_CONF="background_thread:false"

# (Optionnel) augmenter la limite de processus si ton admin/cluster l'autorise
# ulimit -u 4096

# Chargement de l'environnement CUDA (préciser version si nécessaire)
module purge
module load CUDA

# Synchronisation de l'environnement
uv sync

echo "=========================================================="
echo " DÉMARRAGE DU JOB : $(date)"
echo " Nœud             : $SLURM_JOB_NODELIST"
echo "=========================================================="

echo "Lancement de l'évaluation ..."
uv run main.py evaluate_mcts_params

echo "=========================================================="
echo " JOB TERMINÉ : $(date)"
echo "=========================================================="
