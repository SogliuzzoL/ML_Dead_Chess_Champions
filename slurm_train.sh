#!/bin/bash
#SBATCH --job-name=chess_eval
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --partition=batch          # La partition correcte pour Lyra
#SBATCH --gpus=1                   # Une GPU par nœud sur Lyra
#SBATCH --cpus-per-task=16         # 16 cœurs CPU pour le multi-processing MCTS
#SBATCH --mem=100G                 # 100 Go de RAM
#SBATCH --time=4-00:00:00

# Se placer dans le dossier du projet
cd $SLURM_SUBMIT_DIR

# Exposer uv
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Redirection du cache vers le Global Scratch
export UV_CACHE_DIR="$GLOBALSCRATCH/.cache/uv"

# Chargement de l'environnement CUDA
module purge
module load CUDA

# Synchronisation de l'environnement
uv sync

echo "=========================================================="
echo " DÉMARRAGE DU JOB : $(date)"
echo " Nœud             : $SLURM_JOB_NODELIST"
echo "=========================================================="

echo "Lancement de l'évaluation ..."
uv run main.py evaluate_players

echo "=========================================================="
echo " JOB TERMINÉ : $(date)"
echo "=========================================================="
