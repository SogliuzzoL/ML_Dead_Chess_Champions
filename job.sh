#!/bin/bash
#SBATCH --job-name=chess_train
#SBATCH --output=%j_train.out
#SBATCH --partition=batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
module purge
module load releases/2022b
module load Python/3.10.8-GCCcore-12.2.0

pip install -r requirements.txt
srun python src/run_players.py --train 1
