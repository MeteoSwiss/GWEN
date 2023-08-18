#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:4
#SBATCH --job-name=WGN
#SBATCH --time=23:59:00
#SBATCH --partition=normal
#SBATCH --account=s83
#SBATCH --output=logs/%j.out

source /scratch/e1000/meteoswiss/scratch/sadamov/mambaforge/envs/weathergraphnet_alps/bin/activate
srun python src/weathergraphnet/train_gnn.py
