#!/bin/bash
#SBATCH -p bme_gpu
#SBATCH --job-name=w
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH -t 5-00:00:00
source activate idea
nvidia-smi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python loadWeight.py

