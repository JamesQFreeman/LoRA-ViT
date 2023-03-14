#!/bin/bash
#SBATCH -p bme_gpu
#SBATCH --job-name=LoRA_INbreast
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH -t 5-00:00:00
source activate idea
nvidia-smi

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python train_kfold.py -bs 32 \
                -data_path '/public_bme/data/INBreast/' \
                -lr 1e-3 \
                -epochs 20 \
                -train_type "linear" \
                -r 4 \
                
# # PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python train_kfold.py -bs 32 \
#                 -data_path '/public_bme/data/INBreast/' \
#                 -lr 6e-5 \
#                 -epochs 20 \
#                 -train_type "full" \
#                 -r 4 \

# # PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python train_kfold.py -bs 32 \
#                 -data_path '/public_bme/data/INBreast/' \
#                 -lr 1e-3 \
#                 -epochs 20 \
#                 -train_type 'lora' \
#                 -r 4 \
