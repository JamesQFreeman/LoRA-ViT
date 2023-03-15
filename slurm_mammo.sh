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
TYPE=$1

python train_kfold.py -bs 32 \
                -data_path '/public_bme/data/INBreast/' \
                -lr 2e-3 \
                -epochs 10 \
                -train_type lora \
                -r 4 \
                

python train_kfold.py -bs 32 \
                -data_path '/public_bme/data/INBreast/' \
                -lr 2e-3 \
                -epochs 10 \
                -train_type linear \
                -r 4 \


python train_kfold.py -bs 32 \
                -data_path '/public_bme/data/INBreast/' \
                -lr 3e-5 \
                -epochs 10 \
                -train_type full \
                -r 4 \

# python train_kfold.py -bs 32 \
#                 -data_path '/public_bme/data/INBreast/' \
#                 -lr 8e-6 \
#                 -epochs 20 \
#                 -train_type ${TYPE} \
#                 -r 4 \