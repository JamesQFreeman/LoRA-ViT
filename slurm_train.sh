#!/bin/bash
#SBATCH -p bme_gpu
#SBATCH --job-name=LoRA_NIH
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH -t 5-00:00:00
source activate idea
nvidia-smi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python train.py -bs 128 \
                -data_path '/public_bme/data/NIH_X-ray/' \
                -lr 1e-3 \
                -epochs 20 \
                -train_type 'resnet50' \
                -r 4 \

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python train.py -bs 128 \
#                 -data_path '/public_bme/data/NIH_X-ray/' \
#                 -lr 3e-5 \
#                 -epochs 20 \
#                 -train_type 'full' \
#                 -r 4 \

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python train.py -bs 128 \
#                 -data_path '/public_bme/data/NIH_X-ray/' \
#                 -lr 3e-5 \
#                 -epochs 20 \
#                 -train_type 'lora' \
#                 -r 4 \


# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python train.py -bs 128 \
#                 -data_path '/public_bme/data/NIH_X-ray/' \
#                 -lr 3e-4 \
#                 -epochs 20 \
#                 -train_type 'linear' \
#                 -r 4 \