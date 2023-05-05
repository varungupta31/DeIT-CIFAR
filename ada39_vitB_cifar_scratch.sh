#!/bin/bash
#SBATCH -A varungupta
#SBATCH -n 1
#SBATCH -w gnode039
#SBATCH --gres gpu:1
#SBATCH --mem=10G
#SBATCH --time=INFINITE
#SBATCH --mail-type=END
#SBATCH --mail-user=varungupta.iiith@gmail.com

conda init --all
source activate smai
module load u18/cudnn/8.4.0-cuda-11.6
module load u18/cuda/11.6

python train_vit_scratch.py --config configs/vit_b_scratch_cifar.yaml
