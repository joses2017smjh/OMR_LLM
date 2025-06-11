#!/bin/bash

#SBATCH --partition=dgxh
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL

# sq -u negreanj
# ssh ...
# ml cuda/12.4 cudnn
# export CUDA_VISIBLE_DEVICES=0

source /nfs/stak/users/negreanj/hpc-share/nlp-hw2/bin/activate

python generate.py

# srun -A ai539 --partition=dgxh --mem=32G --gres=gpu:2g.20gb:1 --pty bash