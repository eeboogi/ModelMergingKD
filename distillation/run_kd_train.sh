#!/bin/bash
#SBATCH --job-name=kd_train_mnist_to_svhn
#SBATCH --output=kd_train_mnist_to_svhn_%j.out
#SBATCH --error=kd_train_mnist_to_svhn_%j.err
#SBATCH --partition=hoffman-lab
#SBATCH --qos="short"
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --gpus-per-node="a40:1"
#SBATCH --cpus-per-task=14
#SBATCH --ntasks-per-node=1
#SBATCH --exclude="xaea-12"

export PYTHONUNBUFFERED=TRUE
source ~/.bashrc
conda activate model-merging
cd /nethome/becsedi3/flash/develop/ModelMerging

srun -u python -m distillation.kd_train_2