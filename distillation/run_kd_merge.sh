#!/bin/bash
#SBATCH --job-name=kd_pairwise_merge_distilled
#SBATCH --output=kd_merge_distilled_%j.out
#SBATCH --error=kd_merge_distilled_%j.err
#SBATCH --partition=overcap
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

srun -u python -m distillation.multidataset_vector_merging_distill