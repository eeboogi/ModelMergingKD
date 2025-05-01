# Reducing Redundancy and Introducing Task-Aware Guidance via Knowledge Distillation Prior to Merging LoRA Expert Models

This repository contains code for distilling large ViT LoRA expert models into smaller ViT models and merging them using KnOTS-TIES with the aim to investigate how distillation from a related task affects model merging performance.

## Experiment Overview

The experiment:
1. Distills knowledge from a CLIP-ViT-L/14 model fine-tuned on MNIST to a CLIP-ViT-B/32 model on SVHN
2. Merges the distilled SVHN model with 7 other vision task models
3. Compares performance against merges using the original baseline model fine-tuned on SVHN (without distillation)

## Prerequisites

- Linux cluster with SLURM scheduler
- NVIDIA GPU (tested with A40)
- Python 3.9+
- Conda package manager

## Setup

1. Clone the repository:

git clone https://github.com/eeboogi/ModelMergingKD.git


2. Create conda environment:

conda create -n model-merging --file ./distillation/requirements.txt
conda activate model-merging


3. Install dependencies:

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 –extra-index-url https://download.pytorch.org/whl/cu117pip install peft==0.3.0 transformers==4.28.1 wandb numpy==1.23.5


## Directory Structure

ModelMergingKD/├── distillation/│   ├── multidataset_vector_merging_distill.py  # Main merging script│   └── submit_distill_merge.sh                 # SLURM submission script├── configs/                   # Model configuration files├── checkpoints/               # Pre-trained models├── csvs/                      # Evaluation results└── utils/                     # Helper functions


## Running the Experiment

### 1. Prepare Models
Ensure model checkpoints are available at specified directory. Models can be downloaded from: https://huggingface.co/collections/hoffman-lab/knots-model-merging-with-svd-672d3b5fabf766c22989e760


### 2. Submit SLURM Job

sbatch distillation/submit_distill_merge.sh



For questions, contact: [boglarka.ecsedi@gatech.edu](mailto:boglarka.ecsedi@gatech.edu)
