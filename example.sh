#!/bin/bash -x

#SBATCH --mem=32G
#SBATCH --qos=high     
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=1-00:00:00


module load Python3/3.11.2
source /fs/nexus-scratch/agerami/litgpt/.venv/bin/activate
module load gcc
module load cuda
python ./fastmax/setup.py install
litgpt pretrain pythia-14m \
   --config /fs/nexus-scratch/agerami/litgpt/config_hub/pretrain/debug.yaml