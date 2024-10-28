#!/bin/bash -x

#SBATCH --mem=32G
#SBATCH --qos=high     
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=1-00:00:00


module load Python3/3.11.2
# activate you virtual environment or conda
source "virtual env dir"/bin/activate
# the next 3 lines install the fastmax library
module load gcc
module load cuda
python "dir were you put fastmax files"/fastmax/setup.py install
# your commands
litgpt pretrain pythia-14m \
   --config "dir were you put e.g. litgpt"/litgpt/config_hub/pretrain/debug.yaml
