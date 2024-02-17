#!/bin/bash
#SBATCH -J 50
#SBATCH -N 1
#SBATCH -n 1
#sbatch --gpus=2
#SBATCH -p jepyc
#SBATCH -o %x.out
#SBATCH -e %x.err
#SBATCH -D /mnt/lustre/ibs/dscig/kdkyum/workdir/maze

__conda_setup="$('/opt/olaf/anaconda3/2020.11/GNU/4.8/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup
conda activate pydreamer
export WANDB_MODE=offline
export PYTHONPATH='.'

HYDRA_FULL_ERROR=1 python train_epn_3d.py
