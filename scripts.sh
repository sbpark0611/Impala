#!/bin/sh
#SBATCH -J IMPALA
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH -p long
#SBATCH -o %x.out
#SBATCH -e %x.err
#SBATCH -D /proj/internal_group/dscig/kdkyum/workdir/Impala

__conda_setup="$('/opt/olaf/anaconda3/2020.11/GNU/4.8/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup
conda activate pydreamer
export WANDB_MODE=offline
export PYTHONPATH='.'

python launch.py