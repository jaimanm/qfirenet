#!/bin/bash
#SBATCH -n 1
#SBATCH -c 12
#SBATCH -t 2:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100
#SBATCH --output=experiments/slurm-%j.out

echo "Running $SLURM_NTASKS tasks ($SLURM_CPUS_PER_TASK cores) on $SLURM_NODELIST"

module purge
module load cuda cudnn python

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

WORKDIR="$HOME/scratch/wildfires"
cd "$WORKDIR"

# Usage: sbatch scripts/submit_test.sh --config configs/classical_baseline.yaml --restore_from experiments/.../best_model.pth
python test.py "$@"
