#!/bin/bash
#SBATCH -n 1
#SBATCH -c 12
#SBATCH -t 2:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --account=hpcintro-aac
#SBATCH --output=experiments/slurm-%j.out

echo "Running $SLURM_NTASKS tasks ($SLURM_CPUS_PER_TASK cores) on $SLURM_NODELIST"

module purge
module load cuda/12.3.0/gcc/11.3.0/zen2
module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH

cd "$SLURM_SUBMIT_DIR"

# Pass all command-line arguments through to train.py
# Usage: sbatch scripts/submit_train.sh --config configs/classical_baseline.yaml
python3 train.py "$@"
