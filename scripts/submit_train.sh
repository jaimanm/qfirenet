#!/bin/bash
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:h100:1
#SBATCH -t 24:00:00
#SBATCH --partition=gpu-h100
#SBATCH --account=hpcintro-aac
#SBATCH --output=experiments/slurm-%j.out

echo "Running $SLURM_NTASKS tasks ($SLURM_CPUS_PER_TASK cores) on $SLURM_NODELIST"

module purge
module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-x86_64

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH

cd "$SLURM_SUBMIT_DIR"

# Pass all command-line arguments through to train.py
# Usage: sbatch scripts/submit_train.sh --config configs/classical_baseline.yaml
python3 train.py "$@"
