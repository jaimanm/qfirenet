#!/bin/bash
#SBATCH -n 1
#SBATCH -c 12
#SBATCH -t 2:30:00
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --mem=32G
#SBATCH --output=experiments/slurm-%j.out

echo "Running $SLURM_NTASKS tasks ($SLURM_CPUS_PER_TASK cores) on $SLURM_NODELIST"

module purge
module load cuda cudnn

# Initialize the shared miniconda environment
source ~/scratch.hpcintro-shared/bin/condainit.sh quantum

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Use SLURM_SUBMIT_DIR so it automatically starts wherever you ran `sbatch` from
WORKDIR="$SLURM_SUBMIT_DIR"
cd "$WORKDIR"

# Pass all command-line arguments through to train.py
# Usage: sbatch scripts/submit_train.sh --config configs/classical_baseline.yaml
python train.py "$@"
