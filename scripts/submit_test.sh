#!/bin/bash
## Notes: - comments start with '#' ;
## - the first line of the script must be a 'shebang' line (i.e. '#!' followed
##     by the shell flavor for interpreting this script) ;
## - then follow Slurm sbatch directives starting with '#SBATCH'
##
#SBATCH -n 1                 # request 1 main task
#SBATCH -c 8                 # number of cores per task (threads)
#SBATCH -t 2:30:00           # max. run time ('Walltime') in format d-HH:MM:SS
## or: #SBATCH -t MMMM  #max.time in minutes
# #SBATCH --mail-type=ALL  #send a mail when job starts and ends
##
## - you may change the number of cores or walltime
## - also if you need more memory than the default 4GB/core, you may uncomment:
# #SBATCH --mem-per-cpu=8192  #or =8G or other value in MB
##
## - to run on a GPU node:
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
## - to run on an A100 instead, replace the above two lines with:
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:a100
##
#SBATCH --account=hpcintro-aac
#SBATCH --output=experiments/slurm-%j.out
##
## - some printout:
echo "Running $SLURM_NTASKS tasks ($SLURM_CPUS_PER_TASK cores per task) on $SLURM_NODELIST"

## - prevent any loaded software to interfere
module purge

## - load the software environment
##   set mysoftwaresrc to a conda env name to activate it,
##   or leave empty to use system module + user-installed packages
mysoftwaresrc=py3.13q
if [ "X$mysoftwaresrc" = "X" ]; then
  ## system module + pip --user packages (requires torch cu121 installed via pip --user)
  module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-x86_64
  export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH
else
  source ~/scratch.hpcintro-shared/bin/condainit.sh $mysoftwaresrc
fi

## - limit the openMP threadpool to the number of allocated cores
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

## - define working directory (default is the folder where you start the job)
WORKDIR=$SLURM_SUBMIT_DIR
[ -d $WORKDIR ] || mkdir -p $WORKDIR
cd $WORKDIR

## - optional: get timing information (first set counter to 0):
SECONDS=0

## - run the inference script
## Usage: sbatch scripts/submit_test.sh --config configs/classical_baseline.yaml --restore_from experiments/.../best_model.pth
python3 test.py "$@"

## - optional: get timing information and print out:
Tend=$SECONDS
Ncores=`expr $SLURM_NTASKS \* $SLURM_CPUS_PER_TASK`
echo "Running for $Tend seconds on $Ncores cores on $SLURM_NODELIST"
