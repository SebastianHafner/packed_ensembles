#!/usr/bin/env bash
#SBATCH --mem-per-cpu 4600M
#SBATCH --gres gpu:2
#SBATCH --cpus-per-task 16
#SBATCH --output /geoinfo_vol1/%u/run_logs/train_cifar10_%J.log
#SBATCH --error  /geoinfo_vol1/%u/run_logs/train_cifar10_%J.err


echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
. /geoinfo_vol1/dkerekes/miniforge3/etc/profile.d/conda.sh

conda activate packed_ensembles

PYTHONUNBUFFERED=1; python3 train_cifar10.py -c $1 -o output -d data -r $SLURM_ARRAY_TASK_ID