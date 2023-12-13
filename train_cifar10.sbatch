#!/usr/bin/env bash
#SBATCH --mem-per-cpu 4600M
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 8
#SBATCH --output /geoinfo_vol1/%u/run_logs/train_cifar10_%J.log
#SBATCH --error  /geoinfo_vol1/%u/run_logs/train_cifar10_%J.err


echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
. /geoinfo_vol1/dkerekes/miniforge3/etc/profile.d/conda.sh

conda activate packed_ensembles

files=("$@")

PYTHONUNBUFFERED=1; python3 train_cifar10.py -c "${files[$SLURM_ARRAY_TASK_ID]}" -o output -d data