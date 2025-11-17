#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o /mnt/home/rmiranda/all/repos/load_tms/slurm/logs/%j-%x.log
#SBATCH --job-name=nhits
#SBATCH --nodelist=gpu3.omnia.cluster
#SBATCH --partition=h100
 
eval "$(conda shell.bash hook)"
conda activate tms
 
cd /mnt/home/rmiranda/all/repos/load_tms/run/

srun python3 train.py \
    model_params.default=true \
    params.n_samples=2



echo "TMS Train and Test Completed"
