#!/bin/bash
#SBATCH --job-name=transfer
#SBATCH --partition=q36
#SBATCH --constraint=astro
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --export=NONE
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rasmush@phys.au.dk
#SBATCH --output=/home/rasmush/tasoc/slurm-%j-transfer.out

echo "========= Job started  at `date` =========="

# Load required modules and avtivate the virtualenv:
#source /comm/swstack/bin/modules.sh --silent

# Change directory to the local scratch-directory:
cd /scratch/astro/tess/input/

# Move the program to the scratch disk:
rsync -av --progress --stats --log-file=~/tasoc/transfer.log ~/tasoc/input/ .

echo "========= Job finished at `date` =========="
#
