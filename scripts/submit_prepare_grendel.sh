#!/bin/bash
#SBATCH --job-name=tessphot
#SBATCH --partition=q20
#SBATCH --constraint=astro
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=0
#SBATCH --export=NONE
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rasmush@phys.au.dk

echo "========= Job started  at `date` =========="

# Load required modules and avtivate the virtualenv:
source /comm/swstack/bin/modules.sh --silent
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
echo "Using {$OMP_NUM_THREADS} CPUs"
source ~/python-virtualenv/tessphot/bin/activate

# Set environment variables with paths for input and
# output files for the photometry code:
export TESSPHOT_INPUT="/scratch/astro/tess/input"
export TESSPHOT_OUTPUT="/scratch/astro/tess/output"

# Move the program to the scratch disk:
rsync -a --delete ~/tasoc/photometry/ /scratch/astro/tess/program/

rsync -a --stats ~/tasoc/input/ /scratch/astro/tess/input/

# Change directory to the local scratch-directory:
cd /scratch/astro/tess/program

# Run the MPI job:
echo "Running prepare..."
python prepare_photometry.py --debug 14

echo "Running make_todo..."
python make_todo.py

# Copy some of the output to the home directory:
#mv prepare-out.txt ~/tasoc/output-slurm/
#mv slurm-*.out ~/tasoc/output-slurm/

echo "========= Job finished at `date` =========="
#
