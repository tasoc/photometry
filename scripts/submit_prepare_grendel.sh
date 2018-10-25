#!/bin/bash
#SBATCH --job-name=prepare
#SBATCH --partition=q36
#SBATCH --constraint=astro
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=0
#SBATCH --export=NONE
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rasmush@phys.au.dk
#SBATCH --output=/home/rasmush/tasoc/slurm-%j-prepare.out

echo "========= Job started  at `date` =========="

# Load required modules and avtivate the virtualenv:
source /comm/swstack/bin/modules.sh --silent
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
echo "Using $OMP_NUM_THREADS CPUs"
source ~/python-virtualenv/tessphot/bin/activate

# Set environment variables with paths for input and
# output files for the photometry code:
export TESSPHOT_INPUT="/fscratch/astro/tess/input"
export TESSPHOT_OUTPUT="/scratch/astro/tess/output"

# Move the program to the scratch disk:
rsync -a --delete ~/tasoc/photometry/ /fscratch/astro/tess/program/

# Change directory to the local scratch-directory:
cd /fscratch/astro/tess/program

# Run the MPI job:
echo "Running prepare..."
python prepare_photometry.py

echo "Running make_todo..."
python make_todo.py

# Copy some of the output to the home directory:
cp /fscratch/astro/tess/input/todo.sqlite ~/tasoc/input/

echo "========= Job finished at `date` =========="
#
