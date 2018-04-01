#!/bin/bash
#SBATCH --job-name=tessphot
#SBATCH --partition=q36
#SBATCH --constraint=astro
#SBATCH --ntasks=36
#SBATCH --ntasks-per-node=36
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --export=NONE
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rasmush@phys.au.dk
#SBATCH --output=/home/rasmush/tasoc/slurm-%j-tessphot.out

echo "========= Job started  at `date` =========="

# Load required modules and avtivate the virtualenv:
source /comm/swstack/bin/modules.sh --silent
module load intel
module load openmpi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
source ~/python-virtualenv/tessphot/bin/activate
echo "Using $OMP_NUM_THREADS CPUs"

# Set environment variables with paths for input and
# output files for the photometry code:
export TESSPHOT_INPUT="/scratch/astro/tess/input"
export TESSPHOT_OUTPUT="/scratch/astro/tess/output"

# Move the program to the scratch disk:
rsync -a --delete ~/tasoc/photometry/ /scratch/astro/tess/program/

# Change directory to the local scratch-directory:
cd /scratch/astro/tess/program

# Run the MPI job:
mpiexec python mpi_scheduler.py

# Copy some of the output to the home directory:
cp /scratch/astro/tess/input/todo.sqlite ~/tasoc/

echo "========= Job finished at `date` =========="
#
