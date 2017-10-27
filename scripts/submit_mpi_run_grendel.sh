#!/bin/bash
#SBATCH --job-name=tessphot
#SBATCH --partition=q20
#SBATCH --constraint=astro
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --export=NONE
#SBATCH --time=6:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rasmush@phys.au.dk

echo "========= Job started  at `date` =========="

# Load required modules and avtivate the virtualenv:
source /comm/swstack/bin/modules.sh --silent
module load intel
module load openmpi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
source ~/python-virtualenv/tessphot/bin/activate

# Set environment variables with paths for input and
# output files for the photometry code:
export TESSPHOT_INPUT="/scratch/astro/tess/input"
export TESSPHOT_OUTPUT="/scratch/astro/tess/output"

# Move the program to the scratch disk:
rsync -a --delete ~/tasoc/photometry/ /scratch/astro/tess/program/

# Change directory to the local scratch-directory:
cd /scratch/astro/tess/program

# Run the MPI job:
mpiexec python mpi_scheduler.py > out.txt 2>&1

# Copy some of the output to the home directory:
mv out.txt ~/tasoc/output-slurm/
mv slurm-*.out ~/tasoc/output-slurm/
cp /scratch/astro/tess/input/todo.sqlite ~/tasoc/output-slurm/
mv /scratch/astro/tess/output/* ~/tasoc/output-fits/

echo "========= Job finished at `date` =========="
#
