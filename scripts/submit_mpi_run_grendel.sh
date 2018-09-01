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
echo "Using $OMP_NUM_THREADS CPUs per task"

# Set environment variables with paths for input and
# output files for the photometry code:
export TESSPHOT_INPUT="/fscratch/astro/tess/input"
export TESSPHOT_OUTPUT="/home/rasmush/tasoc/output-fits"

# Move the program to the scratch disk:
rsync -a --delete ~/tasoc/photometry/ /fscratch/astro/tess/program/

# Delete old output data before generation new:
rm -rf $TESSPHOT_OUTPUT
mkdir $TESSPHOT_OUTPUT

# Change directory to the local scratch-directory:
cd /fscratch/astro/tess/program

# Run the MPI job:
mpiexec python mpi_scheduler.py
rc=$? # Save the returncode for later

# Copy some of the output to the home directory:
cp /fscratch/astro/tess/input/todo.sqlite ~/tasoc/

echo "========= Job finished at `date` =========="
if [[ $rc != 0 ]]; then exit 2; fi
#
