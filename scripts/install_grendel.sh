#!/bin/bash -e

# Load the required software stack:
source /comm/swstack/bin/modules.sh --silent
module load intel
module load openmpi
module load python/3.6.3

# Create clean virtual environment:
rm -rf ~/python-virtualenv/tessphot
virtualenv --clear ~/python-virtualenv/tessphot
#python3 -m venv --clear ~/python-virtualenv/tessphot

# Activate the new virtual environment:
source ~/python-virtualenv/tessphot/bin/activate

# Install all required Python packages:
pip install -r ../requirements.txt
pip install mpi4py --upgrade --force-reinstall --no-cache-dir
