#!/bin/bash -e

TRANSFER=$(sbatch --parsable submit_transfer_grendel.sh)
echo $TRANSFER

PREPARE=$(sbatch --parsable --dependency=afterok:$TRANSFER submit_prepare_grendel.sh)
echo $PREPARE

TESSPHOT=$(sbatch --parsable --dependency=afterok:$PREPARE submit_mpi_run_grendel.sh)
echo $TESSPHOT
