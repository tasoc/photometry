#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os
import re
import contextlib
import sqlite3
from timeit import default_timer
from conftest import capture_cli
from photometry import STATUS
from test_todolist import todo_file_valid
from test_prepare import hdf5_file_valid

#--------------------------------------------------------------------------------------------------
@pytest.mark.mpi
@pytest.mark.slow
def test_integrations_sector27(PRIVATE_INPUT_DIR):
	pytest.importorskip("mpi4py", reason="MPI not available")

	# Path to the HDF5 and TODO-file:
	hdf5file = os.path.join(PRIVATE_INPUT_DIR, 'sector027_camera4_ccd1.hdf5')
	todofile = os.path.join(PRIVATE_INPUT_DIR, 'todo.sqlite')

	# Delete existing todo-file:
	os.remove(todofile)
	assert not os.path.exists(todofile), "TODO-file was not removed correctly"
	assert not os.path.exists(hdf5file), "HDF5 file is already present"

	# Run prepare_photometry CLI script:
	tic = default_timer()
	out, err, exitcode = capture_cli('run_prepare_photometry.py', params=[
		'--sector=27',
		'--camera=4',
		'--ccd=1',
		PRIVATE_INPUT_DIR
	])
	print(default_timer() - tic)
	assert exitcode == 0
	# Remove this error, since it is a known issue when running with few FFIs:
	out = re.sub(r'^.+ - ERROR - Sector reference time outside timespan of data$', '', out, flags=re.MULTILINE)
	assert '- ERROR -' not in out
	assert '- ERROR -' not in err
	assert '- INFO - Running SECTOR=27, CAMERA=4, CCD=1' in out
	assert '- INFO - Number of files: 2' in out
	assert '- INFO - Done.' in out

	# Check that the HDF5 file as created:
	assert os.path.isfile(hdf5file), "HDF5 was not created"
	hdf5_file_valid(hdf5file, sector=27, camera=4, ccd=1, Ntimes=2)

	# Run make_todo CLI script:
	tic = default_timer()
	out, err, exitcode = capture_cli('run_make_todo.py', params=[
		'--sector=27',
		'--camera=4',
		'--ccd=1',
		PRIVATE_INPUT_DIR
	])
	print(default_timer() - tic)
	assert exitcode == 0
	assert '- ERROR -' not in out
	assert '- ERROR -' not in err
	assert '- INFO - Number of HDF5 files: 1' in err
	assert '- INFO - Number of TPF files: 2' in err
	assert '- INFO - TODO done.' in err

	# Check that the TODO-file was created:
	assert os.path.isfile(todofile), "TODO-file was not created"
	todo_file_valid(todofile, sector=27, camera=4, ccd=1)

	# Run photometry CLI script:
	tic = default_timer()
	out, err, exitcode = capture_cli('run_tessphot_mpi.py', mpiexec=True, params=[
		'--debug',
		'--plot',
		'--datasource=tpf',
		'--version=99',
		PRIVATE_INPUT_DIR
	])
	print(default_timer() - tic)
	assert exitcode == 0
	assert '- ERROR -' not in out
	assert '- ERROR -' not in err
	assert '- INFO - 4 tasks to be run' in err
	assert '- INFO - Master starting with 1 workers' in err
	assert '- DEBUG - Got data from worker 1: {' in err
	assert '- INFO - Worker 1 exited.' in err
	assert '- INFO - Master finishing' in err

	# Check the final TODO-file:
	with contextlib.closing(sqlite3.connect("file:" + todofile + '?mode=ro', uri=True)) as conn:
		conn.row_factory = sqlite3.Row
		cursor = conn.cursor()

		cursor.execute("SELECT * FROM todolist LEFT JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE datasource='tpf';")
		for row in cursor.fetchall():
			print(dict(row))
			assert row['status'] == STATUS.OK.value
			assert row['lightcurve'] is not None
			# Check that the output lightcurve exists:
			assert os.path.isfile(os.path.join(PRIVATE_INPUT_DIR, row['lightcurve']))

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
