#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests that run run_tessphot_mpi with several different inputs.

Most of these tests are SKIPPED if the mpi4py module is not available.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
from conftest import capture_cli
from photometry import TaskManager

#--------------------------------------------------------------------------------------------------
@pytest.mark.mpi
def test_run_tessphot_mpi_invalid_datasource():
	pytest.importorskip("mpi4py", reason="MPI not available")

	out, err, exitcode = capture_cli('run_tessphot_mpi.py', params=["--datasource=invalid"])
	assert exitcode == 2
	assert "error: argument --datasource: invalid choice: 'invalid'" in err

#--------------------------------------------------------------------------------------------------
@pytest.mark.mpi
def test_run_tessphot_mpi_invalid_camera():
	pytest.importorskip("mpi4py", reason="MPI not available")

	out, err, exitcode = capture_cli('run_tessphot_mpi.py', params=["--camera=5"])
	assert exitcode == 2
	assert 'error: argument --camera: invalid choice: 5 (choose from 1, 2, 3, 4)' in err

#--------------------------------------------------------------------------------------------------
@pytest.mark.mpi
def test_run_tessphot_mpi_invalid_ccd():
	pytest.importorskip("mpi4py", reason="MPI not available")

	out, err, exitcode = capture_cli('run_tessphot_mpi.py', params=["--ccd=14"])
	assert exitcode == 2
	assert 'error: argument --ccd: invalid choice: 14 (choose from 1, 2, 3, 4)' in err

#--------------------------------------------------------------------------------------------------
@pytest.mark.mpi
def test_run_tessphot_mpi_invalid_cadence():
	pytest.importorskip("mpi4py", reason="MPI not available")

	out, err, exitcode = capture_cli('run_tessphot_mpi.py', ['-t', '--cadence=121'])
	assert exitcode == 2
	assert 'error: argument --cadence: invalid choice: 121 (choose from 20, 120, 600, 1800)' in err

#--------------------------------------------------------------------------------------------------
@pytest.mark.mpi
def test_run_tessphot_mpi(PRIVATE_INPUT_DIR):
	pytest.importorskip("mpi4py", reason="MPI not available")

	stars_to_test = (
		[260795451, 'tpf'],
		[260795451, 'ffi']
	)

	with TaskManager(PRIVATE_INPUT_DIR) as tm:
		tm.cursor.execute("UPDATE todolist SET status=1;")
		for starid, datasource in stars_to_test:
			tm.cursor.execute("UPDATE todolist SET status=NULL WHERE starid=? AND datasource=?;", [starid, datasource])
		tm.conn.commit()
		tm.cursor.execute("SELECT COUNT(*) FROM todolist WHERE status IS NULL;")
		num = tm.cursor.fetchone()[0]

	print(num)
	assert num == len(stars_to_test)

	out, err, exitcode = capture_cli('run_tessphot_mpi.py', mpiexec=True, params=[
		'--debug',
		'--version=0',
		PRIVATE_INPUT_DIR
	])

	assert ' - INFO - %d tasks to be run' % num in err
	assert ' - INFO - Master starting with 1 workers' in err
	assert ' - DEBUG - Got data from worker 1: {' in err
	assert ' - INFO - Worker 1 exited.' in err
	assert ' - INFO - Master finishing' in err
	assert exitcode == 0

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
