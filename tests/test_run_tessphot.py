#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests that run run_tessphot with several different inputs.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import tempfile
from conftest import capture_cli

STAR_LIST = (
	('aperture', 260795451, 'ffi'),
	('aperture', 260795451, 'tpf'),
	#('psf', 260795451, 'ffi'),
	#('psf', 260795451, 'tpf'),
	#('linpsf', 260795451, 'ffi'),
	#('linpsf', 260795451, 'tpf'),
)

#--------------------------------------------------------------------------------------------------
def test_run_tessphot_invalid_method():
	out, err, exitcode = capture_cli('run_tessphot.py', ['-t', '--method=invalid'])
	assert exitcode == 2
	assert "error: argument -m/--method: invalid choice: 'invalid'" in err

#--------------------------------------------------------------------------------------------------
def test_run_tessphot_invalid_datasource():
	out, err, exitcode = capture_cli('run_tessphot.py', ['-t', '--datasource=invalid'])
	assert exitcode == 2
	assert "error: argument --datasource: invalid choice: 'invalid'" in err

#--------------------------------------------------------------------------------------------------
def test_run_tessphot_invalid_camera():
	out, err, exitcode = capture_cli('run_tessphot.py', ['-t', '--camera=5'])
	assert exitcode == 2
	assert 'error: argument --camera: invalid choice: 5 (choose from 1, 2, 3, 4)' in err

#--------------------------------------------------------------------------------------------------
def test_run_tessphot_invalid_ccd():
	out, err, exitcode = capture_cli('run_tessphot.py', ['-t', '--ccd=14'])
	assert exitcode == 2
	assert 'error: argument --ccd: invalid choice: 14 (choose from 1, 2, 3, 4)' in err

#--------------------------------------------------------------------------------------------------
def test_run_tessphot_invalid_cadence():
	out, err, exitcode = capture_run_tessphot("-t --cadence=121")
	assert exitcode == 2
	assert 'error: argument --cadence: invalid choice: 121 (choose from 20, 120, 600, 1800)' in err

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('method,starid,datasource', STAR_LIST)
def test_run_tessphot(SHARED_INPUT_DIR, method, starid, datasource):
	with tempfile.TemporaryDirectory() as tmpdir:
		params = [
			'-o',
			'-p',
			'--version=0',
			f'--starid={starid:d}',
			'--method=' + method,
			'--datasource=' + datasource,
			'--output', tmpdir,
			SHARED_INPUT_DIR
		]
		out, err, exitcode = capture_cli('run_tessphot.py', params)

	assert ' - ERROR - ' not in err
	assert exitcode == 0

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
