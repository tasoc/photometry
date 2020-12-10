#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests that run run_tessphot with several different inputs.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
import tempfile
import subprocess
import shlex
import sys

STAR_LIST = (
	('aperture', 260795451, 'ffi'),
	('aperture', 260795451, 'tpf'),
	#('psf', 260795451, 'ffi'),
	#('psf', 260795451, 'tpf'),
	#('linpsf', 260795451, 'ffi'),
	#('linpsf', 260795451, 'tpf'),
)

#--------------------------------------------------------------------------------------------------
def capture_run_tessphot(params):

	command = '"%s" run_tessphot.py %s' % (sys.executable, params.strip())
	print(command)

	cmd = shlex.split(command)
	proc = subprocess.Popen(cmd,
		cwd=os.path.join(os.path.dirname(__file__), '..'),
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		universal_newlines=True
	)
	out, err = proc.communicate()
	exitcode = proc.returncode

	print(out)
	print(err)
	print(exitcode)
	return out, err, exitcode

#--------------------------------------------------------------------------------------------------
def test_run_tessphot_invalid_method():
	out, err, exitcode = capture_run_tessphot("-t --method=invalid")
	assert exitcode == 2
	assert "error: argument -m/--method: invalid choice: 'invalid'" in err

#--------------------------------------------------------------------------------------------------
def test_run_tessphot_invalid_datasource():
	out, err, exitcode = capture_run_tessphot("-t --datasource=invalid")
	assert exitcode == 2
	assert "error: argument --datasource: invalid choice: 'invalid'" in err

#--------------------------------------------------------------------------------------------------
def test_run_tessphot_invalid_camera():
	out, err, exitcode = capture_run_tessphot("-t --camera=5")
	assert exitcode == 2
	assert 'error: argument --camera: invalid choice: 5 (choose from 1, 2, 3, 4)' in err

#--------------------------------------------------------------------------------------------------
def test_run_tessphot_invalid_ccd():
	out, err, exitcode = capture_run_tessphot("-t --ccd=14")
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
		#tmpdir = './out-' + method
		params = '-o -p --version=0 --starid={starid:d} --method={method:s} --datasource={datasource:s} --output="{output_dir:s}" "{input_dir:s}"'.format(
			starid=starid,
			method=method,
			datasource=datasource,
			input_dir=SHARED_INPUT_DIR,
			output_dir=tmpdir
		)
		out, err, exitcode = capture_run_tessphot(params)

	assert ' - ERROR - ' not in err
	assert exitcode == 0

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
