#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytest fixture to create temp copy of input data, shared across all tests.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
import tempfile
import shutil
import sys
import subprocess
import shlex

# Insert photometry package as the first on path:
if sys.path[0] != os.path.abspath(os.path.join(os.path.dirname(__file__), '..')):
	sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#--------------------------------------------------------------------------------------------------
def capture_cli(script, params=[], mpiexec=False):

	if isinstance(params, str):
		params = shlex.split(params)

	cmd = [sys.executable, script.strip()] + list(params)
	if mpiexec:
		cmd = ['mpiexec', '-n', '2'] + cmd
	print(cmd)

	proc = subprocess.Popen(cmd,
		cwd=os.path.join(os.path.dirname(__file__), '..'),
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		universal_newlines=True
	)
	out, err = proc.communicate()
	exitcode = proc.returncode
	proc.kill()

	print("ExitCode: %d" % exitcode)
	print("StdOut:\n%s" % out.strip())
	print("StdErr:\n%s" % err.strip())
	return out, err, exitcode

#--------------------------------------------------------------------------------------------------
@pytest.fixture(scope='session')
def SHARED_INPUT_DIR():
	"""
	Pytest fixture to create temp copy of input data, shared across all tests.
	"""
	INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')
	with tempfile.TemporaryDirectory(prefix='pytest-shared-input-dir-') as my_tmpdir:
		tmp = os.path.join(my_tmpdir, 'input')
		shutil.copytree(INPUT_DIR, tmp)
		yield tmp

#--------------------------------------------------------------------------------------------------
@pytest.fixture(scope='function')
def PRIVATE_INPUT_DIR():
	"""
	Pytest fixture to create temp copy of input data, shared across all tests.
	"""
	INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')
	with tempfile.TemporaryDirectory(prefix='pytest-private-input-dir-') as my_tmpdir:
		tmp = os.path.join(my_tmpdir, 'input')
		shutil.copytree(INPUT_DIR, tmp)
		yield tmp

#--------------------------------------------------------------------------------------------------
@pytest.fixture(scope='function')
def PRIVATE_TODO_FILE():
	"""
	Pytest fixture to create temp copy of input todo-file, private to the individual test.
	"""
	TODO_FILE = os.path.join(os.path.dirname(__file__), 'input', 'todo.sqlite')
	with tempfile.TemporaryDirectory(prefix='pytest-private-todo-') as my_tmpdir:
		tmp = os.path.join(my_tmpdir, 'todo.sqlite')
		shutil.copy2(TODO_FILE, tmp)
		yield tmp

#--------------------------------------------------------------------------------------------------
@pytest.fixture(scope='function')
def PRIVATE_SPICE_DIR(monkeypatch):
	"""
	Pytest fixture to create temp copy of SPICE data directory and changes the
	TESSPHOT_SPICE_KERNELS environment variable.
	"""
	SPICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'photometry', 'data', 'spice'))
	with tempfile.TemporaryDirectory(prefix='pytest-private-spice-') as tmpdir:
		tmp = os.path.join(tmpdir, 'spice')
		shutil.copytree(SPICE_DIR, tmp)
		monkeypatch.setenv('TESSPHOT_SPICE_KERNELS', tmp)
		yield tmp
