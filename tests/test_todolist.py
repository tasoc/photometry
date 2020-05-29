#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os
import numpy as np
import tempfile
import logging
import sqlite3
import contextlib
#import itertools
import conftest # noqa: F401
from photometry import todolist
from photometry.utilities import TqdmLoggingHandler

#----------------------------------------------------------------------
def test_methods_file():

	methods_file = os.path.join(os.path.dirname(__file__), '..', 'photometry', 'data', 'todolist-methods.dat')
	data = np.genfromtxt(methods_file, usecols=(0,1,2,3), dtype=None, encoding='utf-8')

	#, sector, datasource, method
	for d in data:
		assert np.issubdtype(d[0].dtype, np.integer)
		assert d[0] > 0
		assert np.issubdtype(d[1].dtype, np.integer)
		assert d[1] > 0
		assert d[2] in ('ffi', 'tpf')
		assert d[3] in ('aperture', 'halo', 'psf', 'linpsf')

#----------------------------------------------------------------------
def test_exclude_file():

	methods_file = os.path.join(os.path.dirname(__file__), '..', 'photometry', 'data', 'todolist-exclude.dat')
	data = np.genfromtxt(methods_file, usecols=(0,1,2), dtype=None, encoding='utf-8')

	#, sector, datasource, method
	for d in data:
		assert np.issubdtype(d[0].dtype, np.integer)
		assert d[0] > 0
		assert np.issubdtype(d[1].dtype, np.integer)
		assert d[1] > 0
		assert d[2] in ('ffi', 'tpf')

#----------------------------------------------------------------------
#def test_calc_cbv_area():
#
#	for camera, ccd in itertools.product((1,2,3,4), (1,2,3,4)):
#
#		settings = {
#			'camera': camera,
#			'ccd': ccd,
#			'camera_centre_ra': 0,
#			'camera_centre_dec': 0
#		}
#
#		catalog_row = {
#			'ra': 0,
#			'decl': 0
#		}
#
#		cbv_area = todolist.calc_cbv_area(catalog_row, settings)
#		print(cbv_area)
#		assert(cbv_area == 131)

#----------------------------------------------------------------------
def test_make_todolist(SHARED_INPUT_DIR):
	with tempfile.NamedTemporaryFile() as tmpfile:
		# Run make_todo and save output to temp-file:
		todolist.make_todo(SHARED_INPUT_DIR, cameras=3, ccds=2, output_file=tmpfile.name)

		tmpfile.flush()
		assert os.path.exists(tmpfile.name + '.sqlite'), "TODO-file was not created"

		with contextlib.closing(sqlite3.connect("file:" + tmpfile.name + '.sqlite?mode=ro', uri=True)) as conn:
			conn.row_factory = sqlite3.Row
			cursor = conn.cursor()

			cursor.execute("SELECT * FROM todolist LIMIT 1;")
			row = cursor.fetchone()
			print(dict(row))
			assert row['camera'] == 3, "Wrong camera returned"
			assert row['ccd'] == 2, "Wrong CCD returned"

			cursor.execute("SELECT COUNT(*) FROM todolist WHERE datasource='tpf';")
			assert cursor.fetchone()[0] == 2, "Expected 2 TPFs in todolist"

			cursor.close()

#----------------------------------------------------------------------
if __name__ == '__main__':
	# Setup logging:
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	console = TqdmLoggingHandler()
	console.setFormatter(formatter)
	logger_parent = logging.getLogger('photometry')
	logger_parent.setLevel(logging.INFO)
	if not logger_parent.hasHandlers():
		logger_parent.addHandler(console)

	pytest.main([__file__])
