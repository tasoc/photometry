#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import os
import numpy as np
import sys
import itertools
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from photometry import todolist

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
if __name__ == '__main__':
	test_methods_file()
	test_exclude_file()
	#test_calc_cbv_area()