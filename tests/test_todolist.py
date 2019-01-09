#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, print_function, with_statement, absolute_import
import os
import numpy as np

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
if __name__ == '__main__':
	test_methods_file()
	test_exclude_file()