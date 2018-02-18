#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:54:10 2017

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, print_function, with_statement, absolute_import
import numpy as np
import sys
import os
import tempfile
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from photometry import AperturePhotometry, STATUS

def test_aperturephotometry():

	INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')
	OUTPUT_DIR = tempfile.mkdtemp(prefix='tessphot_tests_aperture')
	DUMMY_TARGET = 143159

	with AperturePhotometry(DUMMY_TARGET, INPUT_DIR, OUTPUT_DIR, plot=True) as pho:

		pho.photometry()
		pho.save_lightcurve(OUTPUT_DIR)
		print( pho.lightcurve )

		# It should set the status to one of these:
		assert(pho.status in (STATUS.OK, STATUS.WARNING))

		# They shouldn't be exactly zero:
		assert( np.all(pho.lightcurve['flux'] != 0) )
		assert( np.all(pho.lightcurve['pos_centroid'][:,0] != 0) )
		assert( np.all(pho.lightcurve['pos_centroid'][:,1] != 0) )

		# They shouldn't be NaN (in this case!):
		assert( np.all(~np.isnan(pho.lightcurve['flux'])) )
		assert( np.all(~np.isnan(pho.lightcurve['pos_centroid'][:,0])) )
		assert( np.all(~np.isnan(pho.lightcurve['pos_centroid'][:,1])) )

if __name__ == '__main__':
	test_aperturephotometry()
