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
from photometry.plots import plot_image, plt

def _test_aperturephotometry():

	INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')
	OUTPUT_DIR = tempfile.mkdtemp(prefix='tessphot_tests_aperture')

	for datasource in ('tpf', 'ffi'):
		with AperturePhotometry(182092046, INPUT_DIR, OUTPUT_DIR, plot=True, datasource=datasource, camera=1, ccd=1) as pho:

			pho.photometry()
			pho.save_lightcurve()
			print( pho.lightcurve )

			# It should set the status to one of these:
			assert(pho.status in (STATUS.OK, STATUS.WARNING))

			plt.figure()
			plot_image(pho.sumimage, title=datasource)
			plt.show()

			# They shouldn't be exactly zero:
			assert( ~np.all(pho.lightcurve['flux'] == 0) )
			assert( ~np.all(pho.lightcurve['pos_centroid'][:,0] == 0) )
			assert( ~np.all(pho.lightcurve['pos_centroid'][:,1] == 0) )

			# They shouldn't be NaN (in this case!):
			assert( ~np.all(np.isnan(pho.lightcurve['flux'])) )
			assert( ~np.all(np.isnan(pho.lightcurve['pos_centroid'][:,0])) )
			assert( ~np.all(np.isnan(pho.lightcurve['pos_centroid'][:,1])) )


if __name__ == '__main__':
	_test_aperturephotometry()
