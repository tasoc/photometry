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
from photometry import HaloPhotometry, STATUS
from photometry.plots import plot_image, plt
import logging

def test_halo():

	INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')
	OUTPUT_DIR = tempfile.mkdtemp(prefix='tessphot_tests_halo')

	for datasource in (['tpf']): # doesn't work for ffi
		with HaloPhotometry(182092046, INPUT_DIR, OUTPUT_DIR, plot=True, datasource=datasource, camera=1, ccd=1) as pho:

			plt.figure()
			plot_image(pho.sumimage, title=datasource)
			plt.show()

			pho.photometry()
			pho.save_lightcurve()
			print( pho.lightcurve )

			# It should set the status to one of these:
			print(pho.status)
			assert(pho.status in (STATUS.OK, STATUS.WARNING))

			# They shouldn't be exactly zero:
			assert( ~np.all(pho.lightcurve['flux'] == 0) )
			assert( ~np.all(pho.lightcurve['pos_centroid'][:,0] == 0) )
			assert( ~np.all(pho.lightcurve['pos_centroid'][:,1] == 0) )

			# They shouldn't be NaN (in this case!):
			assert( ~np.all(np.isnan(pho.lightcurve['flux'])) )
			assert( ~np.all(np.isnan(pho.lightcurve['pos_centroid'][:,0])) )
			assert( ~np.all(np.isnan(pho.lightcurve['pos_centroid'][:,1])) )
			print("Passed Tests for %s" % datasource)


if __name__ == '__main__':
	test_halo()
