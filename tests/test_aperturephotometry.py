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

def test_aperturephotometry():

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


def test_aperturephotometry_plots():

	INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')
	OUTPUT_DIR = tempfile.mkdtemp(prefix='tessphot_tests_aperture')

	with AperturePhotometry(182092046, INPUT_DIR, OUTPUT_DIR, plot=False, datasource='ffi', camera=1, ccd=1) as pho_noplot:
		pho_noplot.photometry()
		assert(pho_noplot.plot == False)
		print(pho_noplot.status)

	with AperturePhotometry(182092046, INPUT_DIR, OUTPUT_DIR, plot=True, datasource='ffi', camera=1, ccd=1) as pho_plot:
		pho_plot.photometry()
		assert(pho_plot.plot == True)
		print(pho_plot.status)

	assert(pho_plot.status == pho_noplot.status)
	np.testing.assert_allclose(pho_noplot.sumimage, pho_plot.sumimage, equal_nan=True)
	np.testing.assert_allclose(pho_noplot.lightcurve['time'], pho_plot.lightcurve['time'], equal_nan=True)
	np.testing.assert_allclose(pho_noplot.lightcurve['flux'], pho_plot.lightcurve['flux'], equal_nan=True)


if __name__ == '__main__':
	test_aperturephotometry()
	test_aperturephotometry_plots()
