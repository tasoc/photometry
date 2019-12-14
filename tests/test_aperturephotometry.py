#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:54:10 2017

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import sys
import os
from tempfile import TemporaryDirectory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from photometry import AperturePhotometry, STATUS
from photometry.plots import plot_image, plt

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')
DUMMY_TARGET = 260795451
DUMMY_KWARG = {'sector': 1, 'camera': 3, 'ccd': 2}

def test_aperturephotometry():

	with TemporaryDirectory() as OUTPUT_DIR:
		for datasource in ('tpf', 'ffi'):
			with AperturePhotometry(DUMMY_TARGET, INPUT_DIR, OUTPUT_DIR, plot=True, datasource=datasource, **DUMMY_KWARG) as pho:

				pho.photometry()
				pho.save_lightcurve()
				print( pho.lightcurve )

				# It should set the status to one of these:
				assert(pho.status in (STATUS.OK, STATUS.WARNING))

				plt.figure()
				plot_image(pho.sumimage, title=datasource)
				#plt.show()

				# They shouldn't be exactly zero:
				assert( ~np.all(pho.lightcurve['flux'] == 0) )
				assert( ~np.all(pho.lightcurve['pos_centroid'][:,0] == 0) )
				assert( ~np.all(pho.lightcurve['pos_centroid'][:,1] == 0) )

				# They shouldn't be NaN (in this case!):
				assert( ~np.all(np.isnan(pho.lightcurve['flux'])) )
				assert( ~np.all(np.isnan(pho.lightcurve['pos_centroid'][:,0])) )
				assert( ~np.all(np.isnan(pho.lightcurve['pos_centroid'][:,1])) )


def test_aperturephotometry_plots():

	with TemporaryDirectory() as OUTPUT_DIR:
		with AperturePhotometry(DUMMY_TARGET, INPUT_DIR, OUTPUT_DIR, plot=False, datasource='ffi', **DUMMY_KWARG) as pho_noplot:
			pho_noplot.photometry()
			assert isinstance(pho_noplot.plot, bool), "PLOT should be boolean"
			assert not pho_noplot.plot, "PLOT should be False"
			print(pho_noplot.status)

		with AperturePhotometry(DUMMY_TARGET, INPUT_DIR, OUTPUT_DIR, plot=True, datasource='ffi', **DUMMY_KWARG) as pho_plot:
			pho_plot.photometry()
			assert isinstance(pho_plot.plot, bool), "PLOT should be boolean"
			assert pho_plot.plot, "PLOT should be True"
			print(pho_plot.status)

		assert(pho_plot.status == pho_noplot.status)
		np.testing.assert_allclose(pho_noplot.sumimage, pho_plot.sumimage, equal_nan=True)
		np.testing.assert_allclose(pho_noplot.lightcurve['time'], pho_plot.lightcurve['time'], equal_nan=True)
		np.testing.assert_allclose(pho_noplot.lightcurve['flux'], pho_plot.lightcurve['flux'], equal_nan=True)


if __name__ == '__main__':
	test_aperturephotometry()
	test_aperturephotometry_plots()
