#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:54:10 2017

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import logging
import numpy as np
from bottleneck import allnan, anynan
import sys
import os
from tempfile import TemporaryDirectory
from astropy.io import fits
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from photometry import HaloPhotometry, STATUS

#--------------------------------------------------------------------------------------------------
#@pytest.mark.skipif(os.environ.get('CI') == 'true' and os.environ.get('TRAVIS') == 'true',
#	reason="This is simply too slow to run on Travis. We need to do something about that.'")
@pytest.mark.parametrize('datasource', ['tpf',]) # Not testing 'ffi' since there is not enough data
def test_halo(datasource):

	INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')

	with TemporaryDirectory() as OUTPUT_DIR:
		with HaloPhotometry(267211065, INPUT_DIR, OUTPUT_DIR, plot=True, datasource=datasource, sector=1, camera=3, ccd=2) as pho:

			pho.photometry()
			filepath = pho.save_lightcurve()
			print( pho.lightcurve )

			# It should set the status to one of these:
			print(pho.status)
			assert pho.status in (STATUS.OK, STATUS.WARNING)

			# They shouldn't be exactly zero:
			assert not np.all(pho.lightcurve['flux'] == 0)
			assert not np.all(pho.lightcurve['flux_err'] == 0)
			assert not np.all(pho.lightcurve['pos_centroid'][:,0] == 0)
			assert not np.all(pho.lightcurve['pos_centroid'][:,1] == 0)

			# They shouldn't be NaN (in this case!):
			assert not allnan(pho.lightcurve['flux'])
			assert not allnan(pho.lightcurve['flux_err'])
			assert not allnan(pho.lightcurve['pos_centroid'][:,0])
			assert not allnan(pho.lightcurve['pos_centroid'][:,1])

			# Test the outputted FITS file:
			with fits.open(filepath, mode='readonly') as hdu:
				# Should be the same vectors in FITS as returned in Table:
				np.testing.assert_allclose(pho.lightcurve['time'], hdu[1].data['TIME'])
				np.testing.assert_allclose(pho.lightcurve['timecorr'], hdu[1].data['TIMECORR'])
				np.testing.assert_allclose(pho.lightcurve['flux'], hdu[1].data['FLUX_RAW'])
				np.testing.assert_allclose(pho.lightcurve['flux_err'], hdu[1].data['FLUX_RAW_ERR'])
				np.testing.assert_allclose(pho.lightcurve['cadenceno'], hdu[1].data['CADENCENO'])

				# Test FITS aperture image:
				ap = hdu['APERTURE'].data
				print(ap)
				assert np.all(pho.aperture == ap), "Aperture image mismatch"
				assert not anynan(ap), "NaN in aperture image"
				assert np.all(ap >= 0), "Negative values in aperture image"
				assert np.any(ap & 2 != 0), "No photometric mask set"
				#assert np.any(ap & 8 != 0), "No position mask set"

	print("Passed Tests for %s" % datasource)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	# Setup logging:
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	console = logging.StreamHandler()
	console.setFormatter(formatter)
	logger_phot = logging.getLogger('photometry')
	if not logger_phot.hasHandlers(): logger_phot.addHandler(console)
	logger_phot.setLevel(logging.INFO)

	test_halo('tpf')
	#test_halo('ffi')
