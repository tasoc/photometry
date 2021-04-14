#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of Linear PSF photometry.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import numpy as np
from bottleneck import allnan, anynan
import logging
from tempfile import TemporaryDirectory
from astropy.io import fits
import conftest # noqa: F401
from photometry import LinPSFPhotometry, STATUS
from photometry.plots import plot_image, plt

DUMMY_TARGET = 260795451
DUMMY_KWARG = {'sector': 1, 'camera': 3, 'ccd': 2}

#--------------------------------------------------------------------------------------------------
@pytest.mark.skip(reason='PSF Photometry still in development')
def test_linpsfphotometry(SHARED_INPUT_DIR):
	datasource = 'ffi'
	with TemporaryDirectory() as OUTPUT_DIR:
		with LinPSFPhotometry(DUMMY_TARGET, SHARED_INPUT_DIR, OUTPUT_DIR, plot=True, datasource=datasource, **DUMMY_KWARG) as pho:

			pho.photometry()
			filepath = pho.save_lightcurve()
			print( pho.lightcurve )

			# It should set the status to one of these:
			assert(pho.status in (STATUS.OK, STATUS.WARNING))

			# Check the sumimage:
			plt.figure()
			plot_image(pho.sumimage, title=datasource)

			assert not anynan(pho.sumimage), "There are NaNs in the SUMIMAGE"

			# They shouldn't be exactly zero:
			assert not np.all(pho.lightcurve['flux'] == 0), "Flux is zero"
			assert not np.all(pho.lightcurve['flux_err'] == 0), "Flux error is zero"
			#assert not np.all(pho.lightcurve['pos_centroid'][:,0] == 0), "Position is zero"
			#assert not np.all(pho.lightcurve['pos_centroid'][:,1] == 0), "Position is zero"

			# They shouldn't be NaN (in this case!):
			assert not allnan(pho.lightcurve['flux']), "Flux is all NaN"
			assert not allnan(pho.lightcurve['flux_err']), "Flux error is all NaN"
			#assert not allnan(pho.lightcurve['pos_centroid'][:,0]), "Position is all NaN"
			#assert not allnan(pho.lightcurve['pos_centroid'][:,1]), "Position is all NaN"

			assert not np.any(~np.isfinite(pho.lightcurve['time'])), "Time contains NaN"
			assert not np.any(pho.lightcurve['time'] == 0), "Time contains zero"

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
				assert not np.any(ap & 2 != 0), "Photometric mask set - shouldn't for PSF photometry"
				assert not np.any(ap & 8 != 0), "Position mask set - shouldn't for PSF Photometry"

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	# Setup logging:
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	console = logging.StreamHandler()
	console.setFormatter(formatter)
	logger_phot = logging.getLogger('photometry')
	if not logger_phot.hasHandlers():
		logger_phot.addHandler(console)
	logger_phot.setLevel(logging.INFO)

	pytest.main([__file__])
