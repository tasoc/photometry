#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of Aperture Photometry.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import numpy as np
from bottleneck import allnan, anynan
import logging
from tempfile import TemporaryDirectory
from astropy.io import fits
import conftest # noqa: F401
from photometry import AperturePhotometry, STATUS
from photometry.plots import plot_image, plt

DUMMY_TARGET = 260795451
DUMMY_KWARG = {'sector': 1, 'camera': 3, 'ccd': 2}

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('datasource', ['tpf', 'ffi'])
def test_aperturephotometry(SHARED_INPUT_DIR, datasource):
	with TemporaryDirectory() as OUTPUT_DIR:
		with AperturePhotometry(DUMMY_TARGET, SHARED_INPUT_DIR, OUTPUT_DIR, plot=True, datasource=datasource, **DUMMY_KWARG) as pho:

			pho.photometry()
			filepath = pho.save_lightcurve(version=99)
			print( pho.lightcurve )

			# It should set the status to one of these:
			assert(pho.status in (STATUS.OK, STATUS.WARNING))

			# Check the sumimage:
			plt.figure()
			plot_image(pho.sumimage, title=datasource)

			assert not anynan(pho.sumimage), "There are NaNs in the SUMIMAGE"

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

			assert not np.any(~np.isfinite(pho.lightcurve['time']))
			assert not np.any(pho.lightcurve['time'] == 0)

			# Test the outputted FITS file:
			with fits.open(filepath, mode='readonly') as hdu:
				# Check the number of extensions is correct:
				assert len(hdu) == 4
				assert isinstance(hdu[0], fits.PrimaryHDU)
				assert hdu[0].data is None
				assert isinstance(hdu[1], fits.BinTableHDU)
				assert hdu[1].header['EXTNAME'] == 'LIGHTCURVE'
				assert isinstance(hdu[2], fits.ImageHDU)
				assert hdu[2].header['EXTNAME'] == 'SUMIMAGE'
				assert isinstance(hdu[3], fits.ImageHDU)
				assert hdu[3].header['EXTNAME'] == 'APERTURE'

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
				assert np.any(ap & 8 != 0), "No position mask set"

				# Test the generated header:
				hdr = hdu[0].header
				assert hdr['TICID'] == DUMMY_TARGET
				assert hdr['OBJECT'] == 'TIC ' + str(DUMMY_TARGET)
				assert hdr['SECTOR'] == DUMMY_KWARG['sector']
				assert hdr['CAMERA'] == DUMMY_KWARG['camera']
				assert hdr['CCD'] == DUMMY_KWARG['ccd']
				assert hdr['PHOTMET'] == 'aperture'
				assert hdr['TICVER'] == '8.2'
				assert hdr['FILEVER'] == '1.5'
				assert hdr['VERSION'] == 99
				assert hdr['DATAVAL'] == 0

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('datasource', ['tpf', 'ffi'])
def test_aperturephotometry_plots(SHARED_INPUT_DIR, datasource):
	with TemporaryDirectory() as OUTPUT_DIR:
		with AperturePhotometry(DUMMY_TARGET, SHARED_INPUT_DIR, OUTPUT_DIR, plot=False, datasource=datasource, **DUMMY_KWARG) as pho_noplot:
			pho_noplot.photometry()
			assert isinstance(pho_noplot.plot, bool), "PLOT should be boolean"
			assert not pho_noplot.plot, "PLOT should be False"
			print(pho_noplot.status)

		with AperturePhotometry(DUMMY_TARGET, SHARED_INPUT_DIR, OUTPUT_DIR, plot=True, datasource=datasource, **DUMMY_KWARG) as pho_plot:
			pho_plot.photometry()
			assert isinstance(pho_plot.plot, bool), "PLOT should be boolean"
			assert pho_plot.plot, "PLOT should be True"
			print(pho_plot.status)

		assert pho_plot.status == pho_noplot.status
		np.testing.assert_allclose(pho_noplot.sumimage, pho_plot.sumimage, equal_nan=True)
		np.testing.assert_allclose(pho_noplot.lightcurve['time'], pho_plot.lightcurve['time'], equal_nan=True)
		np.testing.assert_allclose(pho_noplot.lightcurve['flux'], pho_plot.lightcurve['flux'], equal_nan=True)

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
