#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of BasePhotometry.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, print_function, with_statement, absolute_import
import numpy as np
import sys
import os
import tempfile
from astropy.io import fits
from astropy.wcs import WCS
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from photometry import BasePhotometry

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')
DUMMY_TARGET = 471012650

def test_stamp():
	with tempfile.TemporaryDirectory() as OUTPUT_DIR:
		with BasePhotometry(DUMMY_TARGET, INPUT_DIR, OUTPUT_DIR, datasource='ffi', camera=2, ccd=2) as pho:

			pho._stamp = (0, 10, 0, 20)
			pho._set_stamp()

			cols, rows = pho.get_pixel_grid()
			print('Rows:')
			print(rows)
			print(rows.shape)
			print('Cols:')
			print(cols)
			print(cols.shape)

			assert(rows.shape == (10, 20))
			assert(cols.shape == (10, 20))
			assert(rows[0,0] == 1)
			assert(cols[0,0] == 1)
			assert(rows[-1,0] == 10)
			assert(cols[-1,0] == 1)
			assert(rows[-1,-1] == 10)
			assert(cols[-1,-1] == 20)

			pho.resize_stamp(up=12)
			cols, rows = pho.get_pixel_grid()
			print('Rows:')
			print(rows)
			print(rows.shape)
			print('Cols:')
			print(cols)
			print(cols.shape)
			assert(rows.shape == (22, 20))
			assert(cols.shape == (22, 20))

def test_images():
	with tempfile.TemporaryDirectory() as OUTPUT_DIR:
		with BasePhotometry(DUMMY_TARGET, INPUT_DIR, OUTPUT_DIR, datasource='ffi', camera=2, ccd=2) as pho:

			pho._stamp = (0, 10, 0, 20)
			pho._set_stamp()

			for img in pho.images:
				assert(img.shape == (10, 20))

def test_backgrounds():
	with tempfile.TemporaryDirectory() as OUTPUT_DIR:
		with BasePhotometry(DUMMY_TARGET, INPUT_DIR, OUTPUT_DIR, datasource='ffi', camera=2, ccd=2) as pho:

			pho._stamp = (0, 10, 0, 20)
			pho._set_stamp()

			for img in pho.backgrounds:
				assert(img.shape == (10, 20))

def test_catalog():
	with tempfile.TemporaryDirectory() as OUTPUT_DIR:
		for datasource in ('ffi', 'tpf'):
			with BasePhotometry(DUMMY_TARGET, INPUT_DIR, OUTPUT_DIR, datasource=datasource, camera=2, ccd=2) as pho:
				print(pho.catalog)
				assert(DUMMY_TARGET in pho.catalog['starid'])

				assert(pho.target_pos_ra >= np.min(pho.catalog['ra']))
				assert(pho.target_pos_ra <= np.max(pho.catalog['ra']))
				assert(pho.target_pos_dec >= np.min(pho.catalog['dec']))
				assert(pho.target_pos_dec <= np.max(pho.catalog['dec']))

				indx_main = (pho.catalog['starid'] == DUMMY_TARGET)

				# Test the real position - TODO: How do we find this?
				#np.testing.assert_allclose(pho.target_pos_column, 1978.082)
				#np.testing.assert_allclose(pho.target_pos_row, 652.5701)

				np.testing.assert_allclose(pho.catalog[indx_main]['column'], pho.target_pos_column)
				np.testing.assert_allclose(pho.catalog[indx_main]['row'], pho.target_pos_row)


def test_catalog_attime():
	with tempfile.TemporaryDirectory() as OUTPUT_DIR:
		for datasource in ('ffi', 'tpf'):
			with BasePhotometry(DUMMY_TARGET, INPUT_DIR, OUTPUT_DIR, datasource=datasource, camera=2, ccd=2) as pho:

				time = pho.lightcurve['time']

				cat = pho.catalog_attime(time[0])

				assert(cat.colnames == pho.catalog.colnames)
				# TODO: Add more tests here, once we change the test input data


def test_wcs():
	with tempfile.TemporaryDirectory() as OUTPUT_DIR:
		with BasePhotometry(DUMMY_TARGET, INPUT_DIR, OUTPUT_DIR, datasource='ffi', camera=2, ccd=2) as pho:
			cols, rows = pho.get_pixel_grid()
			wcs_ffi = pho.wcs
			filepath = pho.save_lightcurve()

		with fits.open(filepath, mode='readonly', memmap=True) as hdu:
			wcs_fits = WCS(header=hdu['APERTURE'].header, relax=True)

	print(wcs_ffi)
	print(wcs_fits)

	# Test the pixels in the corners of the stamp:
	Nr, Nc = cols.shape
	test_pixels = np.array([[0, 0], [Nr-1, Nc-1], [0, Nc-1], [Nr-1, 0]])

	# Corresponding pixels in the FFI:
	# Remember that cols and rows are 1-based.
	test_pixels_ffi = np.array([[cols[r, c]-1, rows[r, c]-1] for c, r in test_pixels])
	print(test_pixels_ffi)

	# Calculate sky-coordinates using both WCS:
	radec_ffi = wcs_ffi.all_pix2world(test_pixels_ffi, 0, ra_dec_order=True)
	radec_fits = wcs_fits.all_pix2world(test_pixels, 0, ra_dec_order=True)

	# Check that the sky-coordinates are the same:
	print(radec_ffi - radec_fits)
	np.testing.assert_allclose(radec_fits, radec_ffi)


if __name__ == '__main__':
	test_stamp()
	test_images()
	test_backgrounds()
	test_catalog()
	test_catalog_attime()
	test_wcs()
