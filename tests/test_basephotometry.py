#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of BasePhotometry.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import sys
import os
try:
	from tempfile import TemporaryDirectory
except ImportError:
	from backports.tempfile import TemporaryDirectory
from astropy.io import fits
from astropy.wcs import WCS
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from photometry import BasePhotometry, PixelQualityFlags, CorrectorQualityFlags
#import photometry.BasePhotometry.hdf5_cache as bf

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')
DUMMY_TARGET = 260795451
DUMMY_KWARG = {'sector': 1, 'camera': 3, 'ccd': 2}

#----------------------------------------------------------------------
def test_stamp():
	with TemporaryDirectory() as OUTPUT_DIR:
		with BasePhotometry(DUMMY_TARGET, INPUT_DIR, OUTPUT_DIR, datasource='ffi', **DUMMY_KWARG) as pho:

			pho._stamp = (50, 60, 50, 70)
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
			assert(rows[0,0] == 51)
			assert(cols[0,0] == 51)
			assert(rows[-1,0] == 60)
			assert(cols[-1,0] == 51)
			assert(rows[-1,-1] == 60)
			assert(cols[-1,-1] == 70)

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

#----------------------------------------------------------------------
def test_images():
	with TemporaryDirectory() as OUTPUT_DIR:
		with BasePhotometry(DUMMY_TARGET, INPUT_DIR, OUTPUT_DIR, datasource='ffi', **DUMMY_KWARG) as pho:

			pho._stamp = (50, 60, 50, 70)
			pho._set_stamp()

			for img in pho.images:
				assert(img.shape == (10, 20))

#----------------------------------------------------------------------
def test_backgrounds():
	with TemporaryDirectory() as OUTPUT_DIR:
		with BasePhotometry(DUMMY_TARGET, INPUT_DIR, OUTPUT_DIR, datasource='ffi', **DUMMY_KWARG) as pho:

			pho._stamp = (50, 60, 50, 70)
			pho._set_stamp()

			for img in pho.backgrounds:
				assert(img.shape == (10, 20))

#----------------------------------------------------------------------
def test_catalog():
	with TemporaryDirectory() as OUTPUT_DIR:
		for datasource in ('ffi', 'tpf'):
			with BasePhotometry(DUMMY_TARGET, INPUT_DIR, OUTPUT_DIR, datasource=datasource, **DUMMY_KWARG) as pho:
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

#----------------------------------------------------------------------
def test_catalog_attime():
	with TemporaryDirectory() as OUTPUT_DIR:
		for datasource in ('ffi', 'tpf'):
			with BasePhotometry(DUMMY_TARGET, INPUT_DIR, OUTPUT_DIR, datasource=datasource, **DUMMY_KWARG) as pho:

				time = pho.lightcurve['time']

				cat = pho.catalog_attime(time[0])

				assert(cat.colnames == pho.catalog.colnames)
				# TODO: Add more tests here, once we change the test input data

#----------------------------------------------------------------------
def test_aperture():
	with TemporaryDirectory() as OUTPUT_DIR:
		for datasource in ('ffi', 'tpf'):
			with BasePhotometry(DUMMY_TARGET, INPUT_DIR, OUTPUT_DIR, datasource=datasource, **DUMMY_KWARG) as pho:

				print("------------------------------------")
				print(pho.aperture)

				print(pho.sumimage.shape)
				print(pho.aperture.shape)
				assert(pho.sumimage.shape == pho.aperture.shape)

				# For this target, all the pixels should be available:
				assert np.all(pho.aperture & 1 != 0)

				# This target should fall on CCD output B:
				assert np.all(pho.aperture & 64 != 0)

				if datasource == 'ffi':
					# For the FFI's all pixels for this target was used for the backgrounds
					# (the target is not bright enough to be masked out)
					assert np.all(pho.aperture & 4 != 0)

				# Make the stamp one pixel smaller:
				# The sumimage and aperture should still match in size!
				pho.resize_stamp(right=-1)
				print(pho.sumimage.shape)
				print(pho.aperture.shape)

				assert pho.sumimage.shape == pho.aperture.shape

		# Try this very bright star, where the centre is saturated.
		# The aperture for this star should have pixels near the centre that
		# were not used in the background calculation for FFIs:
		with BasePhotometry(267211065, INPUT_DIR, '.', datasource='ffi', plot=True, **DUMMY_KWARG) as pho:
			central_pixel = pho.aperture[int(np.round(pho.target_pos_row_stamp)), int(np.round(pho.target_pos_column_stamp))]
			assert central_pixel & 4 == 0, "Central pixel of this bright star should not be used in background"

#----------------------------------------------------------------------
def test_pixelflags():
	with TemporaryDirectory() as OUTPUT_DIR:
		for datasource in ('ffi', 'tpf'):
			with BasePhotometry(DUMMY_TARGET, INPUT_DIR, OUTPUT_DIR, datasource=datasource, **DUMMY_KWARG) as pho:
				# Check the size of the pixelflags cube:
				print(pho.pixelflags_cube.shape)
				expected_size = (pho.stamp[1]-pho.stamp[0], pho.stamp[3]-pho.stamp[2], len(pho.hdf['time']))
				print(expected_size)
				assert pho.pixelflags_cube.shape == expected_size, "pixelflags_cube does not have the correct size"

				# Insert a fake pixel flag:
				pho._pixelflags_cube[:, :, :] = 0
				pho._pixelflags_cube[0, 0, 2] |= PixelQualityFlags.BackgroundShenanigans

				# Try to loop through the pixelflags:
				tpfs_with_flag = []
				for k, pf in enumerate(pho.pixelflags):
					if datasource == 'ffi' and k == 2:
						print(pf)
						assert pf[0,0] & PixelQualityFlags.BackgroundShenanigans != 0
					else:
						if pf[0, 0] & PixelQualityFlags.BackgroundShenanigans != 0:
							tpfs_with_flag.append(k)

				# The pixelflags iterator should have Ntimes points:
				assert k+1 == pho.Ntimes, "The pixelflags iterator does not have the correct number of elements"

				# Save the final lightcurve and load the resulting QUALITY flags:
				tmpfile = pho.save_lightcurve()
				with fits.open(tmpfile) as hdu:
					quality = hdu['LIGHTCURVE'].data['QUALITY']

				# Check that the quality flag is being peopergated through to the QUALITY column:
				if datasource == 'ffi':
					assert quality[2] & CorrectorQualityFlags.BackgroundShenanigans != 0, "BackgroundShenanigans flag not correctly propergated"
				else:
					# Since we have only set the flag for a single FFI, there should be 15 TPFs affected:
					assert len(tpfs_with_flag) == 15, "Not the correct number of TPFs with flag set"

					for i in tpfs_with_flag:
						assert quality[i] & CorrectorQualityFlags.BackgroundShenanigans != 0, "BackgroundShenanigans flag not correctly propergated"

#----------------------------------------------------------------------
def test_wcs():
	with TemporaryDirectory() as OUTPUT_DIR:
		with BasePhotometry(DUMMY_TARGET, INPUT_DIR, OUTPUT_DIR, datasource='tpf', **DUMMY_KWARG) as pho:
			cols_tpf, rows_tpf = pho.get_pixel_grid()
			wcs_tpf = pho.wcs
			filepath = pho.save_lightcurve()
			print(pho.target['ra'])
			print(pho.target['decl'])

		with BasePhotometry(DUMMY_TARGET, INPUT_DIR, OUTPUT_DIR, datasource='ffi', **DUMMY_KWARG) as pho:
			cols, rows = pho.get_pixel_grid()
			wcs_ffi = pho.wcs
			filepath = pho.save_lightcurve()
			print(pho.target['ra'])
			print(pho.target['decl'])

		with fits.open(filepath, mode='readonly', memmap=True) as hdu:
			wcs_fits_aperture = WCS(header=hdu['APERTURE'].header, relax=True)
			wcs_fits_sumimage = WCS(header=hdu['SUMIMAGE'].header, relax=True)

	target_col = 592
	target_row = 155

	print(wcs_tpf)
	print(wcs_ffi)
	print(wcs_fits_aperture)
	print(wcs_fits_sumimage)
	print("------------------------------------")

	test_pixels_ffi = [[target_col, target_row]]
	radec_ffi = wcs_ffi.all_pix2world(test_pixels_ffi, 1, ra_dec_order=True)
	print('FFI: %s' % radec_ffi)

	test_pixels_tpf = np.where((rows_tpf == target_row) & (cols_tpf == target_col))
	test_pixels_tpf = [[test_pixels_tpf[1][0], test_pixels_tpf[0][0]]]
	radec_tpf = wcs_tpf.all_pix2world(test_pixels_tpf, 1, ra_dec_order=True)
	print("TPF: %s " % radec_tpf)

	test_pixels = np.where((rows == target_row) & (cols == target_col))
	test_pixels = [[test_pixels[1][0], test_pixels[0][0]]]
	radec_fits_aperture = wcs_fits_aperture.all_pix2world(test_pixels, 0, ra_dec_order=True)
	radec_fits_sumimage = wcs_fits_sumimage.all_pix2world(test_pixels, 0, ra_dec_order=True)

	print("APERTURE: %s" % radec_fits_aperture)
	print("SUMIMAGE: %s" % radec_fits_sumimage)

	#np.testing.assert_allclose(radec_tpf, radec_ffi)
	np.testing.assert_allclose(radec_fits_aperture, radec_ffi)
	np.testing.assert_allclose(radec_fits_sumimage, radec_ffi)

	# Test the pixels in the corners of the stamp:
	Nr, Nc = cols.shape
	test_pixels = np.array([[0, 0], [Nc-1, Nr-1], [0, Nr-1], [Nc-1, 0]])
	print(test_pixels)

	# Corresponding pixels in the FFI:
	# Remember that cols and rows are 1-based.
	test_pixels_ffi = np.array([[cols[r, c]-1, rows[r, c]-1] for c, r in test_pixels])
	print(test_pixels_ffi)

	# Calculate sky-coordinates using both WCS:
	radec_ffi = wcs_ffi.all_pix2world(test_pixels_ffi, 0, ra_dec_order=True)
	radec_fits_aperture = wcs_fits_aperture.all_pix2world(test_pixels, 0, ra_dec_order=True)
	radec_fits_sumimage = wcs_fits_sumimage.all_pix2world(test_pixels, 0, ra_dec_order=True)

	# Check that the two WCS from the FITS file is the same:
	print(radec_fits_aperture - radec_fits_sumimage)
	np.testing.assert_allclose(radec_fits_aperture, radec_fits_sumimage)

	# Check that the sky-coordinates are the same:
	print(radec_ffi - radec_fits_aperture)
	np.testing.assert_allclose(radec_fits_aperture, radec_ffi)
	print(radec_ffi - radec_fits_sumimage)
	np.testing.assert_allclose(radec_fits_sumimage, radec_ffi)

#----------------------------------------------------------------------
"""
def test_cache():

	#global hdf5_cache
	print(bf)

	with TemporaryDirectory() as OUTPUT_DIR:
		print("Running with no cache...")
		#BasePhotometry.hdf5_cache = {}
		#with BasePhotometry(DUMMY_TARGET, INPUT_DIR, OUTPUT_DIR, datasource='ffi', camera=2, ccd=2, cache='none'):
		#	assert(BasePhotometry.hdf5_cache == {})

		print("Running with basic cache...")
		#hdf5_cache = {}
		with BasePhotometry(DUMMY_TARGET, INPUT_DIR, OUTPUT_DIR, datasource='ffi', camera=2, ccd=2, cache='basic') as pho:
			print("WHAT?", BasePhotometry.hdf5_cache)
			assert(pho.filepath_hdf5 in BasePhotometry.hdf5_cache)
			c = BasePhotometry.hdf5_cache.get(pho.filepath_hdf5)
			assert(c.get('_images_cube_full') is None)

		#print(hdf5_cache)
		BasePhotometry.hdf5_cache = {}
		with BasePhotometry(DUMMY_TARGET, INPUT_DIR, OUTPUT_DIR, datasource='ffi', camera=2, ccd=2, cache='full') as pho:
			c = BasePhotometry.hdf5_cache.get(pho.filepath_hdf5)
			cube = c.get('_images_cube_full')
			assert(cube.shape == (2048, 2048, 2))
"""

#----------------------------------------------------------------------
def test_tpf_with_other_target():
	sub_target = 267091131
	with TemporaryDirectory() as OUTPUT_DIR:
		with BasePhotometry(sub_target, INPUT_DIR, OUTPUT_DIR, datasource='tpf:267211065', sector=1, camera=3, ccd=2) as pho:
			assert(pho.starid == sub_target)
			assert(pho.datasource == 'tpf')

#----------------------------------------------------------------------
if __name__ == '__main__':
	test_stamp()
	test_images()
	test_backgrounds()
	test_catalog()
	test_catalog_attime()
	test_aperture()
	test_pixelflags()
	test_wcs()
	#test_cache()
	test_tpf_with_other_target()
