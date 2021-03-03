#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of BasePhotometry.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import warnings
import os.path
import numpy as np
import h5py
from tempfile import TemporaryDirectory
from astropy.io import fits
from astropy import wcs
import conftest # noqa: F401
from photometry import BasePhotometry, PixelQualityFlags, CorrectorQualityFlags
#import photometry.BasePhotometry.hdf5_cache as bf

DUMMY_TARGET = 260795451
DUMMY_KWARG = {'sector': 1, 'camera': 3, 'ccd': 2}

#--------------------------------------------------------------------------------------------------
def test_basephotometry_invalid_input(SHARED_INPUT_DIR):
	with TemporaryDirectory() as OUTPUT_DIR:

		# Test invalid datatype:
		with pytest.raises(ValueError) as e:
			with BasePhotometry(DUMMY_TARGET, SHARED_INPUT_DIR, OUTPUT_DIR, datasource='invalid', **DUMMY_KWARG):
				pass
		assert str(e.value) == "Invalid datasource: 'invalid'"

		# Test invalid cache option:
		with pytest.raises(ValueError) as e:
			with BasePhotometry(DUMMY_TARGET, SHARED_INPUT_DIR, OUTPUT_DIR, datasource='ffi', cache='invalid', **DUMMY_KWARG):
				pass
		assert str(e.value) == "Invalid cache: 'invalid'"

		# Test an input directory that does not exist:
		with pytest.raises(FileNotFoundError) as e:
			with BasePhotometry(DUMMY_TARGET, 'does/not/exist', OUTPUT_DIR, datasource='ffi', **DUMMY_KWARG):
				pass
		assert str(e.value).startswith('Not a valid input directory: ')

		# Test asking for FFI target without providing SECTOR, CAMERA and CCD:
		with pytest.raises(ValueError) as e:
			with BasePhotometry(DUMMY_TARGET, SHARED_INPUT_DIR, OUTPUT_DIR, datasource='ffi'):
				pass
		assert str(e.value) == "SECTOR, CAMERA and CCD keywords must be provided for FFI targets."

		# Test target not in the catalog:
		with pytest.raises(Exception) as e:
			with BasePhotometry(0, SHARED_INPUT_DIR, OUTPUT_DIR, datasource='ffi', **DUMMY_KWARG):
				pass
		assert str(e.value) == "Star could not be found in catalog: 0"

#--------------------------------------------------------------------------------------------------
def test_stamp(SHARED_INPUT_DIR):
	with TemporaryDirectory() as OUTPUT_DIR:
		with BasePhotometry(DUMMY_TARGET, SHARED_INPUT_DIR, OUTPUT_DIR, datasource='ffi', **DUMMY_KWARG) as pho:

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

			pho.resize_stamp(down=2)
			cols, rows = pho.get_pixel_grid()
			print('Rows:')
			print(rows)
			print(rows.shape)
			print('Cols:')
			print(cols)
			print(cols.shape)
			assert(rows.shape == (24, 20))
			assert(cols.shape == (24, 20))

			pho.resize_stamp(right=3)
			cols, rows = pho.get_pixel_grid()
			print('Rows:')
			print(rows)
			print(rows.shape)
			print('Cols:')
			print(cols)
			print(cols.shape)
			assert(rows.shape == (24, 23))
			assert(cols.shape == (24, 23))

			pho.resize_stamp(left=3)
			cols, rows = pho.get_pixel_grid()
			print('Rows:')
			print(rows)
			print(rows.shape)
			print('Cols:')
			print(cols)
			print(cols.shape)
			assert(rows.shape == (24, 26))
			assert(cols.shape == (24, 26))

			# Set a stamp that is not going to work:
			with pytest.raises(ValueError) as e:
				pho.resize_stamp(left=-100, right=-100)
			assert str(e.value) == "Invalid stamp selected"

#--------------------------------------------------------------------------------------------------
def test_stamp_width_height(SHARED_INPUT_DIR):
	with TemporaryDirectory() as OUTPUT_DIR:
		with BasePhotometry(DUMMY_TARGET, SHARED_INPUT_DIR, OUTPUT_DIR, datasource='ffi', **DUMMY_KWARG) as pho:

			print("Original")
			orig_stamp = pho._stamp
			orig_sumimage = pho.sumimage
			print(orig_stamp)

			# Change the size of the stamp:
			print("New")
			pho.resize_stamp(width=25, height=11)
			large_stamp = pho._stamp
			large_sumimage = pho.sumimage
			print(large_stamp)

			cols, rows = pho.get_pixel_grid()
			print(cols.shape, rows.shape)
			assert cols.shape == (11, 25)
			assert rows.shape == (11, 25)
			assert large_sumimage.shape == (11, 25)

			# Make the stamp the same size as the original again:
			pho.resize_stamp(width=17, height=17)
			print(pho._stamp)
			assert pho._stamp == orig_stamp
			np.testing.assert_allclose(pho.sumimage, orig_sumimage)

			# Make the stamp the same size large one, but only changing width:
			pho.resize_stamp(width=25)
			print(pho._stamp)
			cols, rows = pho.get_pixel_grid()
			assert cols.shape == (17, 25)
			assert rows.shape == (17, 25)

			# Make really large stamp now:
			pho.resize_stamp(height=25)
			print(pho._stamp)
			cols, rows = pho.get_pixel_grid()
			assert cols.shape == (25, 25)
			assert rows.shape == (25, 25)

#--------------------------------------------------------------------------------------------------
def test_images(SHARED_INPUT_DIR):
	with TemporaryDirectory() as OUTPUT_DIR:
		with BasePhotometry(DUMMY_TARGET, SHARED_INPUT_DIR, OUTPUT_DIR, datasource='ffi', **DUMMY_KWARG) as pho:

			pho._stamp = (50, 60, 50, 70)
			pho._set_stamp()

			for img in pho.images:
				assert(img.shape == (10, 20))

#--------------------------------------------------------------------------------------------------
def test_backgrounds(SHARED_INPUT_DIR):
	with TemporaryDirectory() as OUTPUT_DIR:
		with BasePhotometry(DUMMY_TARGET, SHARED_INPUT_DIR, OUTPUT_DIR, datasource='ffi', **DUMMY_KWARG) as pho:

			pho._stamp = (50, 60, 50, 70)
			pho._set_stamp()

			for img in pho.backgrounds:
				assert(img.shape == (10, 20))

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('datasource', ['tpf', 'ffi'])
def test_catalog(SHARED_INPUT_DIR, datasource):
	with TemporaryDirectory() as OUTPUT_DIR:
		with BasePhotometry(DUMMY_TARGET, SHARED_INPUT_DIR, OUTPUT_DIR, datasource=datasource, **DUMMY_KWARG) as pho:
			print(pho.catalog)
			assert(DUMMY_TARGET in pho.catalog['starid'])

			assert(pho.target['ra'] >= np.min(pho.catalog['ra']))
			assert(pho.target['ra'] <= np.max(pho.catalog['ra']))
			assert(pho.target['decl'] >= np.min(pho.catalog['dec']))
			assert(pho.target['decl'] <= np.max(pho.catalog['dec']))

			indx_main = (pho.catalog['starid'] == DUMMY_TARGET)

			# Test the real position - TODO: How do we find this?
			#np.testing.assert_allclose(pho.target_pos_column, 1978.082)
			#np.testing.assert_allclose(pho.target_pos_row, 652.5701)

			np.testing.assert_allclose(pho.catalog[indx_main]['column'], pho.target_pos_column)
			np.testing.assert_allclose(pho.catalog[indx_main]['row'], pho.target_pos_row)

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('datasource', ['tpf', 'ffi'])
def test_catalog_attime(SHARED_INPUT_DIR, datasource):
	with TemporaryDirectory() as OUTPUT_DIR:
		with BasePhotometry(DUMMY_TARGET, SHARED_INPUT_DIR, OUTPUT_DIR, datasource=datasource, **DUMMY_KWARG) as pho:

			time = pho.lightcurve['time']

			cat = pho.catalog_attime(time[0])

			assert(cat.colnames == pho.catalog.colnames)
			# TODO: Add more tests here, once we change the test input data

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('datasource', ['tpf', 'ffi'])
def test_aperture(SHARED_INPUT_DIR, datasource):
	with TemporaryDirectory() as OUTPUT_DIR:
		with BasePhotometry(DUMMY_TARGET, SHARED_INPUT_DIR, OUTPUT_DIR, datasource=datasource, **DUMMY_KWARG) as pho:

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
		with BasePhotometry(267211065, SHARED_INPUT_DIR, OUTPUT_DIR, datasource='ffi', plot=True, **DUMMY_KWARG) as pho:
			central_pixel = pho.aperture[int(np.round(pho.target_pos_row_stamp)), int(np.round(pho.target_pos_column_stamp))]
			assert central_pixel & 4 == 0, "Central pixel of this bright star should not be used in background"

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('datasource', ['tpf', 'ffi'])
def test_pixelflags(SHARED_INPUT_DIR, datasource):
	with TemporaryDirectory() as OUTPUT_DIR:
		with BasePhotometry(DUMMY_TARGET, SHARED_INPUT_DIR, OUTPUT_DIR, datasource=datasource, **DUMMY_KWARG) as pho:
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

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('datasource', ['tpf', 'ffi'])
def test_wcs(SHARED_INPUT_DIR, datasource):

	# The coordinates of the object according to MAST (TIC 8.1):
	radec_mast = [347.420628950546, -67.7328226172799]

	# Open original data products from SPOC to compare against:
	if datasource == 'ffi':
		# Use the HDF5 files to find out which FFI the final WCS is taken from:
		with h5py.File(os.path.join(SHARED_INPUT_DIR, 'sector001_camera3_ccd2.hdf5'), 'r') as hdf:
			refindx = hdf['wcs'].attrs['ref_frame']
			imgpaths = np.asarray(hdf['imagespaths'])
			ref_fname = imgpaths[refindx].decode('utf-8')
			print("FFI file: %s" % ref_fname)

		# We are actually going back to the original FITS header, to make sure
		# that nothing has changed when extracting this and saving it in the HDF5 files:
		hdr = fits.getheader(os.path.join(SHARED_INPUT_DIR, 'images', ref_fname + '.gz'), ext=1)
	else:
		hdr = fits.getheader(os.path.join(SHARED_INPUT_DIR, 'images', 'tess2018206045859-s0001-0000000260795451-0120-s_tp.fits.gz'), extname='APERTURE')

	# Create the "correct" WCS from the extracted FITS header:
	wcs_spoc = wcs.WCS(header=hdr, relax=True)

	# Run the photometry, and load the WCS from the resulting FITS file:
	with TemporaryDirectory() as tmpdir:
		with warnings.catch_warnings():
			warnings.filterwarnings('ignore', category=wcs.FITSFixedWarning)

			with BasePhotometry(DUMMY_TARGET, SHARED_INPUT_DIR, tmpdir, datasource=datasource, **DUMMY_KWARG) as pho:
				#pho.photometry() # Only needed for e.g. checking the output apertures - If enabled, also need to change to an actual photometry
				cols, rows = pho.get_pixel_grid()
				wcs_obj = pho.wcs
				filepath = pho.save_lightcurve()
				radec_target = [pho.target['ra'], pho.target['decl']]

			#report = wcs.validate(filepath)
			#print(report)

			with fits.open(filepath, mode='readonly', memmap=True) as hdu:
				radec_target_fits = [hdu[0].header['RA_OBJ'], hdu[0].header['DEC_OBJ']]
				wcs_aperture = wcs.WCS(header=hdu['APERTURE'].header, relax=True)
				wcs_sumimage = wcs.WCS(header=hdu['SUMIMAGE'].header, relax=True)

	# Check target position:
	# This actually doesn't involve the WCS directly, but is a good sanity check
	# before we go futher
	print("RA/DEC TARGET: %s" % radec_target)
	np.testing.assert_allclose(radec_target, radec_mast, rtol=0.01)
	np.testing.assert_allclose(radec_target_fits, radec_mast, rtol=0.01)

	print("------------------------------------------------------")
	print("OBJECT WCS:")
	wcs_obj.printwcs()
	print("------------------------------------------------------")
	print("APERTURE WCS:")
	wcs_aperture.printwcs()
	print("------------------------------------------------------")
	print("SUMIMAGE WCS:")
	wcs_sumimage.printwcs()
	print("------------------------------------------------------")
	print("SPOC WCS:")
	wcs_spoc.printwcs()
	print("------------------------------------------------------")

	# Test the pixels in the corners and the centre of the stamp:
	Nr, Nc = cols.shape
	test_pixels = np.array([[0, 0], [Nc-1, Nr-1], [0, Nr-1], [Nc-1, 0], [(Nc-1)//2, (Nr-1)//2]])
	print(test_pixels)

	radec_aperture = wcs_aperture.all_pix2world(test_pixels, 0, ra_dec_order=True)
	radec_sumimage = wcs_sumimage.all_pix2world(test_pixels, 0, ra_dec_order=True)

	# Check if the target corrdinates fall within the stamp:
	pix_aperture = wcs_aperture.all_world2pix([radec_mast], 0, ra_dec_order=True).squeeze()
	pix_sumimage = wcs_sumimage.all_world2pix([radec_mast], 0, ra_dec_order=True).squeeze()

	# The pixels coordinates in FFIs are in REAL pixels (the number they have on the detector)
	# and not in the small stamp. Therefore, we have to convert them to the
	# corresponding values in the stamp. For TPFs, we dont have to do that conversion:
	if datasource == 'tpf':
		radec_obj = wcs_obj.all_pix2world(test_pixels, 0, ra_dec_order=True)
		radec_spoc = wcs_spoc.all_pix2world(test_pixels, 0, ra_dec_order=True)

		pix_obj = wcs_obj.all_world2pix([radec_mast], 0, ra_dec_order=True).squeeze()
		pix_spoc = wcs_spoc.all_world2pix([radec_mast], 0, ra_dec_order=True).squeeze()
	else:
		# Corresponding pixels in the FFI:
		# Remember that cols and rows are 1-based.
		test_pixels_ffi = np.array([[cols[r, c]-1, rows[r, c]-1] for c, r in test_pixels])
		print(test_pixels_ffi)

		radec_obj = wcs_obj.all_pix2world(test_pixels_ffi, 0, ra_dec_order=True)
		radec_spoc = wcs_spoc.all_pix2world(test_pixels_ffi, 0, ra_dec_order=True)

		pix_obj = wcs_obj.all_world2pix([radec_mast], 0, ra_dec_order=True).squeeze()
		pix_spoc = wcs_spoc.all_world2pix([radec_mast], 0, ra_dec_order=True).squeeze()

		# Subtract the (real) pixel number of the (0,0) pixel:
		pix_spoc[0] -= test_pixels_ffi[0,0]
		pix_spoc[1] -= test_pixels_ffi[0,1]
		pix_obj[0] -= test_pixels_ffi[0,0]
		pix_obj[1] -= test_pixels_ffi[0,1]

	# Print out extracted values for debugging:
	print("RA/DEC SPOC:     %s" % radec_spoc)
	print("RA/DEC OBJECT:   %s" % radec_obj)
	print("RA/DEC APERTURE: %s" % radec_aperture)
	print("RA/DEC SUMIMAGE: %s" % radec_sumimage)

	print("PIXELS SPOC:     %s" % pix_spoc)
	print("PIXELS OBJECT:   %s" % pix_obj)
	print("PIXELS APERTURE: %s" % pix_aperture)
	print("PIXELS SUMIMAGE: %s" % pix_sumimage)

	# Check that everything agrees with SPOC and each other:
	np.testing.assert_allclose(radec_obj, radec_spoc)
	np.testing.assert_allclose(radec_aperture, radec_spoc)
	np.testing.assert_allclose(radec_sumimage, radec_spoc)

	# Check that everything agrees with SPOC and each other:
	np.testing.assert_allclose(pix_obj, pix_spoc)
	np.testing.assert_allclose(pix_aperture, pix_spoc)
	np.testing.assert_allclose(pix_sumimage, pix_spoc)

	# Check if the target corrdinates fall within the stamp:
	assert -0.5 <= pix_obj[0] <= Nc-0.5
	assert -0.5 <= pix_obj[1] <= Nr-0.5

#--------------------------------------------------------------------------------------------------
"""
def test_cache(SHARED_INPUT_DIR):
	#global hdf5_cache
	print(bf)

	with TemporaryDirectory() as OUTPUT_DIR:
		print("Running with no cache...")
		#BasePhotometry.hdf5_cache = {}
		#with BasePhotometry(DUMMY_TARGET, SHARED_INPUT_DIR, OUTPUT_DIR, datasource='ffi', camera=2, ccd=2, cache='none'):
		#	assert(BasePhotometry.hdf5_cache == {})

		print("Running with basic cache...")
		#hdf5_cache = {}
		with BasePhotometry(DUMMY_TARGET, SHARED_INPUT_DIR, OUTPUT_DIR, datasource='ffi', camera=2, ccd=2, cache='basic') as pho:
			print("WHAT?", BasePhotometry.hdf5_cache)
			assert(pho.filepath_hdf5 in BasePhotometry.hdf5_cache)
			c = BasePhotometry.hdf5_cache.get(pho.filepath_hdf5)
			assert(c.get('_images_cube_full') is None)

		#print(hdf5_cache)
		BasePhotometry.hdf5_cache = {}
		with BasePhotometry(DUMMY_TARGET, SHARED_INPUT_DIR, OUTPUT_DIR, datasource='ffi', camera=2, ccd=2, cache='full') as pho:
			c = BasePhotometry.hdf5_cache.get(pho.filepath_hdf5)
			cube = c.get('_images_cube_full')
			assert(cube.shape == (2048, 2048, 2))
"""

#--------------------------------------------------------------------------------------------------
def test_tpf_with_other_target(SHARED_INPUT_DIR):
	sub_target = 267091131
	with TemporaryDirectory() as OUTPUT_DIR:
		with BasePhotometry(sub_target, SHARED_INPUT_DIR, OUTPUT_DIR, datasource='tpf:267211065', sector=1, camera=3, ccd=2) as pho:
			assert(pho.starid == sub_target)
			assert(pho.datasource == 'tpf')

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
