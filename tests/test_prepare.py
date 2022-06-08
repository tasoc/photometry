#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of Prepare Photometry.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import logging
import tempfile
import h5py
import os.path
import numpy as np
from conftest import capture_cli
from photometry import prepare

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')

#--------------------------------------------------------------------------------------------------
def test_prepare_photometry_invalid_input_dir():

	invalid_input_dir = os.path.join(INPUT_DIR, 'does', 'not', 'exist')
	print(invalid_input_dir)
	with pytest.raises(NotADirectoryError):
		prepare.prepare_photometry(invalid_input_dir)

	not_a_directory = os.path.join(INPUT_DIR, 'catalog_sector001_camera1_ccd1.sqlite')
	print(not_a_directory)
	with pytest.raises(NotADirectoryError):
		prepare.prepare_photometry(not_a_directory)

#--------------------------------------------------------------------------------------------------
def hdf5_file_valid(fname, sector=None, camera=None, ccd=None, Ntimes=4):

	# The known sizes for input-data:
	img_size = (2048, 2048)

	with h5py.File(fname, 'r') as hdf:
		# Check that the required datasets exists:
		assert 'time' in hdf and isinstance(hdf['time'], h5py.Dataset), "There should be an TIME dataset"
		assert 'timecorr' in hdf and isinstance(hdf['timecorr'], h5py.Dataset), "There should be an TIMECORR dataset"
		assert 'cadenceno' in hdf and isinstance(hdf['cadenceno'], h5py.Dataset), "There should be an CADENCENO dataset"
		assert 'quality' in hdf and isinstance(hdf['quality'], h5py.Dataset), "There should be an QUALITY dataset"
		assert 'sumimage' in hdf and isinstance(hdf['sumimage'], h5py.Dataset), "There should be an SUMIMAGE dataset"

		# Check that the required groups exists:
		assert 'images' in hdf and isinstance(hdf['images'], h5py.Group), "There should be an IMAGES group"
		assert 'images_err' in hdf and isinstance(hdf['images_err'], h5py.Group), "There should be an IMAGES_ERR group"
		assert 'backgrounds' in hdf and isinstance(hdf['backgrounds'], h5py.Group), "There should be a BACKGROUNDS group"
		assert 'pixel_flags' in hdf and isinstance(hdf['pixel_flags'], h5py.Group), "There should be a PIXEL_FLAGS group"
		assert 'wcs' in hdf and isinstance(hdf['wcs'], h5py.Group), "There should be a WCS group"

		# Check size of datasets:
		assert len(hdf['time']) == Ntimes, "TIME does not have the correct size"
		assert len(hdf['timecorr']) == Ntimes, "TIMECORR does not have the correct size"
		assert len(hdf['cadenceno']) == Ntimes, "CADENCENO does not have the correct size"
		assert len(hdf['quality']) == Ntimes, "QUALITY does not have the correct size"
		assert hdf['sumimage'].shape == img_size, "SUMIMAGE does not have the correct size"

		# Check size of groups:
		for k in range(Ntimes):
			dset = f'{k:04d}'
			assert 'images/' + dset in hdf and isinstance(hdf['images/' + dset], h5py.Dataset), "IMAGES dset=" + dset + " does not exist"
			assert 'images_err/' + dset in hdf and isinstance(hdf['images_err/' + dset], h5py.Dataset), "IMAGES_ERR dset=" + dset + " does not exist"
			assert 'backgrounds/' + dset in hdf and isinstance(hdf['backgrounds/' + dset], h5py.Dataset), "BACKGROUNDS dset=" + dset + " does not exist"
			assert 'pixel_flags/' + dset in hdf and isinstance(hdf['pixel_flags/' + dset], h5py.Dataset), "PIXEL_FLAGS dset=" + dset + " does not exist"
			assert 'wcs/' + dset in hdf and isinstance(hdf['wcs/' + dset], h5py.Dataset), "WCS dset=" + dset + " does not exist"

			# Check sizes of datasets in depth:
			assert hdf['images/' + dset].shape == img_size, "IMAGES dset=" + dset + " does not have the correct size"
			assert hdf['images_err/' + dset].shape == img_size, "IMAGES_ERR dset=" + dset + " does not have the correct size"
			assert hdf['backgrounds/' + dset].shape == img_size, "BACKGROUNDS dset=" + dset + " does not have the correct size"
			assert hdf['pixel_flags/' + dset].shape == img_size, "PIXEL_FLAGS dset=" + dset + " does not have the correct size"

		# Check headers:
		timestamps = np.asarray(hdf['time']) - np.asarray(hdf['timecorr'])

		if sector is not None:
			assert hdf['images'].attrs['SECTOR'] == sector
		if camera is not None:
			assert hdf['images'].attrs['CAMERA'] == camera
		if ccd is not None:
			assert hdf['images'].attrs['CCD'] == ccd

		assert int(hdf['images'].attrs['CADENCE']) == np.round(86400*np.median(np.diff(timestamps)))

#--------------------------------------------------------------------------------------------------
def test_prepare_photometry(caplog, SHARED_INPUT_DIR):
	with tempfile.NamedTemporaryFile() as tmpfile:
		# Silence logger error (Sector reference time outside timespan of data)
		with caplog.at_level(logging.CRITICAL):
			# Run prepare_photometry and save output to temp-file
			prepare.prepare_photometry(SHARED_INPUT_DIR,
				sectors=1,
				cameras=3,
				ccds=2,
				output_file=tmpfile.name)

		tmpfile.flush()
		assert os.path.isfile(tmpfile.name + '.hdf5'), "HDF5 was not created"

		hdf5_file_valid(tmpfile.name + '.hdf5', sector=1, camera=3, ccd=2)

#--------------------------------------------------------------------------------------------------
def test_run_prepare_photometry(PRIVATE_INPUT_DIR):

	hdf5file = os.path.join(PRIVATE_INPUT_DIR, 'sector001_camera3_ccd2.hdf5')

	# Delete existing HDF5-file:
	os.remove(hdf5file)
	assert not os.path.exists(hdf5file), "HDF5 file was not removed correctly"

	out, err, exitcode = capture_cli('run_prepare_photometry.py', params=[
		'--sector=1',
		'--camera=3',
		'--ccd=2',
		PRIVATE_INPUT_DIR
	])

	assert exitcode == 0
	assert os.path.isfile(hdf5file), "HDF5 was not created"

	hdf5_file_valid(hdf5file, sector=1, camera=3, ccd=2)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
