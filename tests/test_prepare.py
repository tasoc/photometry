#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of Prepare Photometry.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import tempfile
import logging
import h5py
import os.path
import conftest # noqa: F401
from photometry import prepare
from photometry.utilities import TqdmLoggingHandler

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
def test_prepare_photometry(SHARED_INPUT_DIR):

	# The known sizes for input-data:
	Ntimes = 4
	img_size = (2048, 2048)

	# Setup logging:
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	console = TqdmLoggingHandler()
	console.setFormatter(formatter)
	logger_parent = logging.getLogger('photometry')
	logger_parent.setLevel(logging.INFO)
	if not logger_parent.hasHandlers():
		logger_parent.addHandler(console)

	with tempfile.NamedTemporaryFile() as tmpfile:
		# Run prepare_photometry and save output to temp-file:
		prepare.prepare_photometry(SHARED_INPUT_DIR, sectors=1, cameras=3, ccds=2, output_file=tmpfile.name)

		tmpfile.flush()
		assert os.path.exists(tmpfile.name + '.hdf5'), "HDF5 was not created"

		with h5py.File(tmpfile.name + '.hdf5', 'r') as hdf:
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
				dset = '%04d' % k
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

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
