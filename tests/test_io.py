#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of photometry.io.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os
import configparser
from astropy.units import Unit
from astropy.wcs import WCS
from astropy.nddata import CCDData, StdDevUncertainty
import conftest # noqa: F401
from photometry import io

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('sector', [1,11,27])
def test_FFIImage(SHARED_INPUT_DIR, sector):

	files = io.find_ffi_files(os.path.join(SHARED_INPUT_DIR, 'images'), sector=sector)
	for fpath in files:
		print(fpath)
		img = io.FFIImage(fpath)
		print(img)

		assert img.shape == (2048, 2048)
		assert img.unit == Unit("electron / s")
		assert img.is_tess
		assert img.data.dtype == 'float32'

		assert isinstance(img.header, dict)
		assert 'FFIINDEX' in img.header

		assert isinstance(img.wcs, WCS)

		assert isinstance(img.uncertainty, StdDevUncertainty)
		#assert img.uncertainty.value.shape == img.shape
		#assert img.uncertainty.value.dtype == img.dtype

		assert isinstance(img.smear, CCDData)
		assert img.smear.shape == (10, 2048)
		assert img.smear.unit == img.unit

		assert isinstance(img.vsmear, CCDData)
		assert img.vsmear.shape == (10, 2048)
		assert img.vsmear.unit == img.unit

#--------------------------------------------------------------------------------------------------
def test_load_settings():

	settings = io.load_settings()
	assert isinstance(settings, configparser.ConfigParser)
	assert settings.getboolean('fixes', 'time_offset', fallback=True) # Actually checking value

	# Try to run the exact same query, and it should now be taken over by the cache:
	hits_before = io.load_settings.cache_info().hits
	settings2 = io.load_settings()
	assert settings2 == settings
	assert io.load_settings.cache_info().hits == hits_before+1

#--------------------------------------------------------------------------------------------------
def test_load_sector_settings():

	settings = io.load_sector_settings(2)
	print(settings)
	assert isinstance(settings, dict)
	assert int(settings['sector']) == 2

	# Try to run the exact same query, and it should now be taken over by the cache:
	hits_before = io.load_sector_settings.cache_info().hits
	settings2 = io.load_sector_settings(2)
	assert settings2 == settings
	assert io.load_sector_settings.cache_info().hits == hits_before+1

	settings = io.load_sector_settings()
	print(settings)
	assert isinstance(settings, dict)
	assert 'sectors' in settings

	sectors = []
	for key, value in settings['sectors'].items():
		# Make sure they contain what they should:
		assert isinstance(value, dict)
		assert 'sector' in value
		assert 'reference_time' in value
		assert 'ffi_cadence' in value
		assert value['ffi_cadence'] in (1800, 600, 200), "Invalid FFI Cadence"

		# Ensure that sector numbers are unique:
		assert value['sector'] not in sectors
		sectors.append(value['sector'])

#--------------------------------------------------------------------------------------------------
def test_find_ffi_files(SHARED_INPUT_DIR):

	files = io.find_ffi_files(SHARED_INPUT_DIR)
	assert len(files) == 12

	files = io.find_ffi_files(SHARED_INPUT_DIR, camera=1)
	assert len(files) == 4

	files = io.find_ffi_files(SHARED_INPUT_DIR, camera=3)
	assert len(files) == 4

	files = io.find_ffi_files(SHARED_INPUT_DIR, sector=27)
	assert len(files) == 2

#--------------------------------------------------------------------------------------------------
def test_find_tpf_files(SHARED_INPUT_DIR):
	"""
	The current list of test-files are:

	starid    sector  camera  ccd  cadence
	267211065      1       3    2      120
	260795451      1       3    2      120
	 25155310      1       1    4      120  alert
	 25155310     27       4    1      120
	 25155310     27       4    1       20
	"""

	# Find all TPF files in input dir:
	files = io.find_tpf_files(SHARED_INPUT_DIR)
	print(files)
	assert len(files) == 5

	# Find file with specific starid (only one exists)
	files = io.find_tpf_files(SHARED_INPUT_DIR, starid=267211065)
	print(files)
	assert len(files) == 1

	# Find TPFs from sector 1 (2 regular, 1 alert-data):
	files = io.find_tpf_files(SHARED_INPUT_DIR, sector=1)
	print(files)
	assert len(files) == 3

	# Limit with findmax (only the first one should be returned):
	files2 = io.find_tpf_files(SHARED_INPUT_DIR, sector=1, findmax=1)
	print(files2)
	assert len(files2) == 1
	assert files2[0] == files[0]

	# Find TPFs for starid with both 120 and 20s cadences and a alert data file:
	files = io.find_tpf_files(SHARED_INPUT_DIR, starid=25155310)
	print(files)
	assert len(files) == 3

	# Test files from sector 27, with both 120 and 20s cadences:
	files = io.find_tpf_files(SHARED_INPUT_DIR, sector=27)
	print(files)
	assert len(files) == 2

	# Test files from sector 27, with only 120s cadence:
	files = io.find_tpf_files(SHARED_INPUT_DIR, sector=27, cadence=120)
	print(files)
	assert len(files) == 1

	# Test files from sector 27, with only 20s cadence:
	files = io.find_tpf_files(SHARED_INPUT_DIR, sector=27, cadence=20)
	print(files)
	assert len(files) == 1

	files = io.find_tpf_files(SHARED_INPUT_DIR, sector=1, camera=3)
	print(files)
	assert len(files) == 2

	# Find TPFs from sector 1, on camera 2, which should have no files:
	files = io.find_tpf_files(SHARED_INPUT_DIR, sector=1, camera=2)
	print(files)
	assert len(files) == 0

	# Test files from sector 27 on CCD 4, which should have no files:
	files = io.find_tpf_files(SHARED_INPUT_DIR, sector=27, ccd=4)
	print(files)
	assert len(files) == 0

	# Test files from sector 27, with both 120 and 20s cadences:
	files = io.find_tpf_files(SHARED_INPUT_DIR, sector=27, ccd=1)
	print(files)
	assert len(files) == 2

	# Test files from sector 27, with both 120 and 20s cadences:
	files = io.find_tpf_files(SHARED_INPUT_DIR, sector=27, ccd=1, findmax=1)
	print(files)
	assert len(files) == 1

	# Test the cache:
	hits_before = io._find_tpf_files.cache_info().hits
	files2 = io.find_tpf_files(SHARED_INPUT_DIR, sector=27, ccd=1, findmax=1)
	assert files2 == files
	assert io._find_tpf_files.cache_info().hits == hits_before+1

	# Test with invalid cadence:
	with pytest.raises(ValueError) as e:
		io.find_tpf_files(SHARED_INPUT_DIR, sector=27, cadence=123345)
	assert str(e.value) == "Invalid cadence. Must be either 20 or 120."

#--------------------------------------------------------------------------------------------------
def test_find_hdf5_files(SHARED_INPUT_DIR):

	files = io.find_hdf5_files(SHARED_INPUT_DIR)
	assert len(files) == 2

	files = io.find_hdf5_files(SHARED_INPUT_DIR, sector=1)
	assert len(files) == 2

	files = io.find_hdf5_files(SHARED_INPUT_DIR, camera=1)
	assert len(files) == 1

	files = io.find_hdf5_files(SHARED_INPUT_DIR, sector=1, camera=3)
	assert len(files) == 1

	files = io.find_hdf5_files(SHARED_INPUT_DIR, sector=1, camera=3, ccd=2)
	assert len(files) == 1

	# Try to run the exact same query, and it should now be taken over by the cache:
	hits_before = io.find_hdf5_files.cache_info().hits
	files2 = io.find_hdf5_files(SHARED_INPUT_DIR, sector=1, camera=3, ccd=2)
	assert files2 == files
	assert io.find_hdf5_files.cache_info().hits == hits_before+1

#--------------------------------------------------------------------------------------------------
def test_find_catalog_files(SHARED_INPUT_DIR):

	files = io.find_catalog_files(SHARED_INPUT_DIR)
	print(files)
	assert len(files) == 3

	files = io.find_catalog_files(SHARED_INPUT_DIR, sector=1)
	print(files)
	assert len(files) == 2

	files = io.find_catalog_files(SHARED_INPUT_DIR, sector=27)
	print(files)
	assert len(files) == 1

	files = io.find_catalog_files(SHARED_INPUT_DIR, camera=1)
	print(files)
	assert len(files) == 1

	files = io.find_catalog_files(SHARED_INPUT_DIR, sector=1, camera=3, ccd=2)
	print(files)
	assert len(files) == 1

	# Try to run the exact same query, and it should now be taken over by the cache:
	hits_before = io.find_catalog_files.cache_info().hits
	files2 = io.find_catalog_files(SHARED_INPUT_DIR, sector=1, camera=3, ccd=2)
	assert files2 == files
	assert io.find_catalog_files.cache_info().hits == hits_before+1

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
