#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of time offset fixes.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import numpy as np
import os.path
from astropy.table import Table
import conftest # noqa: F401
from photometry import fixes
from photometry.io import load_settings

TOFFDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'input', 'time_offset'))

#--------------------------------------------------------------------------------------------------
def test_fixes_time_offset_invalid_input():
	time = np.linspace(1000, 2000, 100)

	# Check that it raises an KeyError for missing CAMERA:
	hdr = {'DATA_REL': 1}
	with pytest.raises(KeyError):
		fixes.time_offset(time, hdr)

	# Check that it raises an KeyError for missing DATA_REL:
	hdr = {'CAMERA': 1}
	with pytest.raises(KeyError):
		fixes.time_offset(time, hdr)

	# Check that it raises an ValueError for invalid TIMEPOS:
	hdr = {'DATA_REL': 27, 'CAMERA': 1}
	with pytest.raises(ValueError):
		fixes.time_offset(time, hdr, timepos='invalid-input')

	# The cases of data release 27 and 29 and no PROCVER:
	hdr = {'DATA_REL': 27, 'CAMERA': 1, 'PROCVER': None}
	with pytest.raises(ValueError):
		fixes.time_offset(time, hdr)

	hdr = {'DATA_REL': 29, 'CAMERA': 2, 'PROCVER': None}
	with pytest.raises(ValueError):
		fixes.time_offset(time, hdr)

#--------------------------------------------------------------------------------------------------
def test_fixes_time_offset_s20():

	# Load list of timestamps from light curve from sector 20 and the corresponding
	# new timestamps delivered by SPOC.
	# The table meat-data contains the primary FITS header of the timestamps to be corrected.
	tab = Table.read(os.path.join(TOFFDIR, 'time_offset_s20.ecsv.gz'), format='ascii.ecsv')
	hdr = tab.meta

	time1_corrected, fixed = fixes.time_offset(tab['time1'], hdr, datatype='tpf', timepos='mid', return_flag=True)
	assert fixed, "S20v1 data should be fixed"
	np.testing.assert_allclose(time1_corrected, tab['time2'], equal_nan=True, rtol=1e-11, atol=1e-11)

#--------------------------------------------------------------------------------------------------
def test_fixes_time_offset_s21():

	# Load list of timestamps from light curve from sector 21 and the corresponding
	# new timestamps delivered by SPOC.
	# The table meat-data contains the primary FITS header of the timestamps to be corrected.
	tab = Table.read(os.path.join(TOFFDIR, 'time_offset_s21.ecsv.gz'), format='ascii.ecsv')
	hdr = tab.meta

	time1_corrected, fixed = fixes.time_offset(tab['time1'], hdr, datatype='tpf', timepos='mid', return_flag=True)
	assert fixed, "S21v1 data should be fixed"
	np.testing.assert_allclose(time1_corrected, tab['time2'], equal_nan=True, rtol=1e-11, atol=1e-11)

#--------------------------------------------------------------------------------------------------
def test_fixes_time_offset_ffis():

	# Load list of timestamps from FFIs from several sectors and data releases:
	tab = Table.read(os.path.join(TOFFDIR, 'time_offset_ffis.ecsv'), format='ascii.ecsv')

	# Loop through the tables, create a fake header of the FFI
	# and check that we get the same result from the correction
	# as the updated official data products:
	for row in tab:
		hdr = {
			'DATA_REL': row['data_rel'],
			'PROCVER': row['procver'],
			'CAMERA': row['camera'],
			'CCD': row['ccd']
		}
		print(hdr)

		time_mid = 0.5*(row['time_start'] + row['time_stop'])

		time1_start_corrected, fixed = fixes.time_offset(row['time_start'], hdr, datatype='ffi', timepos='start', return_flag=True)
		assert fixed, "S20v1 data should be fixed"

		time1_mid_corrected, fixed = fixes.time_offset(time_mid, hdr, datatype='ffi', timepos='mid', return_flag=True)
		assert fixed, "S20v1 data should be fixed"

		time1_stop_corrected, fixed = fixes.time_offset(row['time_stop'], hdr, datatype='ffi', timepos='end', return_flag=True)
		assert fixed, "S20v1 data should be fixed"

		print( (time1_start_corrected - row['time_start_corrected'])*86400 )
		print( (time1_mid_corrected - row['time_mid_corrected'])*86400 )
		print( (time1_stop_corrected - row['time_stop_corrected'])*86400 )

		np.testing.assert_allclose(time1_start_corrected, row['time_start_corrected'], equal_nan=True, rtol=1e-11, atol=1e-11)
		np.testing.assert_allclose(time1_mid_corrected, row['time_mid_corrected'], equal_nan=True, rtol=1e-11, atol=1e-11)
		np.testing.assert_allclose(time1_stop_corrected, row['time_stop_corrected'], equal_nan=True, rtol=1e-11, atol=1e-11)

#--------------------------------------------------------------------------------------------------
def test_fixes_time_offset_needed():

	time = np.linspace(1000, 2000, 100)
	hdr = {'DATA_REL': 0, 'CAMERA': 1, 'CCD': 3, 'PROCVER': 'shouldnt-matter'}

	# Early releases that should be corrected:
	for data_rel in range(1, 27):
		hdr20 = dict(hdr, DATA_REL=data_rel)
		print(hdr20)
		assert fixes.time_offset(time, hdr20, return_flag=True)[1]

	# Sector 20 that should be corrected:
	for procver in ('spoc-4.0.14-20200108', 'spoc-4.0.15-20200114', 'spoc-4.0.17-20200130'):
		hdr20 = dict(hdr, SECTOR=20, DATA_REL=27, PROCVER=procver)
		print(hdr20)
		assert fixes.time_offset(time, hdr20, return_flag=True)[1]

	# Sector 20 that should NOT be corrected:
	for procver in ('spoc-4.0.26-20200323', 'spoc-4.0.27-20200326'):
		hdr20 = dict(hdr, SECTOR=20, DATA_REL=27, PROCVER=procver)
		print(hdr20)
		assert not fixes.time_offset(time, hdr20, return_flag=True)[1]

	# Sector 20 that should not be corrected (hypothetical future data relase #99):
	hdr20 = dict(hdr, SECTOR=20, DATA_REL=99)
	print(hdr20)
	assert not fixes.time_offset(time, hdr20, return_flag=True)[1]

	# Sector 21 that should be corrected:
	for procver in ('spoc-4.0.17-20200130', 'spoc-4.0.20-20200220', 'spoc-4.0.21-20200227'):
		hdr21 = dict(hdr, SECTOR=21, DATA_REL=29, PROCVER=procver)
		print(hdr21)
		assert fixes.time_offset(time, hdr21, return_flag=True)[1]

	# Sector 21 that should NOT be corrected:
	hdr21 = dict(hdr, SECTOR=21, DATA_REL=29, PROCVER='spoc-4.0.27-20200326')
	print(hdr21)
	assert not fixes.time_offset(time, hdr21, return_flag=True)[1]

	# Sector 21 that should not be corrected (hypothetical future data relase #99):
	hdr21 = dict(hdr, SECTOR=21, DATA_REL=99)
	print(hdr21)
	assert not fixes.time_offset(time, hdr21, return_flag=True)[1]

#--------------------------------------------------------------------------------------------------
def test_fixes_time_offset_apply():

	time = np.linspace(1000, 2000, 200, dtype='float64')
	hdr = {'DATA_REL': 27, 'CAMERA': 1, 'CCD': 3, 'PROCVER': 'spoc-4.0.15-20200114'}

	time_corrected_mid1, fixed1 = fixes.time_offset(time, hdr, datatype='tpf', return_flag=True)
	time_corrected_mid2, fixed2 = fixes.time_offset(time, hdr, datatype='tpf', timepos='mid', return_flag=True)

	# Calling with and without timepos should yield the same result:
	assert fixed1 == fixed2
	np.testing.assert_allclose(time_corrected_mid1, time_corrected_mid2)

	# Test mid-time correction:
	print( (time - time_corrected_mid1)*86400 )
	np.testing.assert_allclose((time_corrected_mid1 - time)*86400, -1.979)

	# Test start-time correction:
	time_corrected_start = fixes.time_offset(time, hdr, datatype='tpf', timepos='start')
	print( (time - time_corrected_start)*86400 )
	np.testing.assert_allclose((time_corrected_start - time)*86400, -1.969)

	# Test end-time correction:
	time_corrected_end = fixes.time_offset(time, hdr, datatype='tpf', timepos='end')
	print( (time - time_corrected_end)*86400 )
	np.testing.assert_allclose((time_corrected_end - time)*86400, -1.989)

#--------------------------------------------------------------------------------------------------
def test_fixes_time_offset_settings():

	# Early releases that should be corrected:
	hdr = {'DATA_REL': 1, 'CAMERA': 1, 'CCD': 3, 'PROCVER': 'shouldnt-matter'}
	time = np.linspace(1000, 2000, 200, dtype='float64')

	time2, applied = fixes.time_offset(time, hdr, datatype='tpf', timepos='mid', return_flag=True)
	print(applied)
	assert applied

	# Change the settings in-memory:
	# This also relies on the caching to work!
	settings = load_settings()
	settings['fixes']['time_offset'] = 'False'

	# When calling the function now, it shouldn't do anything:
	time3, applied = fixes.time_offset(time, hdr, datatype='tpf', timepos='mid', return_flag=True)
	print(applied)
	assert not applied
	np.testing.assert_allclose(time3, time)

	# Change the settings back to original:
	settings['fixes']['time_offset'] = 'True'

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
