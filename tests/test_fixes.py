#!/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of time offset fixes.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import numpy as np
import os.path
import sys
from astropy.table import Table
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from photometry import fixes

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
	with pytest.raises(Exception):
		fixes.time_offset(time, hdr)

	hdr = {'DATA_REL': 29, 'CAMERA': 2, 'PROCVER': None}
	with pytest.raises(Exception):
		fixes.time_offset(time, hdr)

#--------------------------------------------------------------------------------------------------
def test_fixes_time_offset_s20():

	#with fits.open(r'E:\time_offset\tess2019357164649-s0020-0000000004164018-0165-s_lc-v01.fits.gz') as hdu:
	#	hdr1 = hdu[0].header
	#	time1 = np.atleast_2d(hdu[1].data['TIME']).T

	#with fits.open(r'E:\time_offset\tess2019357164649-s0020-0000000004164018-0165-s_lc-v02.fits.gz') as hdu:
	#	time2 = np.atleast_2d(hdu[1].data['TIME']).T

	#with open(os.path.join(TOFFDIR, 's20_hdr.json'), 'w') as fid:
	#	json.dump(dict(hdr1), fid)

	#A = np.concatenate((time1, time2), axis=1)
	#np.savetxt(os.path.join(TOFFDIR, 'time_offset_s20.txt.gz'), A)

	time1, time2 = np.loadtxt(os.path.join(TOFFDIR, 'time_offset_s20.txt.gz'), unpack=True)
	with open(os.path.join(TOFFDIR, 's20_hdr.json'), 'r') as fid:
		hdr1 = json.load(fid)

	time1_corrected, fixed = fixes.time_offset(time1, hdr1, datatype='tpf', timepos='mid', return_flag=True)
	assert fixed, "S20v1 data should be fixed"
	np.testing.assert_allclose(time1_corrected, time2, equal_nan=True, rtol=1e-11, atol=1e-11)

#--------------------------------------------------------------------------------------------------
def test_fixes_time_offset_ffis():

	# Load list of timestamps from FFIs from several sectors and data releases:
	tab = Table.read(os.path.join(TOFFDIR, 'ffis.ecsv'), format='ascii.ecsv')

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
def test_fixes_time_offset_s21():

	#with fits.open(r'E:\time_offset\tess2020020091053-s0021-0000000001096672-0167-s_lc-v01.fits.gz') as hdu:
	#	hdr1 = hdu[0].header
	#	time1 = np.atleast_2d(hdu[1].data['TIME']).T

	#with fits.open(r'E:\time_offset\tess2020020091053-s0021-0000000001096672-0167-s_lc-v02.fits.gz') as hdu:
	#	time2 = np.atleast_2d(hdu[1].data['TIME']).T

	#with open('input/time_offset/s21_hdr.json', 'w') as fid:
	#	json.dump(dict(hdr1), fid)

	#A = np.concatenate((time1, time2), axis=1)
	#np.savetxt('input/time_offset/time_offset_s21.txt.gz', A)

	time1, time2 = np.loadtxt(os.path.join(TOFFDIR, 'time_offset_s21.txt.gz'), unpack=True)
	with open(os.path.join(TOFFDIR, 's21_hdr.json'), 'r') as fid:
		hdr1 = json.load(fid)

	time1_corrected, fixed = fixes.time_offset(time1, hdr1, datatype='tpf', timepos='mid', return_flag=True)
	assert fixed, "S21v1 data should be fixed"
	np.testing.assert_allclose(time1_corrected, time2, equal_nan=True, rtol=1e-11, atol=1e-11)

#--------------------------------------------------------------------------------------------------
def test_fixes_time_offset():

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
		assert fixes.time_offset(time, hdr20, return_flag=True)[1] == False

	# Sector 20 that should not be corrected (hypothetical future data relase #99):
	hdr20 = dict(hdr, SECTOR=20, DATA_REL=99)
	print(hdr20)
	assert fixes.time_offset(time, hdr20, return_flag=True)[1] == False

	# Sector 21 that should be corrected:
	for procver in ('spoc-4.0.17-20200130', 'spoc-4.0.20-20200220', 'spoc-4.0.21-20200227'):
		hdr21 = dict(hdr, SECTOR=21, DATA_REL=29, PROCVER=procver)
		print(hdr21)
		assert fixes.time_offset(time, hdr21, return_flag=True)[1]

	# Sector 21 that should NOT be corrected:
	hdr21 = dict(hdr, SECTOR=21, DATA_REL=29, PROCVER='spoc-4.0.27-20200326')
	print(hdr21)
	assert fixes.time_offset(time, hdr21, return_flag=True)[1] == False

	# Sector 21 that should not be corrected (hypothetical future data relase #99):
	hdr21 = dict(hdr, SECTOR=21, DATA_REL=99)
	print(hdr21)
	assert fixes.time_offset(time, hdr21, return_flag=True)[1] == False

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
if __name__ == '__main__':
	pytest.main([__file__])
