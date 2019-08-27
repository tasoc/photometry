#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from photometry.utilities import (move_median_central, find_ffi_files, find_tpf_files,
	find_hdf5_files, find_catalog_files, load_ffi_fits,
	sphere_distance, radec_to_cartesian, cartesian_to_radec,
	rms_timescale)

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')

#----------------------------------------------------------------------
def test_move_median_central():

	x_1d = np.array([4, 2, 2, 0, 0, np.NaN, 0, 2, 2, 4])
	result_1d = move_median_central(x_1d, 3)
	expected_1d = np.array([3, 2, 2, 0, 0, 0, 1, 2, 2, 3])
	np.testing.assert_allclose(result_1d, expected_1d)

	x_2d = np.tile(x_1d, (10, 1))
	result_2d = move_median_central(x_2d, 3, axis=1)
	expected_2d = np.tile(expected_1d, (10, 1))
	np.testing.assert_allclose(result_2d, expected_2d)

#----------------------------------------------------------------------
def test_find_ffi_files():

	files = find_ffi_files(INPUT_DIR)
	assert(len(files) == 8)

	files = find_ffi_files(INPUT_DIR, camera=1)
	assert(len(files) == 4)

	files = find_ffi_files(INPUT_DIR, camera=3)
	assert(len(files) == 4)

#----------------------------------------------------------------------
def test_find_tpf_files():

	files = find_tpf_files(INPUT_DIR)
	assert(len(files) == 2)

	files = find_tpf_files(INPUT_DIR, starid=267211065)
	assert(len(files) == 1)

#----------------------------------------------------------------------
def test_find_hdf5_files():

	files = find_hdf5_files(INPUT_DIR)
	assert(len(files) == 2)

	files = find_hdf5_files(INPUT_DIR, sector=1)
	assert(len(files) == 2)

	files = find_hdf5_files(INPUT_DIR, camera=1)
	assert(len(files) == 1)

	files = find_hdf5_files(INPUT_DIR, sector=1, camera=3)
	assert(len(files) == 1)

#----------------------------------------------------------------------
def test_find_catalog_files():

	files = find_catalog_files(INPUT_DIR)
	assert(len(files) == 2)

	files = find_catalog_files(INPUT_DIR, sector=1)
	assert(len(files) == 2)

	files = find_catalog_files(INPUT_DIR, camera=1)
	assert(len(files) == 1)

	files = find_catalog_files(INPUT_DIR, sector=1, camera=3, ccd=2)
	assert(len(files) == 1)

#----------------------------------------------------------------------
def test_load_ffi_files():

	files = find_ffi_files(INPUT_DIR, camera=1)

	img = load_ffi_fits(files[0])
	assert(img.shape == (2048, 2048))

	img, hdr = load_ffi_fits(files[0], return_header=True)
	assert(img.shape == (2048, 2048))

#----------------------------------------------------------------------
def test_sphere_distance():
	np.testing.assert_allclose(sphere_distance(0, 0, 90, 0), 90)
	np.testing.assert_allclose(sphere_distance(90, 0, 0, 0), 90)
	np.testing.assert_allclose(sphere_distance(0, -90, 0, 90), 180)
	np.testing.assert_allclose(sphere_distance(45, 45, 45, 45), 0)
	np.testing.assert_allclose(sphere_distance(33.2, 45, 33.2, -45), 90)
	np.testing.assert_allclose(sphere_distance(337.5, 0, 22.5, 0), 45)
	np.testing.assert_allclose(sphere_distance(22.5, 0, 337.5, 0), 45)
	np.testing.assert_allclose(sphere_distance(0, 0, np.array([0, 90]), np.array([90, 90])), np.array([90, 90]))

#----------------------------------------------------------------------
def test_coordtransforms():

	inp = np.array([[0, 0], [0, 90], [0, -90], [30, 0]], dtype='float64')

	expected_xyz = np.array([
		[1, 0, 0],
		[0, 0, 1],
		[0, 0, -1],
		[np.cos(30*np.pi/180), np.sin(30*np.pi/180), 0]
	], dtype='float64')

	xyz = radec_to_cartesian(inp)
	print( xyz )
	print( expected_xyz )

	print( xyz - expected_xyz )

	np.testing.assert_allclose(xyz, expected_xyz, atol=1e-7)

	# Transform back:
	radec2 = cartesian_to_radec(xyz)
	print( radec2 )

	# Test that we recoved the input:
	np.testing.assert_allclose(radec2, inp, atol=1e-7)

#----------------------------------------------------------------------
def test_rms_timescale():

	time = np.linspace(0, 27, 100)
	flux = np.zeros(len(time))

	rms = rms_timescale(time, flux)
	print(rms)
	np.testing.assert_allclose(rms, 0)

	rms = rms_timescale(time, flux*np.nan)
	print(rms)
	assert np.isnan(rms), "Should return nan on pure nan input"

	rms = rms_timescale([], [])
	print(rms)
	assert np.isnan(rms), "Should return nan on empty input"

	# Pure nan in the time-column should raise ValueError:
	with np.testing.assert_raises(ValueError):
		rms = rms_timescale(time*np.nan, flux)

	# Test with timescale longer than timespan should return zero:
	flux = np.random.randn(1000)
	time = np.linspace(0, 27, len(flux))
	rms = rms_timescale(time, flux, timescale=30.0)
	print(rms)
	np.testing.assert_allclose(rms, 0)

#----------------------------------------------------------------------
if __name__ == '__main__':
	test_move_median_central()
	test_find_ffi_files()
	test_find_tpf_files()
	test_find_hdf5_files()
	test_find_catalog_files()
	test_load_ffi_files()
	test_sphere_distance()
	test_coordtransforms()
	test_rms_timescale()
