#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:54:10 2017

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, print_function, with_statement, absolute_import
import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from photometry.utilities import move_median_central, find_ffi_files, load_ffi_fits

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')

def test_move_median_central():

	x_1d = np.array([4, 2, 2, 0, 0, np.NaN, 0, 2, 2, 4])
	result_1d = move_median_central(x_1d, 3)
	expected_1d = np.array([3, 2, 2, 0, 0, 0, 1, 2, 2, 3])
	np.testing.assert_allclose(result_1d, expected_1d)

	x_2d = np.tile(x_1d, (10, 1))
	result_2d = move_median_central(x_2d, 3, axis=1)
	expected_2d = np.tile(expected_1d, (10, 1))
	np.testing.assert_allclose(result_2d, expected_2d)


def test_find_ffi_files():
	
	files = find_ffi_files(INPUT_DIR)
	assert(len(files) == 4)

	files = find_ffi_files(INPUT_DIR, camera=1)
	assert(len(files) == 2)
	
	files = find_ffi_files(INPUT_DIR, camera=2)
	assert(len(files) == 2)


def test_load_ffi_files():
	
	files = find_ffi_files(INPUT_DIR, camera=1)
	
	img = load_ffi_fits(files[0])
	assert(img.shape == (2048, 2048))
	
	img, hdr = load_ffi_fits(files[0], return_header=True)
	assert(img.shape == (2048, 2048))


if __name__ == '__main__':
	test_move_median_central()
	test_find_ffi_files()
	test_load_ffi_files()
