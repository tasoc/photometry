#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, print_function, with_statement, absolute_import
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from photometry.backgrounds import fit_background
from photometry.utilities import find_ffi_files, load_ffi_fits

#------------------------------------------------------------------------------
def test_background():
	"""Test of background estimator"""

	# Load the first image in the input directory:
	INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input', 'images')
	fname = find_ffi_files(INPUT_DIR)[0]
	img, hdr = load_ffi_fits(fname, return_header=True)

	# Estimate the background:
	bck, mask = fit_background(fname)

	# Print some information:
	print(fname)
	print(bck.shape)
	print(mask.shape)

	# Check the sizes of the returned images:
	assert(bck.shape == img.shape)
	assert(mask.shape == img.shape)

#------------------------------------------------------------------------------
def test_background_with_radial():
	"""Test of background estimator including radial component"""

	# Load the first image in the input directory:
	INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input', 'images')
	fname = find_ffi_files(INPUT_DIR)[0]
	img, hdr = load_ffi_fits(fname, return_header=True)

	# Estimate the background:
	camera_centre = [324.566998914166, -33.172999301379]
	bck, mask = fit_background(fname, camera_centre=camera_centre)

	# Print some information:
	print(fname)
	print(bck.shape)
	print(mask.shape)

	# Check the sizes of the returned images:
	assert(bck.shape == img.shape)
	assert(mask.shape == img.shape)

#------------------------------------------------------------------------------
if __name__ == '__main__':
	test_background()
	test_background_with_radial()