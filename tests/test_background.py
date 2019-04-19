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
	bck, mask = fit_background(fname, )

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
