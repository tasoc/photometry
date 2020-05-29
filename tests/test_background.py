#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import conftest # noqa: F401
from photometry.backgrounds import fit_background
from photometry.utilities import find_ffi_files, load_ffi_fits

#--------------------------------------------------------------------------------------------------
def test_background(SHARED_INPUT_DIR):
	"""Test of background estimator"""

	# Load the first image in the input directory:
	fname = find_ffi_files(SHARED_INPUT_DIR)[0]
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

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
