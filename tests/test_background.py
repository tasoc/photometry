#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import numpy as np
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
	assert bck.shape == img.shape
	assert mask.shape == img.shape
	assert np.all(np.isfinite(bck))
	assert mask.dtype == 'bool'

#--------------------------------------------------------------------------------------------------
def test_background_fakeimg(SHARED_INPUT_DIR):

	# Create fake image, which is constant:
	# FIXME: This doesn't test the radial background component,
	#        since the image is not seen as a TESS image
	fakeimg = np.full([2048, 2048], 1000, dtype='float32')

	# Fit fake image:
	bck, mask = fit_background(fakeimg)
	print(bck)

	# Check the sizes of the returned images:
	assert bck.shape == fakeimg.shape
	assert mask.shape == fakeimg.shape
	assert np.all(np.isfinite(bck))
	assert mask.dtype == 'bool'

	assert not np.any(mask), "Nothing should be masked out"
	np.testing.assert_allclose(bck, 1000)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
