#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of photometry.pixel_flags.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os
import numpy as np
import conftest # noqa: F401
from photometry import io
import photometry.pixel_flags as pxf

#--------------------------------------------------------------------------------------------------
def test_pixel_manual_exclude_mars(SHARED_INPUT_DIR):

	fpath = io.find_ffi_files(os.path.join(SHARED_INPUT_DIR, 'images'), sector=1)[0]
	img = io.FFIImage(fpath)
	img.is_tess = True
	img.header['CAMERA'] = 1
	img.header['CCD'] = 4
	img.header['FFIINDEX'] = 4724
	print(img)

	mask = pxf.pixel_manual_exclude(img)
	print(mask)

	assert isinstance(mask, np.ndarray)
	assert mask.shape == img.shape
	assert mask.dtype == 'bool'
	assert np.all(mask[:, 1536:])
	assert not np.any(mask[:, :1536])

#--------------------------------------------------------------------------------------------------
def test_pixel_manual_exclude_zero(SHARED_INPUT_DIR):

	fpath = io.find_ffi_files(os.path.join(SHARED_INPUT_DIR, 'images'))[0]
	img = io.FFIImage(fpath)
	img.is_tess = True
	img.data[:,:] = 0
	img.mask[:,:] = False
	print(img)

	mask = pxf.pixel_manual_exclude(img)
	print(mask)

	assert isinstance(mask, np.ndarray)
	assert mask.shape == img.shape
	assert mask.dtype == 'bool'
	assert np.all(mask)

#--------------------------------------------------------------------------------------------------
def test_pixel_manual_exclude_earth(SHARED_INPUT_DIR):

	fpath = io.find_ffi_files(os.path.join(SHARED_INPUT_DIR, 'images'))[0]
	img = io.FFIImage(fpath)
	img.is_tess = True
	img.header['CAMERA'] = 1
	img.header['FFIINDEX'] = 11354
	print(img)

	mask = pxf.pixel_manual_exclude(img)
	print(mask)

	assert isinstance(mask, np.ndarray)
	assert mask.shape == img.shape
	assert mask.dtype == 'bool'
	assert np.all(mask)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
