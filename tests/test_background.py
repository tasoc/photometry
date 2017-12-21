#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, print_function, with_statement, absolute_import
import sys
import os
from astropy.io import fits
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from photometry.backgrounds import fit_background

def test_background():
	"""Test of background estimator"""

	# Load the first image in the input directory:
	INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input', 'images')
	files = glob.glob(os.path.join(INPUT_DIR, '*.fits.gz'))
	fname = sorted(files)[0]

	# Find the shape of the original image:
	hdr = fits.getheader(fname, ext=0)
	img_shape = (hdr['NAXIS1'], hdr['NAXIS2'])

	# Estimate the background:
	bck, mask = fit_background(fname)

	# Print some information:
	print(fname)
	print(bck.shape)
	print(mask.shape)

	# Check the sizes of the returned images:
	assert(bck.shape == img_shape)
	assert(mask.shape == img_shape)

if __name__ == '__main__':
	test_background()
