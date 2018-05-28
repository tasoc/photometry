#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Jonas Svenstrup Hansen <jonas.svenstrup@gmail.com>
"""

from __future__ import division, print_function, with_statement, absolute_import
import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from photometry.psf import PSF

def test_psf():
	"""Test of PSF"""
	# Set stamp as square with length != width:
	row_min = 0.
	row_max = 10.
	col_min = 0.
	col_max = 20.
	stamp = (row_min, row_max, col_min, col_max)

	# PSF location:
	star_row = 4.
	star_col = 13.
	star_flux = 1.
	params = [[star_row, star_col, star_flux]]

	# Generate stamp image with PSF:
	psf = PSF(camera=1, ccd=1, stamp=stamp)
	img = psf.integrate_to_image(params=params, cutoff_radius=5)

	# Print some information:
	print(psf.shape)
	print(img.shape)
	print(img[int(star_row), int(star_col)])
	print(np.sum(img, axis=(0,1)))
	print(np.abs(img[int(star_row), int(star_col)] - 0.31211548134) < 1e-6)

#	from photometry.plots import plot_image
#	plot_image(img)

	# Check the value at the central pixel:
	assert(np.abs(img[int(star_row), int(star_col)] - 0.31211548134) < 1e-6)


if __name__ == '__main__':
	test_psf()
