#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, print_function, with_statement, absolute_import
import sys
import os.path
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from photometry.image_motion import ImageMovementKernel
from photometry.utilities import find_ffi_files, load_ffi_fits

def test_imagemotion():
	"""Test of ImageMovementKernel"""

	# Load the first image in the input directory:
	INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input', 'images')
	files = find_ffi_files(INPUT_DIR, camera=1, ccd=1)
	fname = files[0]

	# Load the image:
	img = load_ffi_fits(fname)

	# Create new image, moved down by one pixel:
	img2 = np.roll(img, 1, axis=0)

	# Some positions across the image:
	xx, yy = np.meshgrid(
		np.linspace(0, img.shape[0], 5, dtype='float64'),
		np.linspace(0, img.shape[1], 5, dtype='float64'),
	)
	xy = np.column_stack((xx.flatten(), yy.flatten()))
	print("Positions to be tested:")
	print(xy)

	desired1 = np.zeros_like(xy)
	desired2 = np.zeros_like(xy)
	desired2[:, 1] = 1

	for warpmode in ('unchanged', 'translation', 'euclidian'):
		print("Testing warpmode=" + warpmode)

		# Create ImageMovementKernel instance:
		imk = ImageMovementKernel(image_ref=img, warpmode=warpmode)

		# Calculate kernel for the same image:
		kernel = imk.calc_kernel(img)
		print("Kernel:")
		print(kernel)
		assert(len(kernel) == imk.n_params)

		# Calculate the new positions based on the kernel:
		delta_pos = imk(xy, kernel)
		print("Extracted movements:")
		print(delta_pos)

		assert(delta_pos.shape == xy.shape)

		# The movements should all be very close to zero,
		# since we used the same image as the reference:
		np.testing.assert_allclose(delta_pos, desired1, atol=1e-5, rtol=1e-5)

		"""
		kernel = imk.calc_kernel(img2)
		print("Kernel 2:")
		print(kernel)

		# Calculate the new positions based on the kernel:
		delta_pos = imk(xy, kernel)
		print("Extracted movements:")
		print(delta_pos)

		assert(delta_pos.shape == xy.shape)

		# The movements should all be very close to zero,
		# since we used the same image as the reference:
		# FIXME: Would LOVE this to be more accurate!
		np.testing.assert_allclose(delta_pos, desired2, atol=1e-3, rtol=1e-2)
		"""

	print("Done")

if __name__ == '__main__':
	test_imagemotion()
