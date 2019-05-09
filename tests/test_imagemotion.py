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
from photometry.utilities import find_ffi_files, find_hdf5_files, load_ffi_fits
#from photometry.plots import plt
import h5py

def test_imagemotion():
	"""Test of ImageMovementKernel"""

	# Load the first image in the input directory:
	INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input', 'images')
	files = find_ffi_files(INPUT_DIR, camera=1, ccd=1)
	fname = files[0]

	# Load the image:
	img = load_ffi_fits(fname)

	# Create new image, moved down by one pixel:
	#img2 = np.roll(img, 1, axis=0)

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

def test_imagemotion_wcs():
	"""Test of ImageMovementKernel"""

	# Some positions across the image:
	xx, yy = np.meshgrid(
		np.linspace(0, 2047, 5, dtype='float64'),
		np.linspace(0, 2047, 5, dtype='float64'),
	)
	xy = np.column_stack((xx.flatten(), yy.flatten()))
	print("Positions to be tested:")
	print(xy)

	# Load the first image in the input directory:
	INPUT_FILE = find_hdf5_files(os.path.join(os.path.dirname(__file__), 'input'), sector=1, camera=1, ccd=1)[0]

	with h5py.File(INPUT_FILE, 'r') as h5:

		times = np.asarray(h5['time']) - np.asarray(h5['timecorr'])
		kernels = [h5['wcs'][dset][0] for dset in h5['wcs']]

		# Create ImageMovementKernel instance:
		imk = ImageMovementKernel(warpmode='wcs', wcs_ref=kernels[0])
		imk.load_series(times, kernels)

		# Ask for the jitter at the time of the reference image.
		# The movements should all be very close to zero,
		# since we used the same image as the reference:
		jitter = imk.interpolate(times[0], xy)
		assert(jitter.shape == xy.shape)
		np.testing.assert_allclose(jitter, 0, atol=1e-5, rtol=1e-5)

		# Ask for a jitter in-between timestamp to check the interpolation:
		jitter = imk.interpolate(0.5*(times[0] + times[1]), xy)
		assert(jitter.shape == xy.shape)

		# Check other critical timestamps:
		jitter = imk.interpolate(times[1], xy)
		assert(jitter.shape == xy.shape)

		# Last timestamp should also work:
		jitter = imk.interpolate(times[-1], xy)
		assert(jitter.shape == xy.shape)

		#plt.close('all')
		#plt.figure()
		#for x, y in xy:
		#	j = imk.jitter(times, x, y)
		#	plt.scatter(x+j[:,0], y+j[:,1], alpha=0.3)
		#plt.show()

		# Check that we are correctly throwing a ValueError when asking for a timestamp outside range:
		np.testing.assert_raises(ValueError, imk.interpolate, times[0] - 0.5, xy)
		np.testing.assert_raises(ValueError, imk.interpolate, times[-1] + 0.5, xy)

		# Overwrite the second WCS object to be identical to the first one:
		imk.series_kernels[1] = imk.series_kernels[0]
		jitter = imk.interpolate(0.5*(times[0] + times[1]), xy)
		assert(jitter.shape == xy.shape)
		np.testing.assert_allclose(jitter, 0, atol=1e-5, rtol=1e-5)

		# Add one to the reference pixels of the second:
		# Offset should now be one pixel over the entire FOV.
		# We have to remove the SIP headers for this test to work
		imk.wcs_ref.sip = None
		imk.series_kernels[1].sip = None
		imk.series_kernels[1].wcs.crpix += 1.0
		jitter = imk.interpolate(times[1], xy)
		assert(jitter.shape == xy.shape)
		np.testing.assert_allclose(jitter, 1)

	print("Done")

if __name__ == '__main__':
	test_imagemotion()
	test_imagemotion_wcs()
