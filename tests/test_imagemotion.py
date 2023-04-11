#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
import numpy as np
import h5py
import conftest # noqa: F401
from photometry.image_motion import ImageMovementKernel
from photometry import io
#from photometry.plots import plt

#--------------------------------------------------------------------------------------------------
def test_imagemotion_invalid_warpmode():
	"""Test ImageMovementKernel with invalid warpmode."""
	with pytest.raises(ValueError):
		ImageMovementKernel(warpmode='invalid')

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('warpmode', ['unchanged', 'translation', 'euclidian', 'affine'])
def test_imagemotion(SHARED_INPUT_DIR, warpmode):
	"""Test of ImageMovementKernel"""

	print("Testing warpmode=" + warpmode)

	# Load the first image in the input directory:
	files = io.find_ffi_files(os.path.join(SHARED_INPUT_DIR, 'images'), camera=1, ccd=1)
	fname = files[0]

	# Load the image:1
	img = io.FFIImage(fname)

	# Trying to calculate kernel with no reference image should give error:
	if warpmode == 'unchanged':
		imk = ImageMovementKernel(image_ref=None, warpmode=warpmode).calc_kernel(img)
		assert len(imk) == 0, "Kernel should be an empty array for 'unchanged'"
	else:
		# Trying to calculate kernel with no reference image should give error:
		with pytest.raises(RuntimeError) as e:
			ImageMovementKernel(image_ref=None, warpmode=warpmode).calc_kernel(img)
		assert str(e.value) == "Reference image not defined"

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

	# Create ImageMovementKernel instance:
	imk = ImageMovementKernel(image_ref=img, warpmode=warpmode)

	# If calling interpolate before defining kernels, we should get an error:
	with pytest.raises(ValueError):
		imk.interpolate(1234, 1234)

	# Calculate kernel for the same image:
	kernel = imk.calc_kernel(img)
	print("Kernel:")
	print(kernel)
	assert len(kernel) == imk.n_params

	# Calculate the new positions based on the kernel:
	delta_pos = imk(xy, kernel)
	print("Extracted movements:")
	print(delta_pos)

	assert delta_pos.shape == xy.shape

	# The movements should all be very close to zero,
	# since we used the same image as the reference:
	np.testing.assert_allclose(delta_pos, desired1, atol=1e-5, rtol=1e-5)

	#
	with pytest.raises(ValueError) as excinfo:
		imk.load_series([1234, 1235], kernel)

	print(excinfo.value)
	assert str(excinfo.value).startswith('Wrong shape of kernels.')

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

#--------------------------------------------------------------------------------------------------
def test_imagemotion_wcs(SHARED_INPUT_DIR):
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
	INPUT_FILE = io.find_hdf5_files(SHARED_INPUT_DIR, sector=1, camera=1, ccd=1)[0]

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

		# Just before first timestamp should also work (roundoff check):
		jitter = imk.interpolate(times[0] - np.finfo('float64').eps, xy)
		assert(jitter.shape == xy.shape)

		# Just after last timestamp should also work (roundoff check):
		jitter = imk.interpolate(times[-1] + np.finfo('float64').eps, xy)
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

		# Removing some timestamps should so we can test if we correctky throw an exceotion;
		times = times[1:]
		with pytest.raises(ValueError):
			imk.load_series(times, kernels)

	print("Done")

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
