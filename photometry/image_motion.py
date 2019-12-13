#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calculate image shifts between images.

Example
-------
To calculate the image shifts between the reference image (``ref_image``) and another image (``image``):

	>>> imk = ImageMotionKernel(ref_image=ref_image, warpmode='translation')
	>>> kernel = img.calc_kernel(image)
	>>> print(kernel)


.. codeauthor:: Mikkel N. Lund
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import cv2
#from skimage.transform import estimate_transform, warp, AffineTransform, EuclideanTransform
import math
from bottleneck import replace
from skimage.filters import scharr
from scipy.interpolate import interp1d
from astropy.io import fits
from astropy.wcs import WCS, NoConvergence
import warnings

class ImageMovementKernel(object):

	N_PARAMS = {
		'unchanged': 0,
		'translation': 2,
		'euclidian': 3,
		'wcs': 1
	}

	#----------------------------------------------------------------------------------------------
	def __init__(self, warpmode='euclidian', image_ref=None, wcs_ref=None):
		"""
		Initialize ImageMovementKernel.

		Parameters:
			warpmode (string): Options are ``'unchanged'``, ``'translation'``, ``'euclidian'`` and ``'wcs'``. Default is ``'euclidian'``.
			image_ref (2D ndarray): Reference image used.
			wcs_ref (``astropy.wcs.WCS`` object): Reference WCS when using `warpmode`='wcs'.
		"""

		if warpmode not in ('unchanged', 'translation', 'euclidian', 'wcs'):
			raise ValueError("Invalid warpmode")

		self.warpmode = warpmode
		self.image_ref = image_ref
		self.wcs_ref = wcs_ref
		self.n_params = ImageMovementKernel.N_PARAMS[self.warpmode]

		if self.image_ref is not None:
			self.image_ref = self._prepare_flux(self.image_ref)

		if self.wcs_ref is not None and not isinstance(self.wcs_ref, WCS):
			if not isinstance(self.wcs_ref, str): self.wcs_ref = self.wcs_ref.decode("utf-8") # For Python 3
			self.wcs_ref = WCS(header=fits.Header().fromstring(self.wcs_ref))

		self._interpolator = None

	#----------------------------------------------------------------------------------------------
	def __call__(self, *args, **kwargs):
		return self.apply_kernel(*args, **kwargs)

	#----------------------------------------------------------------------------------------------
	def _prepare_flux(self, flux):
		"""
		Preparation of images for Enhanced Correlation Coefficient (ECC) Maximization
		estimation of movement - used for estimation of jitter.

		Parameters:
			flux (array): flux pixel image

		Returns:
			array: Gradient (using Scharr method) of image in logarithmic units.

		.. codeauthor:: Mikkel N. Lund
		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		# Convert to logarithmic units, avoiding taking log if zero:
		flux = np.asarray(flux)
		flux = np.log10(flux - np.nanmin(flux) + 1.0)

		# Convert image to flux in range -1 to 1 (for gradient determination)
		fmax = np.nanmax(flux)
		fmin = np.nanmin(flux)
		ran = np.abs(fmax - fmin)
		flux1 = -1 + 2*((flux - fmin)/ran)

		# Calculate Scharr gradient
		flux1 = scharr(flux1)

		# Remove potential NaNs in gradient image
		replace(flux1, np.NaN, 0)

		# Make sure image is in proper units for ECC routine
		return np.asarray(flux1, dtype='float32')

	#----------------------------------------------------------------------------------------------
	def apply_kernel(self, xy, kernel):
		"""
		Application of warp matrix to pixel coordinates

		Parameters:
			xy (2D ndarray): 2D array of image positions to be transformed.
			kernel (1D ndarray): The kernel to transform against.

		Returns:
			ndarray: Change in positions compared to reference.

		.. codeauthor:: Mikkel N. Lund
		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		xy = np.atleast_2d(xy)
		delta_pos = np.empty_like(xy)

		if self.warpmode == 'euclidian':
			dx = kernel[0]
			dy = kernel[1]
			theta = kernel[2]

			# Set up warp matrix:
			c = np.cos(theta)
			s = np.sin(theta)
			R = np.array([[c, -s, dx], [s, c, dy]])

			# Apply warp to all positions:
			for i in range(xy.shape[0]):
				x = xy[i, 0]
				y = xy[i, 1]
				delta_pos[i, :] = np.dot(R, [x, y, 1])

			# Subtract the reference positions to return the change in positions:
			delta_pos -= xy

		elif self.warpmode == 'translation':
			delta_pos[:, 0] = kernel[0]
			delta_pos[:, 1] = kernel[1]

		elif self.warpmode == 'unchanged':
			delta_pos.fill(0)

		elif self.warpmode == 'wcs':
			# Calculate RA and DEC of target in the reference image:
			radec = self.wcs_ref.all_pix2world(xy, 0, ra_dec_order=True)
			# Use RA and DEC to find the position in the kernel image:
			# TODO: Better handling of NoConvergence exception, which is currently silenced
			delta_pos = kernel.all_world2pix(radec, 0, ra_dec_order=True, maxiter=50, quiet=True)
			# Calculate the difference in pixel-position:
			delta_pos -= xy

		return delta_pos

	#----------------------------------------------------------------------------------------------
	def calc_kernel(self, image, number_of_iterations=10000, termination_eps=1e-6):
		"""
		Calculates the position movement kernel for a given image. This kernel is
		a set of numbers that can be passed to `apply_kernel` to calculate the movement
		of a star at specific coordinates.

		Calculation of Enhanced Correlation Coefficient (ECC) Maximization using OpenCV.

		Parameters:
			image (ndarray): Image to calculate kernel for.
			number_of_iterations (integer, optional): Specify the number of iterations.
			termination_eps (float, optional): Specify the threshold of the increment in the correlation coefficient between two iterations.

		Returns:
			ndarray:

		.. codeauthor:: Mikkel N. Lund
		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		if self.warpmode == 'unchanged':
			return []

		# Check that reference image was actually given:
		if self.image_ref is None:
			raise Exception("Reference image not defined")

		# Define the motion model
		if self.warpmode == 'euclidian':
			warp_mode = cv2.MOTION_EUCLIDEAN
		elif self.warpmode == 'translation':
			warp_mode = cv2.MOTION_TRANSLATION

		# Prepare comparison image for estimation of motion
		image = self._prepare_flux(image)

		# Define 2x3 warp matrix and initialize the matrix to identity
		warp_matrix = np.eye(2, 3, dtype='float32')

		# Define termination criteria
		criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

		# Run the ECC algorithm. The results are stored in warp_matrix.
		try:
			cc, warp_matrix = cv2.findTransformECC(self.image_ref, image, warp_matrix, warp_mode, criteria)
		except:
			return np.NaN*np.ones(self.n_params)

		# Extract movement in pixel units in x- and y direction
		dx = warp_matrix[0,2]
		dy = warp_matrix[1,2]

		if self.warpmode == 'euclidian':
			# Estimate rotation angle in radians
			theta = math.atan2(warp_matrix[1,0], warp_matrix[0,0])
			return [dx, dy, theta]
		else:
			# Translation only:
			return [dx, dy]

	#----------------------------------------------------------------------------------------------
	def load_series(self, times, kernels):
		"""
		Load time-series of kernels and create interpolator.

		The interpolator (:py:func:`interpolator`) can be used to obtain movements
		at a arbitrery timestamp within the timestamps provided in ``times``.

		Parameters:
			time (1D array): Timestamps to be interpolated against. Timestamps must be sorted.
			kernels (2D array): List of kernels.

		Raises:
			ValueError: If kernels have the wrong shape.
		"""

		self.series_times = np.asarray(times)
		self.series_kernels = kernels

		if self.warpmode == 'wcs':
			# Check the lenghts of the provided vectors:
			if len(kernels) != len(times):
				raise ValueError("Wrong shape of kernels.")

			good_series = np.ones_like(self.series_times, dtype='bool')
			for k in range(len(kernels)):
				if not isinstance(self.series_kernels[k], WCS):
					# Assuming that is is a string then:
					hdr_string = self.series_kernels[k]
					if not isinstance(hdr_string, str):
						hdr_string = hdr_string.decode("utf-8") # For Python 3

					# If the string is empty, remove the point from the series:
					if hdr_string.strip() == '':
						good_series[k] = False
						continue

					# Create a WCS object from the header string:
					self.series_kernels[k] = WCS(header=fits.Header().fromstring(hdr_string), relax=True)

				# Try if the WCS can return pixel coordinates for the test-coordinates:
				# If it can't we will remove that timestamp from the series, and the
				# pixel coordinates will therefore be interpolated when calculation jitter.
				# Using a (ra, dec) coordinates from the actual footprint of the WCS, and
				# using axes=(2,2), since we here don't nessacerily know the size of the image,
				# and we are only using the first corner anyway.
				fp = self.series_kernels[k].calc_footprint(axes=(2, 2))
				test_coords = np.atleast_2d(fp[0, :])
				try:
					self.series_kernels[k].all_world2pix(test_coords, 0, ra_dec_order=True, maxiter=50)
				except (NoConvergence, ValueError):
					good_series[k] = False

			# Remove any bad series points from the lists:
			self.series_kernels = np.asarray(self.series_kernels)
			self.series_times = self.series_times[good_series]
			self.series_kernels = self.series_kernels[good_series]

		else:
			# For these warpmodes, the kernels should be 2D arrays:
			self.series_kernels = np.atleast_2d(self.series_kernels)

			# Check shape of the input:
			if self.series_kernels.shape != (len(self.series_times), self.n_params):
				raise ValueError("Wrong shape of kernels. Anticipated ({0},{1}), but got {2}".format(
					len(self.series_times),
					self.n_params,
					self.series_kernels.shape
				))

			# Only take the kernels that are well-defined:
			# TODO: Should we raise a warning if there are many undefined?
			indx = np.isfinite(times) & np.all(np.isfinite(kernels), axis=1)

			# Create interpolator:
			self._interpolator = interp1d(times[indx], kernels[indx, :],
				axis=0,
				assume_sorted=True,
				bounds_error=False,
				fill_value=(kernels[0, :], kernels[-1, :]))

	#----------------------------------------------------------------------------------------------
	def interpolate(self, time, xy):
		"""
		Interpolate in the kernel time-series provided in :py:func:`load_series`
		to obtain movment a arbitrery time.

		Parameters:
			time (float): Timestamp to return movement for.
			xy (2D array): row and column positions to be modified.

		Returns:
			``numpy.ndarray``: Array with the same size as `xy` containing the
				changes to rows and columns. These can be added
				to `xy` to yield the new positions.

		Raises:
			ValueError: If timeseries has not been provided.
		"""

		if self.warpmode == 'wcs':
			# Methods where the kernel is complex (non-numeric)
			# Handle the case where we are requesting a timestamp outside the
			# range of the loaded kernel timeseries:
			if time < self.series_times[0] or time > self.series_times[-1]:
				# Allow for a bit of a margin before and after the ends of the
				# timeseries, to account for e.g. round-off errors in the timestamps:
				dt = np.median(np.diff(self.series_times))
				if np.abs(time - self.series_times[0]) < dt:
					return self.apply_kernel(xy, self.series_kernels[0])
				elif np.abs(time - self.series_times[-1]) < dt:
					return self.apply_kernel(xy, self.series_kernels[-1])
				else:
					raise ValueError("Timestamp outside timeseries interval")

			# Find the point in the series where the timestamp falls:
			k = np.searchsorted(self.series_times, time, side='right')
			t1 = self.series_times[k-1]
			# Find the jitter in that kernel:
			jitter_1 = self.apply_kernel(xy, self.series_kernels[k-1])
			if t1 == time:
				# We actually hit spot on, so let's just return the jitter:
				return jitter_1
			else:
				#
				t2 = self.series_times[k]
				jitter_2 = self.apply_kernel(xy, self.series_kernels[k])

				int_time = [t1, t2]
				jitter_row = interp1d(int_time, np.column_stack((jitter_1[:,0], jitter_2[:,0])), axis=1, assume_sorted=True)
				jitter_col = interp1d(int_time, np.column_stack((jitter_1[:,1], jitter_2[:,1])), axis=1, assume_sorted=True)

				return np.column_stack((jitter_row(time), jitter_col(time)))
		else:
			#
			if self._interpolator is None:
				raise ValueError("Interpolator is not defined. ")

			# Get the kernel parameters for the timestamp:
			with warnings.catch_warnings():
				warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy')
				kernel = self._interpolator(time)

			return self.apply_kernel(xy, kernel)

	#----------------------------------------------------------------------------------------------
	def jitter(self, time, column, row):
		"""
		Calculate the change to a given position as a function of time.

		Parameters:
			time (ndarray): Array of timestamps to calculate position changes for.
			column (float): Column position at reference time.
			row (float): Row position at reference time.

		Returns:
			ndarray: 2D array with changes in column and row for each timestamp.
		"""

		xy = np.array([column, row])
		jtr = np.empty((len(time), 2), dtype='float64')
		for k in range(len(time)):
			jtr[k, :] = self.interpolate(time[k], xy)

		return jtr
