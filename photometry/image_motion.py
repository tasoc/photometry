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

from __future__ import division, with_statement, print_function, absolute_import
from six.moves import range
import numpy as np
import cv2
#from skimage.transform import estimate_transform, warp, AffineTransform, EuclideanTransform
import math
from bottleneck import replace
from skimage.filters import scharr

class ImageMovementKernel(object):

	N_PARAMS = {
		'translation': 2,
		'euclidian': 3
	}

	#==============================================================================
	def __init__(self, warpmode='euclidian', image_ref=None):
		"""
		Initialize ImageMovementKernel.

		Parameters:
			warpmode (string): Options are ``'translation'`` and ``'euclidian'``. Default is ``'euclidian'``.
			image_ref (2D ndarray): Reference image used
		"""

		if warpmode not in ('translation', 'euclidian'):
			raise ValueError("Invalid warpmode")

		self.warpmode = warpmode
		self.image_ref = image_ref
		self.n_params = ImageMovementKernel.N_PARAMS[self.warpmode]

		if self.image_ref is not None:
			self.image_ref = self._prepare_flux(self.image_ref)

	#==============================================================================
	def __call__(self, *args, **kwargs):
		return self.apply_kernel(*args, **kwargs)

	#==============================================================================
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

		# Convert to logarithmic units
		flux = np.log10(flux)

		# Convert image to flux in range -1 to 1 (for gradient determination)
		fmax = np.nanmax(flux)
		fmin = np.nanmin(flux)
		ran = np.abs(fmax - fmin)
		flux1 = -1 + ((flux - fmin)/ran)*2

		# Calculate Scharr gradient
		flux1 = scharr(flux1)

		# Remove potential NaNs in gradient image
		replace(flux1, np.NaN, 0)

		# Make sure image is in proper units for ECC routine
		return np.asarray(flux1, dtype='float32')

	#==============================================================================
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
			R = np.matrix([[c, -s, dx], [s, c, dy]])

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

		return delta_pos

	#==============================================================================
	def calc_kernel(self, image, number_of_iterations=10000, termination_eps=1e-5):
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
