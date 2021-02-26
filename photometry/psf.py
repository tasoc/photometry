#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Point Spread Function (PSF).

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import os
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import RectBivariateSpline
import glob
from .plots import plt, plot_image

class PSF(object):
	"""
	Point Spread Function (PSF).

	Attributes:
		camera (int): TESS camera (1-4).
		ccd (int): TESS CCD (1-4).
		stamp (tuple): The pixel sub-stamp used to generate PSF.
		shape (tuple): Shape of pixel sub-stamp.
		PSFfile (string): Path to PSF file that was interpolated in.
		ref_column (float): Reference CCD column that PSF is calculated for.
		ref_row (float): Reference CCD row that PSF is calculated for.
		splineInterpolation (:py:class:`scipy.interpolate.RectBivariateSpline`): 2D Interpolation
			to evaluate PSF on arbitrary position relative to center of PSF.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	#----------------------------------------------------------------------------------------------
	def __init__(self, sector, camera, ccd, stamp):
		"""
		Point Spread Function (PSF).

		Parameters:
			sector (int): TESS Observation sector.
			camera (int): TESS camera number (1-4).
			ccd (int): TESS CCD number (1-4).
			stamp (4-tuple): Sub-stamp on CCD to load PSF for.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		# Simple input checks:
		if sector < 1:
			raise ValueError("Sector number must be greater than zero")
		if camera not in (1, 2, 3, 4):
			raise ValueError("Camera must be 1, 2, 3 or 4.")
		if ccd not in (1, 2, 3, 4):
			raise ValueError("CCD must be 1, 2, 3 or 4.")
		if len(stamp) != 4:
			raise ValueError("Incorrect stamp provided.")

		# Store information given in call:
		self.sector = sector
		self.camera = camera
		self.ccd = ccd
		self.stamp = stamp

		# Target pixel file shape in pixels:
		self.shape = (int(stamp[1] - stamp[0]), int(stamp[3] - stamp[2]))

		# Get path to corresponding TESS PRF file:
		PSFdir = os.path.join(os.path.dirname(__file__), 'data', 'psf')
		SectorDir = 'start_s0004' if sector >= 4 else 'start_s0001'
		PSFglob = os.path.join(PSFdir, SectorDir, f'tess*-{camera:d}-{ccd:d}-characterized-prf.mat')
		self.PSFfile = glob.glob(PSFglob)[0]

		# Set minimum PRF weight to avoid dividing by almost 0 somewhere:
		minimum_prf_weight = 1e-6

		# Interpolate the calibrated PRF shape to middle of the stamp:
		self.ref_column = 0.5*(stamp[3] + stamp[2])
		self.ref_row = 0.5*(stamp[1] + stamp[0])

		# Read the TESS PRF file, which is a MATLAB data file:
		mat = loadmat(self.PSFfile)
		mat = mat['prfStruct']

		# Center around 0 and convert to PSF subpixel resolution:
		# We are just using the first one here, assuming they are all the same
		PRFx = np.asarray(mat['prfColumn'][0][0], dtype='float64').flatten()
		PRFy = np.asarray(mat['prfRow'][0][0], dtype='float64').flatten()

		# Find size of PSF images and
		# the pixel-scale of the PSF images:
		n_hdu = len(mat['values'][0])
		xdim = len(PRFx)
		ydim = len(PRFy)
		cdelt1p = np.median(np.diff(PRFx))
		cdelt2p = np.median(np.diff(PRFy))

		# Preallocate prf array:
		prf = np.zeros((xdim, ydim), dtype='float64')

		# Loop through the PRFs measured at different positions:
		for i in range(n_hdu):
			prfn = mat['values'][0][i]
			crval1p = float(mat['ccdColumn'][0][i])
			crval2p = float(mat['ccdRow'][0][i])

			# Weight with the distance between each PRF sample and the target:
			prfWeight = np.sqrt((self.ref_column - crval1p)**2 + (self.ref_row - crval2p)**2)

			# Catch too small weights
			prfWeight = max(prfWeight, minimum_prf_weight)

			# Add the weighted values to the PRF array:
			prf += prfn / prfWeight

		# Normalize the PRF:
		prf /= (np.nansum(prf) * cdelt1p * cdelt2p)

		# Interpolation function over the PRF:
		self.splineInterpolation = RectBivariateSpline(PRFx, PRFy, prf)

	#----------------------------------------------------------------------------------------------
	def integrate_to_image(self, params, cutoff_radius=5):
		"""
		Integrate the underlying high-res PSF onto pixels.

		Parameters:
			params (iterator, numpy.array): List of stars to add to image. Should be an iterator
				where each element is an numpy array with three elements: row, column and flux.
			cutoff_radius (float, optional): Maximal radius away from center of star in pixels
				to integrate PSF model.

		Returns:
			numpy.array: Image
		"""

		img = np.zeros(self.shape, dtype='float64')
		for i in range(self.shape[0]):
			for j in range(self.shape[1]):
				for star in params:
					star_row = star[0]
					star_column = star[1]
					if np.sqrt((j-star_column)**2 + (i-star_row)**2) < cutoff_radius:
						star_flux = star[2]
						column_cen = j - star_column
						row_cen = i - star_row
						img[i,j] += star_flux * self.splineInterpolation.integral(column_cen-0.5, column_cen+0.5, row_cen-0.5, row_cen+0.5)

		return img

	#----------------------------------------------------------------------------------------------
	def plot(self):
		"""
		Create a plot of the shape of the PSF.
		"""

		stars = np.array([
			[self.ref_row-self.stamp[0], self.ref_column-self.stamp[2], 1],
		])

		y = np.linspace(-6, 6, 500)
		x = np.linspace(-6, 6, 500)
		xx, yy = np.meshgrid(y, x)

		spline = self.splineInterpolation(x, y, grid=True)
		spline += np.abs(spline.min()) + 1e-14
		spline = np.log10(spline)

		img = self.integrate_to_image(stars)

		fig = plt.figure()
		ax = fig.add_subplot(121)
		ax.contourf(yy, xx, spline, 20, cmap='bone_r')
		ax.set_xlim(-6, 6)
		ax.set_ylim(-6, 6)
		ax.axis('equal')

		ax = fig.add_subplot(122)
		plot_image(img, ax=ax)
		ax.scatter(stars[:,1], stars[:,0], c='r', alpha=0.5)

		plt.tight_layout()
		return fig
