#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Point Spread Function (PSF).

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, print_function, with_statement, absolute_import
from six.moves import range
import os
import numpy as np
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline
import glob
from .plots import plt, plot_image

class PSF(object):
	"""
	Point Spread Function (PSF).

	Attributes:
		camera (integer): TESS camera (1-4).
		ccd (integer): TESS CCD (1-4).
		stamp (tuple): The pixel sub-stamp used to generate PSF.
		shape (tuple): Shape of pixel sub-stamp.
		PSFfile (string): Path to PSF file that was interpolated in.
		ref_column (float): Reference CCD column that PSF is calculated for.
		ref_row (float): Reference CCD row that PSF is calculated for.
		splineInterpolation (`scipy.interpolate.RectBivariateSpline` object): Interpolation to evaluate PSF on arbitrery position relative to center of PSF.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	def __init__(self, camera, ccd, stamp):
		"""
		Point Spread Function (PSF).

		Parameters:
			camera (integer): TESS camera number (1-4).
			ccd (integer): TESS CCD number (1-4).
			stamp (4-tuple): Sub-stamp on CCD to load PSF for.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		# Store information given in call:
		self.camera = camera
		self.ccd = ccd
		self.stamp = stamp

		# Target pixel file shape in pixels:
		self.shape = (int(stamp[1] - stamp[0]), int(stamp[3] - stamp[2]))

		# The number of header units in the Kepler PSF files:
		n_hdu = 5

		# Get path to corresponding Kepler PSF file:
		PSFdir = os.path.join(os.path.dirname(__file__), 'data', 'psf')
		PSFglob = os.path.join(PSFdir, 'kplr{0:02d}.{1:d}*_prf.fits'.format(11, 2))
		self.PSFfile = glob.glob(PSFglob)[0]

		# Set minimum PRF weight to avoid dividing by almost 0 somewhere:
		minimum_prf_weight = 1e-6

		# Interpolate the calibrated PRF shape to middle of the stamp:
		self.ref_column = 0.5*(stamp[3] + stamp[2])
		self.ref_row = 0.5*(stamp[1] + stamp[0])

		# Read the Kepler PRF images:
		with fits.open(self.PSFfile, mode='readonly', memmap=True) as hdu:
			# Find size of PSF images and
			# the pixel-scale of the PSF images:
			xdim = hdu[1].header['NAXIS1']
			ydim = hdu[1].header['NAXIS2']
			cdelt1p = hdu[1].header['CDELT1P']
			cdelt2p = hdu[1].header['CDELT2P']

			# Preallocate prf array:
			prf = np.zeros((xdim, ydim), dtype='float64')

			for i in range(1, n_hdu+1):
				prfn = hdu[i].data
				crval1p = hdu[1].header['CRPIX1P']
				crval2p = hdu[1].header['CRPIX2P']

				# Weight with the distance between each PRF sample and the target:
				prfWeight = np.sqrt((self.ref_column - crval1p)**2 + (self.ref_row - crval2p)**2)

				# Catch too small weights
				if prfWeight < minimum_prf_weight:
					prfWeight = minimum_prf_weight

				# Add the weighted values to the PRF array:
				prf += prfn / prfWeight

		# Normalize the PRF:
		prf /= (np.nansum(prf) * cdelt1p * cdelt2p)

		# Define pixel centered index arrays for the interpolator:
		PRFx = np.arange(0.5, xdim + 0.5)
		PRFy = np.arange(0.5, ydim + 0.5)
		# Center around 0 and convert to PSF subpixel resolution:
		PRFx = (PRFx - np.size(PRFx) / 2) * cdelt1p
		PRFy = (PRFy - np.size(PRFy) / 2) * cdelt2p

		# Interpolation function over the PRF:
		self.splineInterpolation = RectBivariateSpline(PRFx, PRFy, prf) #: 2D-interpolation of PSF (RectBivariateSpline).


	def integrate_to_image(self, params, cutoff_radius=5):
		"""
		Integrate the underlying high-res PSF onto pixels.

		Parameters:
			params (iterator, numpy.array): List of stars to add to image. Should be an iterator where each element is an numpy array with three elements: row, column and flux.
			cutoff_radius (float, optional): Maximal radius away from center of star in pixels to integrate PSF model.

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
						img[j,i] += star_flux * self.splineInterpolation.integral(column_cen-0.5, column_cen+0.5, row_cen-0.5, row_cen+0.5)

		return img


	def plot(self):
		"""
		Create a plot of the shape of the PSF.
		"""

		stars = np.array([
			[self.ref_row-self.stamp[0], self.ref_column-self.stamp[2], 1],
		])

		y = np.linspace(-5, 5, 500)
		x = np.linspace(-5, 5, 500)
		xx, yy = np.meshgrid(y, x)

		spline = self.splineInterpolation(x, y, grid=True)
		spline += np.abs(spline.min()) + 1e-14
		spline = np.sqrt(spline)

		img = self.integrate_to_image(stars)

		fig = plt.figure()
		ax = fig.add_subplot(121)
		ax.contourf(yy, xx, spline, 20, cmap='bone')
		ax.axis('equal')
		ax.set_xlim(-5, 5)
		ax.set_ylim(-5, 5)
		ax = fig.add_subplot(122)
		plot_image(img)
		ax.scatter(stars[:,1], stars[:,0], c='r', alpha=0.5)
		plt.show()
