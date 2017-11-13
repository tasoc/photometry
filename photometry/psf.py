#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, with_statement, absolute_import
from six.moves import range
import os
import numpy as np
from astropy.io import fits
from scipy.interpolate import RectBivariateSpline
import glob
import matplotlib.pyplot as plt
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize

class PSF(object):

	def __init__(self, camera, ccd, stamp):

		# Store information given in call:
		self.camera = 20
		self.ccd = 1
		self.stamp = stamp

		# Target pixel file shape in pixels:
		# TODO: check that the following are indeed x and y and not reversed:
		self.shape = (int(stamp[1] - stamp[0]), int(stamp[3] - stamp[2]))

		# The number of header units in the Kepler PSF files:
		n_hdu = 5

		# Get path to corresponding Kepler PSF file:
		PSFdir = os.path.join(os.path.dirname(__file__), 'data', 'psf')
		PSFglob = os.path.join(PSFdir, 'kplr{0:02d}.{1:d}*_prf.fits'.format(20, 1))
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

		# Normalise the PRF:
		prf /= (np.nansum(prf) * cdelt1p * cdelt2p)

		# Define pixel centered index arrays for the interpolater:
		PRFx = np.arange(0.5, xdim + 0.5)
		PRFy = np.arange(0.5, ydim + 0.5)
		# Center around 0 and convert to PSF subpixel resolution:
		PRFx = (PRFx - np.size(PRFx) / 2) * cdelt1p
		PRFy = (PRFy - np.size(PRFy) / 2) * cdelt2p

		# Interpolation function over the PRF:
		self.splineInterpolation = RectBivariateSpline(PRFx, PRFy, prf)


	def integrate_to_image(self, params, cutoff_radius=5):
		"""
		Integrate the underlying high-res PSF onto pixels.

		Parameters
		----------
		params : iterator, numpy.array
			List of stars to add to image. Should be an interator where each element is an numpy array with three elements: row, column and flux.
		cutoff_radius : float, optional
			Maximal radius away from center of star in pixels to integrate PSF model.

		Returns
		-------
		numpy.array : Image
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

		norm = ImageNormalize(stretch=LogStretch())

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
		ax.imshow(img, origin='lower', cmap='bone', norm=norm)
		ax.scatter(stars[:,1], stars[:,0], c='r', alpha=0.5)
		plt.show()
