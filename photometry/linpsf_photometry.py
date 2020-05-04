#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Linear PSF Photometry.

Do point spread function photometry with fixed centroids. The flux of
all stars in the image are fitted simultaneously using a linear least
squares method.

.. codeauthor:: Jonas Svenstrup Hansen <jonas.svenstrup@gmail.com>
"""

import numpy as np
#import scipy
import logging
import os
from bottleneck import allnan
from . import BasePhotometry, STATUS
from .psf import PSF
#from .utilities import mag2flux
from .plots import plt, plot_image_fit_residuals, save_figure

#--------------------------------------------------------------------------------------------------
class LinPSFPhotometry(BasePhotometry):

	def __init__(self, *args, **kwargs):
		"""
		Linear PSF photometry.

		Do point spread function photometry with fixed centroids. The flux of
		all stars in the image are fitted simultaneously using a linear least
		squares method.

		Note:
			Inspired by the :py:class:`psf_photometry` class set up by
			Rasmus Handberg <rasmush@phys.au.dk>. The code in this
			:py:func:`__init__` function as well as the logging, catalog call,
			time domain loop structure, catalog star limits and lightcurve
			output is copied from that class.

		.. codeauthor:: Jonas Svenstrup Hansen <jonas.svenstrup@gmail.com>
		"""
		# Call the parent initializing:
		# This will set several default settings
		super().__init__(*args, **kwargs)

		# Create instance of the PSF for the given pixel stamp:
		# NOTE: If we run resize_stamp at any point in the code,
		#       we should also update self.PSF.
		# TODO: Maybe we should move this into BasePhotometry?
		self.psf = PSF(self.sector, self.camera, self.ccd, self.stamp)

	#----------------------------------------------------------------------------------------------
	def do_photometry(self):
		"""Linear PSF Photometry
		TODO: add description of method and what A and b are
		"""

		logger = logging.getLogger(__name__)

		# Load catalog to determine what stars to fit:
		cat = self.catalog
		staridx = np.squeeze(np.where(cat['starid'] == self.starid))

		# Log full catalog for current stamp:
		logger.debug(cat)

		# Calculate distance from main target:
		cat['dist'] = np.sqrt((cat['row_stamp'][staridx] - cat['row_stamp'])**2
						+ (cat['column_stamp'][staridx] - cat['column_stamp'])**2)

		# Find indices of stars in catalog to fit:
		# (only include stars that are close to the main target and that are
		# not much fainter)
		indx = (cat['dist'] < 5) & (cat['tmag'][staridx]-cat['tmag'] > -5)
		nstars = int(np.sum(indx))

		# Get target star index in the reduced catalog of stars to fit:
		staridx = np.squeeze(np.where(cat[indx]['starid'] == self.starid))
		logger.debug('Target star index: %s', np.str(staridx))

		# Preallocate flux sum array for contamination calculation:
		fluxes_sum = np.zeros(nstars, dtype='float64')

		# Start looping through the images (time domain):
		for k, img in enumerate(self.images):
			# Get catalog at current time in MJD:
			cat = self.catalog_attime(self.lightcurve['time'][k] - self.lightcurve['timecorr'][k])

			# Reduce catalog to only include stars that should be fitted:
			cat = cat[indx]
			logger.debug(cat)

			# Get the number of pixels in the image:
			good_pixels = np.isfinite(img)
			npx = int(np.sum(good_pixels))

			# Create A, the 2D of vertically reshaped PRF 1D arrays:
			A = np.empty([npx, nstars], dtype='float64')
			for col, target in enumerate(cat):
				# Get star parameters with flux set to 1 and reshape:
				params0 = np.atleast_2d([target['row_stamp'], target['column_stamp'], 1.])

				# Fill out column of A with reshaped PRF array from one star:
				A[:, col] = self.psf.integrate_to_image(params0, ctoff_radius=20)[good_pixels].flatten()

			# Crate b, the solution array by reshaping the image to a 1D array:
			b = img[good_pixels].flatten()

			# Do linear least squares fit to solve Ax=b:
			try:
				# Linear least squares:
				res = np.linalg.lstsq(A, b)
				fluxes = res[0]

				# Non-negative linear least squares:
				#fluxes, rnorm = scipy.optimize.nnls(A, b)
			except np.linalg.LinAlgError:
				logger.debug("Linear PSF Fitting failed")
				fluxes = None

			# Pass result if fit did not fail:
			if fluxes is None:
				logger.warning("We should flag that this has not gone well.")
				self.lightcurve['flux'][k] = np.NaN
				self.lightcurve['quality'][k] = 1 # FIXME: Use the real flag!

			else:
				# Get flux of target star:
				result = fluxes[staridx]

				logger.debug('Fluxes are: %s', fluxes)
				logger.debug('Result is: %f', result)

				# Add the result of the main star to the lightcurve:
				self.lightcurve['flux'][k] = result

				# Add current fitted fluxes for contamination calculation:
				fluxes_sum += fluxes

				# Make plots for debugging:
				if self.plot and logger.isEnabledFor(logging.DEBUG):
					fig = plt.figure()
					result4plot = []
					for star, target in enumerate(cat):
						result4plot.append(np.array([
							target['row_stamp'],
							target['column_stamp'],
							fluxes[star]
						]))

					# Add subplots with the image, fit and residuals:
					ax_list = plot_image_fit_residuals(
						fig=fig,
						image=img,
						fit=self.psf.integrate_to_image(result4plot, cutoff_radius=20)
					)

					# Add star position to the first plot:
					ax_list[0].scatter(result4plot[staridx][1], result4plot[staridx][0], c='r', alpha=0.5)

					# Save figure to file:
					fig_name = 'tess_{0:011d}_linpsf_{1:05d}'.format(self.starid, k)
					save_figure(os.path.join(self.plot_folder, fig_name))
					plt.close(fig)

		# Set contamination to NaN if all flux values are NaN:
		if allnan(self.lightcurve['flux']):
			self.report_details(error='All target flux values are NaN.')
			return STATUS.ERROR

		# Divide by number of added fluxes to get the mean flux:
		fluxes_mean = fluxes_sum / np.sum(~np.isnan(self.lightcurve['flux']))
		logger.debug('Mean fluxes are: %s', fluxes_mean)

		# Calculate contamination from other stars in target PSF using latest A:
		not_target_star = np.arange(len(fluxes_mean)) != staridx
		contamination = np.sum(A[:,not_target_star].dot(fluxes_mean[not_target_star]) * A[:,staridx]) / fluxes_mean[staridx]

		logger.info("Contamination: %f", contamination)
		self.additional_headers['PSF_CONT'] = (contamination, 'PSF contamination')

		# If contamination is high, return a warning:
		if contamination > 0.1:
			self.report_details(error='High contamination')
			return STATUS.WARNING

		# Return whether you think it went well:
		return STATUS.OK
