#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:08:36 2018

@author: Jonas Svenstrup Hansen <jonas.svenstrup@gmail.com>
"""

from __future__ import division, with_statement, print_function, absolute_import
import numpy as np
import scipy
import matplotlib.pyplot as plt
import logging
import os
from scipy.optimize import minimize
from .BasePhotometry import BasePhotometry, STATUS
from .psf import PSF
from .utilities import mag2flux
from .plots import plot_image_fit_residuals, save_figure
from .residual_mask import four_pixel_mask

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

		.. code author:: Jonas Svenstrup Hansen <jonas.svenstrup@gmail.com>
		"""
		# Call the parent initializing:
		# This will set several default settings
		super(self.__class__, self).__init__(*args, **kwargs)

		# Create instance of the PSF for the given pixel stamp:
		# NOTE: If we run resize_stamp at any point in the code,
		#       we should also update self.PSF.
		# TODO: Maybe we should move this into BasePhotometry?
		self.psf = PSF(self.camera, self.ccd, self.stamp)

	def _lhood(self, params, img, bkg, lhood_stat='Gaussian_d', include_bkg=True):
		"""
		Log-likelihood function to be minimized for the PSF fit.

		Parameters:
			params (numpy array): Parameters for the PSF integrator.
			img_bkg (list): List containing the image and background numpy arrays.
			lhood_stat (string): Determines what statistic to use. Default is
			``Gaussian_d``. Can also be ``Gaussian_m`` or ``Poisson``.
			include_bkg (boolean): Determine whether to include background. Default
			is ``True``.
		"""

		# Reshape the parameters into a 2D array:
		params = params.reshape(len(params)//3, 3)

		# Define minimum weights to avoid dividing by 0:
		minweight = 1e-9
		minvar = 1e-9

		# Pass the list of stars to the PSF integrator to produce an artificial image:
		mdl = self.psf.integrate_to_image(params, cutoff_radius=10)

		# Calculate the likelihood value:
		if lhood_stat.startswith('Gaussian'):
			if lhood_stat == 'Gaussian_d':
				if include_bkg:
					var = np.abs(img + bkg) # can be outside _lhood
				else:
					var = np.abs(img) # can be outside _lhood

			elif lhood_stat == 'Gaussian_m':
				if include_bkg:
					var = np.abs(mdl + bkg) # has to be in _lhood
				else:
					var = np.abs(mdl) # has to be in _lhood
			# Add 2nd term of Erwin (2015), eq. (13):
			var += self.n_readout * self.readnoise**2 / self.gain**2
			var[var<minvar] = minvar
			weightmap = 1 / var
			weightmap[weightmap<minweight] = minweight
			# Return the chi2:
			return np.nansum( weightmap * (img - mdl)**2 )

		elif lhood_stat == 'Poisson':
			# Prepare model for logarithmic expression by changing zeros to small values:
			mdl_for_log = mdl
			mdl_for_log[mdl_for_log < 1e-9] = 1e-9
			# Return the Cash statistic:
			return 2 * np.nansum( mdl - img * np.log(mdl_for_log) )

		elif lhood_stat == 'old_Gaussian':
			# Return the chi2:
			return np.nansum( (img - mdl)**2 / img )

		else:
			raise ValueError("Invalid statistic: '%s'" % lhood_stat)


	def do_photometry(self):
		"""Linear PSF Photometry
		TODO: add description of method and what A and b are
		"""

		logger = logging.getLogger(__name__)

		# Load catalog to determine what stars to fit:
		cat = self.catalog
		staridx = np.squeeze(np.where(cat['starid']==self.starid))

		# Log full catalog for current stamp:
		logger.debug(cat)

		# Calculate distance from main target:
		cat['dist'] = np.sqrt((cat['row_stamp'][staridx] - cat['row_stamp'])**2 + \
						(cat['column_stamp'][staridx] - cat['column_stamp'])**2)

		# Find indices of stars in catalog to fit:
		# (only include stars that are close to the main target and that are
		# not much fainter)
		indx = (cat['dist'] < 5) & (cat['tmag'][staridx]-cat['tmag'] > -5)
		nstars = np.sum(indx)

		# Reduce catalog to only include stars that should be fitted:
		cat = cat[indx]

		# Get target star index in the reduced catalog of stars to fit:
		staridx = np.squeeze(np.where(cat['starid']==self.starid))
		logger.debug('Target star index: %s', np.str(staridx))

		# Find catalog inaccuracies by PSF fit to the sum image:
		PSF_correction_factor = 0.
		if PSF_correction_factor != 0:
			# Prepare catalog for minimizer:
			params0 = np.empty((len(cat), 3), dtype='float64')
			for k, target in enumerate(cat):
				params0[k,:] = [target['row_stamp'],
								target['column_stamp'],
								mag2flux(target['tmag'])]
			params0 = params0.flatten() # Make the parameters into a 1D array

			# Call minimizer with sumimage and zero background:
			img = self.sumimage
			bkg = np.zeros_like(img)
			maxiter = 1500
			try:
				res_PSF = minimize(self._lhood,
					params0, args=(img, bkg), method='Nelder-Mead',
					options={'maxiter': maxiter})
			except:
				res_PSF = 'failed'
				logger.info('Initial PSF fit to determine location failed.')

			# Get PSF-catalog offset results from minimize results object:
			if res_PSF == 'failed':
				pass
			else:
				if res_PSF.x.ndim > 1:
					PSF_position = res_PSF.x[staridx, 0:2]
				else:
					PSF_position = res_PSF.x[0:2]
				logger.info('PSF fit to sumimage target [row,col] result: {0}'.format(PSF_position))

				# Determine offset magnitude:
				PSF_offset = PSF_position - np.array([cat['row_stamp'][staridx],
												cat['column_stamp'][staridx]])
				PSF_offset_norm = np.linalg.norm(PSF_offset)
				logger.info('Catalog off by {:3.12f} pixel'.format(PSF_offset_norm))
				if PSF_offset_norm > 0.9:
					self.report_details(error='Catalog position off by {0} pixel'.format(PSF_offset_norm))

		# Preallocate flux sum array for contamination calculation:
		fluxes_sum = np.zeros(nstars)

		# Start looping through the images (time domain):
		for k, img in enumerate(self.images):

			# Get catalog at current time in MJD:
			cat = self.catalog_attime(self.lightcurve['time'][k])

			# Modify catalog with PSF fit to sumimage to fix catalog errors:
			if PSF_correction_factor != 0 and res_PSF != 'failed':
				PSF_offset = PSF_position - np.array([cat['row_stamp'][staridx],
											cat['column_stamp'][staridx]])
				cat['row_stamp'][staridx] += PSF_correction_factor*PSF_offset[0]
				cat['column_stamp'][staridx] += PSF_correction_factor*PSF_offset[1]

			# Reduce catalog to only include stars that should be fitted:
			cat = cat[indx]

			# Log reduced catalog for the stamp at the current time:
			logger.debug(cat)

			# Get the number of pixels in the image:
			npx = img.size

			# Create A, the 2D of vertically reshaped PRF 1D arrays:
			A = np.empty([npx, nstars])

			# Preallocate target row and col position:
			for col,target in enumerate(cat):
				# Get star parameters with flux set to 1 and reshape:
				params0 = np.array(
						[target['row_stamp'], target['column_stamp'], 1.]
						).reshape(1, 3)
				# Write
				if col == staridx:
					target_row = params0[0][0]
					target_col = params0[0][1]

				# Fill out column of A with reshaped PRF array from one star:
				A[:,col] = np.reshape(self.psf.integrate_to_image(params0,
										cutoff_radius=20), npx)

			# Crate b, the solution array by reshaping the image to a 1D array:
			b = np.reshape(img, npx)

			# Do linear least squares fit to solve Ax=b:
			try:
				# Linear least squares:
				res = np.linalg.lstsq(A,b)
				fluxes = res[0]

				# Non-negative linear least squares:
#				fluxes, rnorm = scipy.optimize.nnls(A,b)
#				res = 'notfailed'
			except:
				res = 'failed'
			logger.debug('Result of linear psf photometry: ' + np.str(res))

			# Do non-negative least squares fit if the target had negative flux:
			if fluxes[staridx] < 0:
				logger.debug('Negative fitted target flux. Re-fitting with non-negative algorithm')
				try:
					fluxes, rnorm = scipy.optimize.nnls(A,b)
					res = 'notfailed'
				except:
					res = 'failed'

			# Pass result if fit did not fail:
			if res is not 'failed':
				# Get flux of target star:
				result = fluxes[staridx]

				logger.debug('PSF fitted fluxes are: ' + np.str(fluxes))
				logger.debug('PSF fitted result is: ' + np.str(result))

				# Generate fitted and residual images from A and fitted fluxes:
				img_fit = np.reshape(np.sum(A*fluxes, 1), img.shape)
				img_res = img - img_fit

				# Get indices of mask in residual image:
				res_mask = four_pixel_mask(target_row, target_col)
				logger.debug('Indices of residual mask, 2D: ' + np.array_str(res_mask))
				res_mask_for_ravel = ([idx[0] for idx in res_mask],[idx[1] for idx in res_mask])
				res_mask = np.ravel_multi_index(res_mask_for_ravel, dims=img.shape)
				logger.debug('Indices of residual mask, ravelled: ' + np.array_str(res_mask))

				# Do aperture photometry on residual image:
				res_mask_sum = np.sum(img_res.ravel()[res_mask])
				logger.debug('Residual aperture photometry result: ' + np.str(res_mask_sum))

				# Add residual photometry result to target flux value:
				result += res_mask_sum

				# Add the result of the main star to the lightcurve:
				self.lightcurve['flux'][k] = result
				self.lightcurve['pos_centroid'][k] = [np.NaN, np.NaN]
				self.lightcurve['quality'][k] = 0

				# Add current fitted fluxes for contamination calculation:
				fluxes_sum += fluxes

				if self.plot:
					# Make plot for debugging:
					fig = plt.figure()

					# Add subplots with the image, fit and residuals:
					ax_list = plot_image_fit_residuals(fig=fig,
							image=img,
							fit=img_fit,
							residuals=img_res)

					# Set subplot titles:
					title_list = ['Simulated image', 'Least squares PSF fit', 'Residual image']
					for ax, title in zip(ax_list, title_list):
						# Add title to subplot:
						ax.set_title(title)

						# Add star position to subplot:
						# TODO: get target star position from somewhere else than result4plot which is to be outphased
						ax_list[0].scatter(target_col, target_row, c='r', alpha=0.5)

					# Save figure to file:
					fig_name = 'tess_{0:09d}'.format(self.starid) + '_linpsf_{0:09d}'.format(k)
					save_figure(os.path.join(self.plot_folder, fig_name))
					plt.close(fig)

			# Pass result if fit failed:
			else:
				logger.warning("We should flag that this has not gone well.")

				self.lightcurve['flux'][k] = np.NaN
				self.lightcurve['pos_centroid'][k] = [np.NaN, np.NaN]
				self.lightcurve['quality'][k] = 1 # FIXME: Use the real flag!


		if np.sum(np.isnan(self.lightcurve['flux'])) == len(self.lightcurve['flux']):
			# Set contamination to NaN if all flux values are NaN:
			self.report_details(error='All target flux values are NaN.')
			return STATUS.ERROR
		else:
			# Divide by number of added fluxes to get the mean flux:
			fluxes_mean =  fluxes_sum / np.sum(~np.isnan(self.lightcurve['flux']))
			logger.debug('Mean fluxes are: '+np.str(fluxes_mean))

			# Calculate contamination from other stars in target PSF using latest A:
			not_target_star = np.arange(len(fluxes_mean))!=staridx
			contamination = \
				np.sum(A[:,not_target_star].dot(fluxes_mean[not_target_star]) * A[:,staridx]) \
				/fluxes_mean[staridx]

			logger.info("Contamination: %f", contamination)
			self.additional_headers['AP_CONT'] = (contamination, 'AP contamination')

			# If contamination is high, return a warning:
			if contamination > 0.1:
				self.report_details(error='High contamination')
				return STATUS.WARNING


		# Return whether you think it went well:
		return STATUS.OK
