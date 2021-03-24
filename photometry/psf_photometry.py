#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSF Photometry.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import os.path
import logging
import numpy as np
from bottleneck import nansum
#from copy import deepcopy
from scipy.optimize import minimize
from . import BasePhotometry, STATUS
from .utilities import mag2flux
from .plots import plt, plot_image_fit_residuals, save_figure, plot_outline

class PSFPhotometry(BasePhotometry):

	def __init__(self, *args, **kwargs):
		# Call the parent initializing:
		# This will set several default settings
		super().__init__(*args, **kwargs)

		self.cutoff_radius = 5

	#----------------------------------------------------------------------------------------------
	def _minimum_aperture(self):
		# Map of valid pixels that can be included:
		collected_pixels = (self.aperture & 1 != 0)

		# Create minimum 2x2 mask around target position:
		cols, rows = self.get_pixel_grid()
		mask_main = (( np.abs(cols - self.target_pos_column - 1) <= 1 )
			& ( np.abs(rows - self.target_pos_row - 1) <= 1 ))

		# Return the 2x2 mask, but only the pixels that are actually collected:
		return mask_main & collected_pixels

	#----------------------------------------------------------------------------------------------
	def _logprior(self, params):
		# Reshape the parameters into a 2D array:
		params = params.reshape(len(params)//3, 3)
		fluxes = params[:, 2]
		# Fluxes are not allowed to be negative:
		if np.any(fluxes < 0):
			return -np.inf
		return 0

	#----------------------------------------------------------------------------------------------
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
		mdl = self.psf.integrate_to_image(params, cutoff_radius=self.cutoff_radius)

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
			var[var < minvar] = minvar
			weightmap = 1 / var
			weightmap[weightmap < minweight] = minweight
			# Return the chi2:
			return nansum( weightmap * (img - mdl)**2 )

		elif lhood_stat == 'Poisson':
			# Prepare model for logarithmic expression by changing zeros to small values:
			mdl_for_log = mdl
			mdl_for_log[mdl_for_log < 1e-9] = 1e-9
			# Return the Cash statistic:
			return 2 * nansum( mdl - img * np.log(mdl_for_log) )

		elif lhood_stat == 'old_Gaussian':
			# Return the chi2:
			return nansum( (img - mdl)**2 / img )

		else:
			raise ValueError("Invalid statistic: '%s'" % lhood_stat)

	#----------------------------------------------------------------------------------------------
	def do_photometry(self):
		"""PSF Photometry"""

		logger = logging.getLogger(__name__)

		# Generate list of stars to fit:
		cat = self.catalog

		# Calculate distance from main target:
		cat['dist'] = np.sqrt((self.target_pos_row_stamp - cat['row_stamp'])**2 + (self.target_pos_column_stamp - cat['column_stamp'])**2)

		# Only include stars that are close to the main target and that are not much fainter:
		cat = cat[(cat['dist'] < 5) & (self.target['tmag']-cat['tmag'] > -5)]

		# Sort the catalog by distance and include at max the five closest stars:
		# FIXME: Make sure that the main target is in there!!!
		cat.sort('dist')
		if len(cat) > 5:
			cat = cat[:5]

		# Because the minimize routine used below only likes 1D numpy arrays
		# we have to restructure the catalog:
		params0 = np.empty((len(cat), 3), dtype='float64')
		for k, target in enumerate(cat):
			params0[k,:] = [target['row_stamp'], target['column_stamp'], mag2flux(target['tmag'])]
		#params_start = deepcopy(params0) # Save the starting parameters for later
		params0 = params0.flatten() # Make the parameters into a 1D array

		# Small aperture around target centre to do MOMF-style aperture correction on:
		mini_aperture = self._minimum_aperture()

		# Start looping through the images (time domain):
		for k, (img, bkg) in enumerate(zip(self.images, self.backgrounds)):
			# Print timestep index to logger:
			logger.debug('Current timestep: %d', k)

			# Set the maximum number of iterations for the minimize routine:
			if k > 0:
				maxiter = 500
			else: # The first step requires more iterations due to bad starting guess
				maxiter = 1500

			# Run the fitting routine for this image:
			res = minimize(self._lhood, params0, args=(img, bkg), method='Nelder-Mead',
				options={'maxiter': maxiter})
			logger.debug(res)

			if res.success:
				result = res.x
				result = np.array(result.reshape(len(result)//3, 3))
				logger.debug(result)
				target_flux = result[0, 2]

				# Calculate final best fit image and residuals:
				best_fit = self.psf.integrate_to_image(result, cutoff_radius=self.cutoff_radius)
				residuals = img - best_fit

				# Perform aperture photometry on the residuals:
				flux_ap = nansum(residuals[mini_aperture])
				logger.debug("Aperture correction: %f%%", 100*flux_ap/target_flux)
				target_flux += flux_ap

				# Add the result of the main star to the lightcurve:
				self.lightcurve['flux'][k] = target_flux
				self.lightcurve['flux_err'][k] = np.NaN # FIXME: Add errors!
				self.lightcurve['pos_centroid'][k] = result[0, 0:2]

				# TODO: use debug figure toggle to decide if to plot and export
				if self.plot and logger.isEnabledFor(logging.DEBUG):
					fig = plt.figure()
					ax_list = plot_image_fit_residuals(fig, img, best_fit, residuals)
					ax_list[0].scatter(result[:,1], result[:,0], c='r', alpha=0.5)
					plot_outline(mini_aperture, ax=ax_list[2], color='k', lw=2)
					fig_name = f'psf_photometry_s{self.sector:02d}_{self.starid:011d}_{k:05d}'
					save_figure(os.path.join(self.plot_folder, fig_name), fig=fig)
					plt.close(fig)

				# In the next iteration, start from the current solution:
				params0 = res.x
			else:
				logger.warning("We should flag that this has not gone well.")

				self.lightcurve['flux'][k] = np.NaN
				self.lightcurve['flux_err'][k] = np.NaN
				self.lightcurve['pos_centroid'][k] = [np.NaN, np.NaN]
				#self.lightcurve['quality'][k] |= 1 # FIXME: Use the real flag!

		# Return whether you think it went well:
		return STATUS.OK
