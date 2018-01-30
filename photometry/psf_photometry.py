#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 16:37:12 2017

@author: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import logging
from copy import deepcopy
from scipy.optimize import minimize
from .BasePhotometry import BasePhotometry, STATUS
from .psf import PSF
from .utilities import mag2flux

class PSFPhotometry(BasePhotometry):

	def __init__(self, *args, **kwargs):
		# Call the parent initializing:
		# This will set several default settings
		super(self.__class__, self).__init__(*args, **kwargs)

		# Create instance of the PSF for the given pixel stamp:
		# NOTE: If we run resize_stamp at any point in the code,
		#       we should also update self.PSF.
		# TODO: Maybe we should move this into BasePhotometry?
		self.psf = PSF(self.camera, self.ccd, self.stamp)

	def _lhood(self, params, img, lhood_stat = 'Gaussian_d'):
		# Reshape the parameters into a 2D array:
		params = params.reshape(len(params)//3, 3)
		# Pass the list of stars to the PSF integrator to produce an artificial image:
		mdl = self.psf.integrate_to_image(params, cutoff_radius=10)
		# Define minimum weights to avoid dividing by 0:
		minweight = 1e-9
		minvar = 1e-9
		# Calculate the likelihood value:
		if lhood_stat[0:8] == 'Gaussian':
			if lhood_stat == 'Gaussian_m':
				# FIXME: Include background here:
#				var = np.abs(img + self.background) # can be outside _lhood
				var = np.abs(img) # can be outside _lhood
			elif lhood_stat == 'Gaussian_d':
				# FIXME: Include background here:
#				var = np.abs(mdl + self.background) # has to be in _lhood
				var = np.abs(mdl) # has to be in _lhood
			# Add 2nd term of Erwin (2015), eq. (13):
			var += self.n_readout * self.readnoise**2 / self.gain**2
			var[var<minvar] = minvar
			weightmap = 1 / var
			weightmap[weightmap<minweight] = minweight
			# Return the chi2:
			return np.nansum( weightmap * (img - mdl)**2 )
		elif lhood_stat == 'Poisson':
			# Return the Cash statistic:
			mdlforlog = mdl
			mdlforlog[mdlforlog < 1e-9] = 1e-9 # set 0 to low
			return 2 * np.nansum( mdl - img * np.log(mdlforlog) )
		elif lhood_stat == 'old_Gaussian':
			# Return the chi2:
			return np.nansum( (img - mdl)**2 / img )

	def do_photometry(self):
		"""PSF Photometry"""

		logger = logging.getLogger(__name__)

		# Generate list of stars to fit:
		cat = self.catalog

		# Calculate distance from main target:
		cat['dist'] = np.sqrt((self.target_pos_row_stamp - cat['row_stamp'])**2 + (self.target_pos_column_stamp - cat['column_stamp'])**2)

		# Only include stars that are close to the main target and that are not much fainter:
		cat = cat[(cat['dist'] < 5) & (self.target_tmag-cat['tmag'] > -5)]

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
		params_start = deepcopy(params0) # Save the starting parameters for later
		params0 = params0.flatten() # Make the parameters into a 1D array

		# Start looping through the images (time domain):
		for k, img in enumerate(self.images):
			# Run the fitting routine for this image:
			res = minimize(self._lhood, params0, args=img, method='Nelder-Mead')
			logger.debug(res)

			if res.success:
				result = res.x
				result = np.array(result.reshape(len(result)//3, 3))
				logger.debug(result)

				# Add the result of the main star to the lightcurve:
				self.lightcurve['flux'][k] = result[0,2]
				self.lightcurve['pos_centroid'][k] = result[0,0:2]
				self.lightcurve['quality'][k] = 0

				fig = plt.figure()
				ax = fig.add_subplot(131)
				ax.imshow(np.log10(img), origin='lower')
				ax.scatter(params_start[:,1], params_start[:,0], c='b', alpha=0.5)
				ax.scatter(result[:,1], result[:,0], c='r', alpha=0.5)
				ax = fig.add_subplot(132)
				ax.imshow(np.log10(self.psf.integrate_to_image(result)), origin='lower')
				ax = fig.add_subplot(133)
				ax.imshow(img - self.psf.integrate_to_image(result, cutoff_radius=10), origin='lower')
#				plt.show()

				# In the next iteration, start from the current solution:
				params0 = res.x
			else:
				logger.warning("We should flag that this has not gone well.")

				self.lightcurve['flux'][k] = np.NaN
				self.lightcurve['pos_centroid'][k] = [np.NaN, np.NaN]
				self.lightcurve['quality'][k] = 1 # FIXME: Use the real flag!

		# Return whether you think it went well:
		return STATUS.OK
