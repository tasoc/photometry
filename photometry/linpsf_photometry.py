#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:08:36 2018

@author: Jonas Svenstrup Hansen <jonas.svenstrup@gmail.com>
"""

from __future__ import division, with_statement, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import logging
from .BasePhotometry import BasePhotometry, STATUS
from .psf import PSF
from .utilities import mag2flux

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

	def do_photometry(self):
		"""Linear PSF Photometry
		TODO: add description of method and what A and b are
		"""

		logger = logging.getLogger(__name__)

		# Start looping through the images (time domain):
		for k, img in enumerate(self.images):
			# Get catalog at current time in MJD:
			cat = self.catalog_attime(self.lightcurve['time'][k])

			# Get target star index in the catalog:
			staridx = np.where(cat['starid']==self.starid)[0][0]

			# Calculate distance from main target:
			# FIXME: the target_pos_row_stamp is not at the right time!
			cat['dist'] = np.sqrt((cat['row_stamp'][staridx] - cat['row_stamp'])**2 + \
							(cat['column_stamp'][staridx] - cat['column_stamp'])**2)

			# Log full catalog for current stamp:
			logger.debug(cat)
	
			# Only include stars that are close to the main target and that are not much fainter:
			cat = cat[(cat['dist'] < 5) & (cat['tmag'][staridx]-cat['tmag'] > -10)]

			# Log reduced catalog for current stamp:
			logger.debug(cat)

			# Update target star index in the reduced catalog:
			staridx_cut = np.where(cat['starid']==self.starid)[0][0]

			# Get info about the image:
			npx = img.size
			nstars = len(cat['tmag'])

			# Set up parameters for PSF integration to PRF:
			params = np.empty((len(cat), 3), dtype='float64')
			for row, target in enumerate(cat):
				params[row,:] = [target['row_stamp'], target['column_stamp'], 
								mag2flux(target['tmag'])]

			# Create A, the 2D of vertically reshaped PRF 1D arrays:
			A = np.empty([npx, nstars])
			for star,col in enumerate(range(nstars)):
				# Reshape the parameters of each single star in the loop:
				params0 = params[star,:].reshape(1, 3)

				# Fill out column of A with reshaped PRF array from one star:
				A[:,col] = np.reshape(self.psf.integrate_to_image(params0, 
										cutoff_radius=20), npx)
			
			# Crate b, the solution array by reshaping the image to a 1D array:
			b = np.reshape(img, npx)

			# Do linear least squares fit to solve Ax=b:
			try:
				res = np.linalg.lstsq(A,b)
			except:
				res = 'failed'
			logger.debug(res)

			# Pass result if fit did not fail:
			if res is not 'failed':
				# Get flux of target star:
				fluxes = res[0]
				result = fluxes[staridx_cut]
				logger.debug(result)

				# Add the result of the main star to the lightcurve:
				self.lightcurve['flux'][k] = result
				self.lightcurve['pos_centroid'][k] = [np.NaN, np.NaN]
#				self.lightcurve['pos_centroid'][k] = params[k,0:2]
				self.lightcurve['quality'][k] = 0

				fig = plt.figure()
				result4plot = []
				for star in range(nstars):
					result4plot.append(np.array([params[star,0], 
												params[star,1],
												fluxes[star]]))
				# Plot image:
				ax = fig.add_subplot(131)
				ax.imshow(np.log10(img), origin='lower')
				# Plot least squares fit:
				ax = fig.add_subplot(132)
				ax.imshow(np.log10(self.psf.integrate_to_image(result4plot)), origin='lower')
				# Plot the residuals:
				ax = fig.add_subplot(133)
				ax.imshow(img - self.psf.integrate_to_image(result4plot), origin='lower')
#				plt.show()


			# Pass result if fit failed:
			else:
				logger.warning("We should flag that this has not gone well.")

				self.lightcurve['flux'][k] = np.NaN
				self.lightcurve['pos_centroid'][k] = [np.NaN, np.NaN]
				self.lightcurve['quality'][k] = 1 # FIXME: Use the real flag!

		# Return whether you think it went well:
		return STATUS.OK
