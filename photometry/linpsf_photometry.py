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
			time domain loop structure and lightcurve output is copied from 
			that class.
		
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

			# TODO: limit amount of stars to fit here if necessary

			# Get info about the image:
			npx = img.size
			nstars = len(cat['tmag'])
			currentstar = 0
			# FIXME: check that this is the right star!

			# Set up parameters for PSF integration to PRF:
			params = np.empty((len(cat), 3), dtype='float64')
			for k, target in enumerate(cat):
				params[k,:] = [target['row_stamp'], target['column_stamp'], mag2flux(target['tmag'])]
		

			# Preallocate A:
			A = np.empty([npx, nstars])
			# Set up A and b:
			for col in range(nstars):
				# Reshape the parameters into a 2D array:
				params = params.reshape(len(params)//3, 3)

				# Fill out column of A with reshaped PRF:
				A[:,col] = np.reshape(self.psf.integrate_to_image(params, 
										cutoff_radius=20), npx)
			b = np.reshape(img, npx)

			# Do linear least squares fit:
			try:
				res = np.linalg.lstsq(A,b)
			except:
				res = 'failed'
			logger.debug(res)

			if res is not 'failed':
				result = res[0][currentstar]
				logger.debug(result)

				# Add the result of the main star to the lightcurve:
				self.lightcurve['flux'][k] = result
				self.lightcurve['pos_centroid'][k] = result[0,0:2]
				self.lightcurve['quality'][k] = 0

				#TODO: do a debugging plot
			else:
				logger.warning("We should flag that this has not gone well.")

				self.lightcurve['flux'][k] = np.NaN
				self.lightcurve['pos_centroid'][k] = [np.NaN, np.NaN]
				self.lightcurve['quality'][k] = 1 # FIXME: Use the real flag!

		# Return whether you think it went well:
		return STATUS.OK
