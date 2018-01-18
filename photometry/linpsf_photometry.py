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

class linPSFPhotometry(BasePhotometry):

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
		"""Linear PSF Photometry"""

		logger = logging.getLogger(__name__)

		# Generate list of stars to fit:
		cat = self.catalog
		
		# loop through the image in time domain
			# set up A and b
			# do photometry with np.linalg.lstsq(A,b)
			# report to logger.debug how it went
			# if it went well:
				# save photometry output to lightcurve
				# do a debugging plot
			# if it didn't go so well:
				# report warning to logger
				# set lightcurve to np.NaN
		# return STATUS.OK
		