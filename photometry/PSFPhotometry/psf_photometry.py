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
from ..BasePhotometry import BasePhotometry, STATUS
from ..psf import PSF

class PSFPhotometry(BasePhotometry):

	def __init__(self, *args, **kwargs):
		# Call the parent initializing:
		# This will set several default settings
		super(self.__class__, self).__init__(*args, **kwargs)

		# Create instance of the PSF for the given pixel stamp:
		self.psf = PSF(self.stamp)

	def _lhood(self, params, img):
		# Reshape the parameters into a 2D array:
		params = params.reshape(len(params)//3, 3)
		# Pass the list of stars to the PSF integrator to produce an artificial image:
		mdl = self.psf.integrate_to_image(params, cutoff_radius=10)
		# Return the chi2:
		return np.nansum( (img - mdl)**2 / img )

	def do_photometry(self):

		logger = logging.getLogger(__name__)

		# Generate list of stars to fit:
		cat = self.catalog
		cat.sort('tmag')

		dist = np.sqrt( (self.target_pos_row_stamp - cat['row_stamp'])**2 + (self.target_pos_column_stamp - cat['column_stamp'])**2 )

		cat = cat[(dist < 5) & (cat['tmag'] < 15) & (cat['row_stamp'] > 0) & (cat['column_stamp'] > 0)]
		if len(cat) > 5:
			cat = cat[:5]

		params0 = np.empty((0,3), dtype='float64')
		for target in cat[('row_stamp', 'column_stamp', 'tmag')]:
			flux = 10**(-0.4*(target['tmag'] - 28.24))
			params0 = np.append(params0, [[target['row_stamp'], target['column_stamp'], flux]], axis=0)
		print(params0)
		params_start = deepcopy(params0)

		params0 = params0.flatten()

		for img in self.images:
			fig = plt.figure()
			ax = fig.add_subplot(111)
			ax.imshow(np.log10(img), origin='lower')
			ax.scatter(params_start[:,1], params_start[:,0], c='b', alpha=0.5)
			plt.show()

			res = minimize(self._lhood, params0, args=img, method='Nelder-Mead')
			logger.info(res)

			if res.success:
				result = res.x
				result = np.array(result.reshape(len(result)//3, 3))
				print(result)

				fig = plt.figure()
				ax = fig.add_subplot(131)
				ax.imshow(np.log10(img), origin='lower')
				ax.scatter(params_start[:,1], params_start[:,0], c='b', alpha=0.5)
				ax.scatter(result[:,1], result[:,0], c='r', alpha=0.5)

				ax = fig.add_subplot(132)
				ax.imshow(np.log10(self.psf.integrate_to_image(result)), origin='lower')

				ax = fig.add_subplot(133)
				ax.imshow(img - self.psf.integrate_to_image(result, cutoff_radius=10), origin='lower')

				fig.show()
				plt.show()

				# In the next iteration, start from the current solution:
				params0 = res.x

		# Return whether you think it went well:
		return STATUS.OK
