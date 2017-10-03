#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 17:52:09 2017

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
from six.moves import range, zip
import numpy as np
import logging
from .. import BasePhotometry
from . import k2p2v2 as k2p2

#------------------------------------------------------------------------------
class AperturePhotometry(BasePhotometry):
	"""Simple Aperture Photometry using K2P2 to define masks."""

	def __init__(self, starid):
		# Call the parent initializing:
		# This will set several default settings
		super(self.__class__, self).__init__(starid)

		# Here you could do other things that needs doing in the beginning
		# of the run on each target.

	def do_photometry(self):
		"""Perform photometry on the given target.

		This function needs to set
			* self.lightcurve
		"""

		logger = logging.getLogger(__name__)
		logger.info('='*80)
		logger.info("starid: %d", self.starid)

		for retries in range(1, 5):

			SumImage = self.sumimage
			logger.info("SumImage shape: %s", SumImage.shape)

			target_pixel_row = int(self.target_pos_row) - self.stamp[0]
			target_pixel_column = int(self.target_pos_column) - self.stamp[2]

			logger.info(self.stamp)
			logger.info("Target position in stamp: (%d,%d)", target_pixel_row, target_pixel_column )

			catalog = self.catalog

			# plt.figure()
			# plt.imshow(np.log10(self.sumimage), origin='lower')
			# plt.scatter(catalog['row']+0.5, catalog['column']+0.5, s=100/catalog['tmag'], c='r')

			cat = np.column_stack((catalog['row'], catalog['column'], catalog['tmag']))

			logger.info("Creating new masks...")
			masks, background_bandwidth = k2p2.k2p2FixFromSum(SumImage, None, plot_folder=None, thresh=5, min_no_pixels_in_mask=4, catalog=cat)
			masks = np.asarray(masks, dtype='bool')

			if len(masks.shape) == 0:
				logger.error("No masks found")
				return AperturePhotometry.STATUS_ERROR

			# Look at the central pixel where the target should be:
			indx_main = masks[:, target_pixel_row, target_pixel_column].flatten()

			if not np.any(indx_main):
				logger.error('No pixels')
				return AperturePhotometry.STATUS_ERROR
			elif np.sum(indx_main) > 1:
				logger.error('Too many masks')
				return AperturePhotometry.STATUS_ERROR

			# Mask of the main target:
			mask_main = masks[indx_main, :, :].reshape(SumImage.shape)

			resize_args = {}
			if np.any(mask_main[0, :]):
				resize_args['down'] = 10
			if np.any(mask_main[-1, :]):
				resize_args['up'] = 10
			if np.any(mask_main[:, 0]):
				resize_args['left'] = 10
			if np.any(mask_main[:, -1]):
				resize_args['right'] = 10

			if resize_args:
				logger.error("Touching the edges! Retrying")
				logger.info(resize_args)
				self.resize_stamp(**resize_args)
				logger.info('-'*70)
			else:
				break

		# XY of pixels in frame
		cols, rows = self.get_pixel_grid()
		members = np.column_stack((cols[mask_main], rows[mask_main]))

		#
		for k, (img, bck) in enumerate(zip(self.images, self.backgrounds)):

			flux_in_cluster = img[mask_main] - bck[mask_main]

			# Calculate flux in mask:
			self.lightcurve['flux'][k] = np.sum(flux_in_cluster)
			self.lightcurve['flux_background'][k] = np.sum(bck[mask_main])

			# Calculate flux centroid:
			finite_vals = (flux_in_cluster > 0)
			self.lightcurve['pos_centroid'][k, :] = np.average(members[finite_vals, :], weights=flux_in_cluster[finite_vals], axis=0)

		#
		self.final_mask = mask_main

		# Return whether you think it went well:
		return AperturePhotometry.STATUS_OK
