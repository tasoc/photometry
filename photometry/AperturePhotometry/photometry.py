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
	"""Simple Aperture Photometry using K2P2 to define masks.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	def __init__(self, starid, input_folder):
		# Call the parent initializing:
		# This will set several default settings
		super(self.__class__, self).__init__(starid, input_folder)

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

			cat = np.column_stack((self.catalog['row'], self.catalog['column'], self.catalog['tmag']))

			logger.info("Creating new masks...")
			k2p2_settings = {
				'thresh': 5,
				'min_no_pixels_in_mask': 4
			}

			masks, background_bandwidth = k2p2.k2p2FixFromSum(SumImage, None, plot_folder=None, catalog=cat, **k2p2_settings)
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

			# Find out if we are touching any of the edges:
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
				logger.warning("Touching the edges! Retrying")
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

		# Add additional headers specific to this method:
		#if custom_mask:
		#	self.additional_headers['KP_MODE']	= 'Custom Mask'
		#else:
		#	self.additional_headers['KP_MODE'] = 'Normal'
		#self.additional_headers['KP_SUBKG'] = (bool(subtract_background), 'K2P2 subtract background?')
		#self.additional_headers['KP_THRES'] = (k2p2_settings['sumimage_threshold'], 'K2P2 sum-image threshold')
		#self.additional_headers['KP_MIPIX'] = (k2p2_settings['min_no_pixels_in_mask'], 'K2P2 min pixels in mask')
		#self.additional_headers['KP_MICLS'] = (k2p2_settings['min_for_cluster'], 'K2P2 min pix. for cluster')
		#self.additional_headers['KP_CLSRA'] = (k2p2_settings['cluster_radius'], 'K2P2 cluster radius')
		#self.additional_headers['KP_WS'] = (bool(ws), 'K2P2 watershed segmentation')
		#self.additional_headers['KP_WSALG'] = (k2p2_settings['ws_alg'], 'K2P2 watershed weighting')
		#self.additional_headers['KP_WSBLR'] = (k2p2_settings['ws_blur'], 'K2P2 watershed blur')
		#self.additional_headers['KP_WSTHR'] = (k2p2_settings['ws_threshold'], 'K2P2 watershed threshold')
		#self.additional_headers['KP_WSFOT'] = (k2p2_settings['ws_footprint'], 'K2P2 watershed footprint')
		#self.additional_headers['KP_EX'] = (bool(extend_overflow), 'K2P2 extend overflow')

		# Return whether you think it went well:
		return AperturePhotometry.STATUS_OK
