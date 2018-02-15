#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Aperture Photometry using K2P2 to define masks.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
from six.moves import range, zip
import os
import numpy as np
import logging
from .. import BasePhotometry, STATUS
from . import k2p2v2 as k2p2

#------------------------------------------------------------------------------
class AperturePhotometry(BasePhotometry):
	"""Simple Aperture Photometry using K2P2 to define masks.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	def __init__(self, *args, **kwargs):
		# Call the parent initializing:
		# This will set several default settings
		super(self.__class__, self).__init__(*args, **kwargs)

		# Here you could do other things that needs doing in the beginning
		# of the run on each target.

	def _minimum_aperture(self):
		x = int(self.target_pos_row_stamp + 0.5)
		y = int(self.target_pos_column_stamp - 0.5)
		mask_main = np.zeros_like(self.sumimage, dtype='bool')
		mask_main[y:y+2, x:x+2] = True
		return mask_main

	def do_photometry(self):
		"""Perform photometry on the given target.

		This function needs to set
			* self.lightcurve
		"""

		logger = logging.getLogger(__name__)
		logger.info('='*80)
		logger.info("starid: %d", self.starid)

		for retries in range(5):

			SumImage = self.sumimage
			logger.info("SumImage shape: %s", SumImage.shape)

			logger.info(self.stamp)
			logger.info("Target position in stamp: (%f, %f)", self.target_pos_row_stamp, self.target_pos_column_stamp )

			cat = np.column_stack((self.catalog['column_stamp'], self.catalog['row_stamp'], self.catalog['tmag']))

			logger.info("Creating new masks...")
			k2p2_settings = {
				'thresh': 1,
				'min_no_pixels_in_mask': 4
			}

			masks, background_bandwidth = k2p2.k2p2FixFromSum(SumImage, None, plot_folder=self.plot_folder, show_plot=False, catalog=cat, **k2p2_settings)
			masks = np.asarray(masks, dtype='bool')

			if len(masks.shape) == 0:
				logger.warning("No masks found")
				self.report_details(error='No masks found')
				mask_main = self._minimum_aperture()

			else:
				# Look at the central pixel where the target should be:
				indx_main = masks[:, int(self.target_pos_row_stamp), int(self.target_pos_column_stamp)].flatten()

				if not np.any(indx_main):
					logger.warning('No pixels')
					self.report_details(error='No pixels')
					mask_main = self._minimum_aperture()

				elif np.sum(indx_main) > 1:
					logger.error('Too many masks')
					self.report_details(error='Too many masks')
					return STATUS.ERROR

				else:
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
				if not self.resize_stamp(**resize_args):
					resize_args = {}
					logger.warning("Could not resize stamp any further")
					break
			else:
				break

		# If we reached the last retry but still needed a resize, give up:
		if resize_args:
			self.report_details(error='Too many stamp resizes')
			return STATUS.ERROR

		# XY of pixels in frame
		cols, rows = self.get_pixel_grid()
		members = np.column_stack((cols[mask_main], rows[mask_main]))

		# Loop through the images and backgrounds together:
		for k, (img, bck) in enumerate(zip(self.images, self.backgrounds)):

			flux_in_cluster = img[mask_main] - bck[mask_main]

			# Calculate flux in mask:
			self.lightcurve['flux'][k] = np.sum(flux_in_cluster)
			self.lightcurve['flux_background'][k] = np.sum(bck[mask_main])

			# Calculate flux centroid:
			finite_vals = (flux_in_cluster > 0)
			if np.any(finite_vals):
				self.lightcurve['pos_centroid'][k, :] = np.average(members[finite_vals], weights=flux_in_cluster[finite_vals], axis=0)
			else:
				self.lightcurve['pos_centroid'][k, :] = np.NaN

		# Save the mask to be stored in the outout file:
		self.final_mask = mask_main

		# Add additional headers specific to this method:
		#self.additional_headers['KP_SUBKG'] = (bool(subtract_background), 'K2P2 subtract background?')
		self.additional_headers['KP_THRES'] = (k2p2_settings['thresh'], 'K2P2 sum-image threshold')
		self.additional_headers['KP_MIPIX'] = (k2p2_settings['min_no_pixels_in_mask'], 'K2P2 min pixels in mask')
		#self.additional_headers['KP_MICLS'] = (k2p2_settings['min_for_cluster'], 'K2P2 min pix. for cluster')
		#self.additional_headers['KP_CLSRA'] = (k2p2_settings['cluster_radius'], 'K2P2 cluster radius')
		#self.additional_headers['KP_WS'] = (bool(ws), 'K2P2 watershed segmentation')
		#self.additional_headers['KP_WSALG'] = (k2p2_settings['ws_alg'], 'K2P2 watershed weighting')
		#self.additional_headers['KP_WSBLR'] = (k2p2_settings['ws_blur'], 'K2P2 watershed blur')
		#self.additional_headers['KP_WSTHR'] = (k2p2_settings['ws_threshold'], 'K2P2 watershed threshold')
		#self.additional_headers['KP_WSFOT'] = (k2p2_settings['ws_footprint'], 'K2P2 watershed footprint')
		#self.additional_headers['KP_EX'] = (bool(extend_overflow), 'K2P2 extend overflow')

		# Targets that are in the mask:
		target_in_mask = [k for k,t in enumerate(self.catalog) if np.floor(t['row'])+1 in rows[mask_main] and np.floor(t['column'])+1 in cols[mask_main]]

		# Calculate contamination from the other targets in the mask:
		if len(target_in_mask) == 1 and self.catalog[target_in_mask][0]['starid'] == self.starid:
			contamination = 0
		else:
			# Calculate contamination metric as defined in Lund & Handberg (2014):
			mags_in_mask = self.catalog[target_in_mask]['tmag']
			mags_total = -2.5*np.log10(np.nansum(10**(-0.4*mags_in_mask)))
			contamination = 1.0 - 10**(0.4*(mags_total - self.target_tmag))
			contamination = np.abs(contamination) # Avoid stupid signs due to round-off errors

		logger.info("Contamination: %f", contamination)
		self.additional_headers['AP_CONT'] = (contamination, 'AP contamination')

		# If contamination is high, return a warning:
		if contamination > 0.1:
			self.report_details(error='High contamination')
			return STATUS.WARNING

		#
		skip_targets = [t['starid'] for t in self.catalog[target_in_mask] if t['starid'] != self.starid]
		if skip_targets:
			logger.info("These stars could be skipped:")
			logger.info(skip_targets)
			self.report_details(skip_targets=skip_targets)

		# Return whether you think it went well:
		return STATUS.OK
