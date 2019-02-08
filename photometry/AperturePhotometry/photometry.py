#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Aperture Photometry using K2P2 to define masks.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
from six.moves import range, zip
import numpy as np
from bottleneck import allnan
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
		cols, rows = self.get_pixel_grid()
		mask_main = ( np.abs(cols - self.target_pos_column - 1) <= 1 ) \
					& ( np.abs(rows - self.target_pos_row - 1) <= 1 )
		return mask_main

	def do_photometry(self):
		"""Perform photometry on the given target.

		This function needs to set
			* self.lightcurve
		"""

		logger = logging.getLogger(__name__)
		logger.info("Running aperture photometry...")

		k2p2_settings = {
			'thresh': 0.8,
			'min_no_pixels_in_mask': 4,
			'min_for_cluster': 4,
			'cluster_radius': np.sqrt(2) + np.finfo(np.float64).eps,
			'segmentation': True,
			'ws_blur': 0.5,
			'ws_thres': 0,
			'ws_footprint': 3,
			'extend_overflow': True
		}

		for retries in range(5):
			# Delete any plots left over in the plots folder from an earlier iteration:
			self.delete_plots()

			# Create the sum-image:
			SumImage = self.sumimage

			logger.info(self.stamp)
			logger.info("Target position in stamp: (%f, %f)", self.target_pos_row_stamp, self.target_pos_column_stamp )

			cat = np.column_stack((self.catalog['column_stamp'], self.catalog['row_stamp'], self.catalog['tmag']))

			logger.info("Creating new masks...")
			try:
				masks, background_bandwidth = k2p2.k2p2FixFromSum(SumImage, plot_folder=self.plot_folder, show_plot=False, catalog=cat, **k2p2_settings)
				masks = np.asarray(masks, dtype='bool')
			except k2p2.K2P2NoStars:
				self.report_details(error='No flux above threshold.')
				masks = np.asarray(0, dtype='bool')

			using_minimum_mask = False
			if len(masks.shape) == 0:
				logger.warning("No masks found")
				self.report_details(error='No masks found. Using minimum aperture.')
				mask_main = self._minimum_aperture()
				using_minimum_mask = True

			else:
				# Look at the central pixel where the target should be:
				indx_main = masks[:, int(round(self.target_pos_row_stamp)), int(round(self.target_pos_column_stamp))].flatten()

				if not np.any(indx_main):
					logger.warning('No mask found for main target. Using minimum aperture.')
					self.report_details(error='No mask found for main target. Using minimum aperture.')
					mask_main = self._minimum_aperture()
					using_minimum_mask = True

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
		for k, (img, imgerr, bck) in enumerate(zip(self.images, self.images_err, self.backgrounds)):

			flux_in_cluster = img[mask_main]

			# Calculate flux in mask:
			if allnan(flux_in_cluster) or np.all(flux_in_cluster == 0):
				self.lightcurve['flux'][k] = np.NaN
				self.lightcurve['flux_err'][k] = np.NaN
				self.lightcurve['pos_centroid'][k, :] = np.NaN
				#self.lightcurve['quality']
			else:
				self.lightcurve['flux'][k] = np.sum(flux_in_cluster)
				self.lightcurve['flux_err'][k] = np.sqrt(np.sum(imgerr[mask_main]**2))

				# Calculate flux centroid:
				finite_vals = (flux_in_cluster > 0)
				if np.any(finite_vals):
					self.lightcurve['pos_centroid'][k, :] = np.average(members[finite_vals], weights=flux_in_cluster[finite_vals], axis=0)
				else:
					self.lightcurve['pos_centroid'][k, :] = np.NaN

			if allnan(bck[mask_main]):
				self.lightcurve['flux_background'][k] = np.NaN
			else:
				self.lightcurve['flux_background'][k] = np.nansum(bck[mask_main])

		# Save the mask to be stored in the outout file:
		self.final_mask = mask_main

		# Add additional headers specific to this method:
		#self.additional_headers['KP_SUBKG'] = (bool(subtract_background), 'K2P2 subtract background?')
		self.additional_headers['KP_THRES'] = (k2p2_settings['thresh'], 'K2P2 sum-image threshold')
		self.additional_headers['KP_MIPIX'] = (k2p2_settings['min_no_pixels_in_mask'], 'K2P2 min pixels in mask')
		self.additional_headers['KP_MICLS'] = (k2p2_settings['min_for_cluster'], 'K2P2 min pix. for cluster')
		self.additional_headers['KP_CLSRA'] = (k2p2_settings['cluster_radius'], 'K2P2 cluster radius')
		self.additional_headers['KP_WS'] = (bool(k2p2_settings['segmentation']), 'K2P2 watershed segmentation')
		#self.additional_headers['KP_WSALG'] = (k2p2_settings['ws_alg'], 'K2P2 watershed weighting')
		self.additional_headers['KP_WSBLR'] = (k2p2_settings['ws_blur'], 'K2P2 watershed blur')
		self.additional_headers['KP_WSTHR'] = (k2p2_settings['ws_thres'], 'K2P2 watershed threshold')
		self.additional_headers['KP_WSFOT'] = (k2p2_settings['ws_footprint'], 'K2P2 watershed footprint')
		self.additional_headers['KP_EX'] = (bool(k2p2_settings['extend_overflow']), 'K2P2 extend overflow')

		# Targets that are in the mask:
		target_in_mask = [k for k,t in enumerate(self.catalog) if np.round(t['row'])+1 in rows[mask_main] and np.round(t['column'])+1 in cols[mask_main]]

		# Figure out which status to report back:
		my_status = STATUS.OK

		# Calculate contamination from the other targets in the mask:
		if len(target_in_mask) == 0:
			logger.error("No targets in mask")
			self.report_details(error='No targets in mask')
			contamination = np.nan
			my_status = STATUS.ERROR
		elif len(target_in_mask) == 1 and self.catalog[target_in_mask][0]['starid'] == self.starid:
			contamination = 0
		else:
			# Calculate contamination metric as defined in Lund & Handberg (2014):
			mags_in_mask = self.catalog[target_in_mask]['tmag']
			mags_total = -2.5*np.log10(np.nansum(10**(-0.4*mags_in_mask)))
			contamination = 1.0 - 10**(0.4*(mags_total - self.target_tmag))
			contamination = np.abs(contamination) # Avoid stupid signs due to round-off errors

		logger.info("Contamination: %f", contamination)
		if not np.isnan(contamination):
			self.additional_headers['AP_CONT'] = (contamination, 'AP contamination')

		# Check if there are other targets in the mask that could then be skipped from
		# processing, and report this back to the TaskManager. The TaskManager will decide
		# if this means that this target or the other targets should be skipped in the end.
		skip_targets = [t['starid'] for t in self.catalog[target_in_mask] if t['starid'] != self.starid]
		if skip_targets:
			logger.info("These stars could be skipped: %s", skip_targets)
			self.report_details(skip_targets=skip_targets)

		# Figure out which status to report back:
		if using_minimum_mask:
			my_status = STATUS.WARNING

		# Return whether you think it went well:
		return my_status
