#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Aperture Photometry using K2P2 to define masks.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
from bottleneck import allnan
import logging
from .. import BasePhotometry, STATUS
from ..utilities import mag2flux
from . import k2p2v2 as k2p2

#--------------------------------------------------------------------------------------------------
class AperturePhotometry(BasePhotometry):
	"""
	Simple Aperture Photometry using K2P2 to define masks.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	#----------------------------------------------------------------------------------------------
	def __init__(self, *args, **kwargs):
		# Call the parent initializing:
		# This will set several default settings
		super().__init__(*args, **kwargs)

	#----------------------------------------------------------------------------------------------
	def _minimum_aperture(self):
		cols, rows = self.get_pixel_grid()
		mask_main = ( np.abs(cols - self.target_pos_column - 1) <= 1 ) \
			& ( np.abs(rows - self.target_pos_row - 1) <= 1 )
		return mask_main

	#----------------------------------------------------------------------------------------------
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

		# For bright saturated stars we allow for more retries:
		ExpectedFlux = mag2flux(self.target['tmag'])
		haloswitch_tmag_limit = self.settings.getfloat('haloswitch', 'tmag_limit')
		haloswitch_flux_limit = self.settings.getfloat('haloswitch', 'flux_limit')

		allow_retries = 5
		if self.target['tmag'] < 6:
			allow_retries = 10

		for retries in range(allow_retries):
			# Delete any plots left over in the plots folder from an earlier iteration:
			self.delete_plots()

			# Create the sum-image:
			SumImage = self.sumimage

			logger.info(self.stamp)
			logger.info("Target position in stamp: (%f, %f)",
				self.target_pos_row_stamp, self.target_pos_column_stamp )

			cat = np.column_stack((
				self.catalog['column_stamp'],
				self.catalog['row_stamp'],
				self.catalog['tmag']))

			logger.info("Creating new masks...")
			try:
				masks, background_bandwidth = k2p2.k2p2FixFromSum(SumImage, plot_folder=self.plot_folder, show_plot=False, catalog=cat, **k2p2_settings)
				masks = np.asarray(masks, dtype='bool')
			except k2p2.K2P2NoStars:
				logger.error('No flux above threshold.')
				masks = np.asarray(0, dtype='bool')

			using_minimum_mask = False
			if len(masks.shape) == 0:
				logger.warning("No masks found. Using minimum aperture.")
				mask_main = self._minimum_aperture()
				using_minimum_mask = True

			else:
				# Look at the central pixel where the target should be:
				indx_main = masks[:, int(round(self.target_pos_row_stamp)), int(round(self.target_pos_column_stamp))].flatten()

				if not np.any(indx_main):
					logger.warning('No mask found for main target. Using minimum aperture.')
					mask_main = self._minimum_aperture()
					using_minimum_mask = True

				elif np.sum(indx_main) > 1:
					logger.error('Too many masks.')
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
				logger.warning("Touching the edges! Retrying.")
				logger.info(resize_args)
				stamp_before = self.stamp
				sumimage_before = self.sumimage
				if not self.resize_stamp(**resize_args):
					resize_args = {}
					logger.warning("Could not resize stamp any further.")
					break

				# It did resize, but let's just check if it tried
				# to resize in a direction, but it hit the limit.
				# In that case, let's check if we are already over the "HaloSwitch" limit
				# Don't do this for secondary targets though.
				if self.target['tmag'] <= haloswitch_tmag_limit and not self.datasource.startswith('tpf:'):
					edge = np.zeros_like(mask_main, dtype='bool')
					if resize_args.get('down') and self.stamp[0] == stamp_before[0]:
						edge[0, :] = True
					if resize_args.get('up') and self.stamp[1] == stamp_before[1]:
						edge[-1, :] = True
					if resize_args.get('left') and self.stamp[2] == stamp_before[2]:
						edge[:, 0] = True
					if resize_args.get('right') and self.stamp[3] == stamp_before[3]:
						edge[:, -1] = True

					if np.any(edge):
						EdgeFlux = np.nansum(sumimage_before[mask_main & edge])
						if EdgeFlux/ExpectedFlux > haloswitch_flux_limit:
							logger.error('Stamp resize hit limit. Haloswitch quick break.')
							self._details['edge_flux'] = EdgeFlux
							return STATUS.ERROR
			else:
				break

		# If we reached the last retry but still needed a resize, give up:
		if resize_args:
			logger.error('Too many stamp resizes.')
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
				#self.lightcurve['quality'][k] |= ?
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
		self.final_phot_mask = mask_main
		self.final_position_mask = mask_main

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
		target_in_mask = [k for k,t in enumerate(self.catalog) if np.any(mask_main & (rows == np.round(t['row'])+1) & (cols == np.round(t['column'])+1))]

		# Figure out which status to report back:
		my_status = STATUS.OK

		# Calculate contamination from the other targets in the mask:
		if len(target_in_mask) == 0:
			logger.error("No targets in mask.")
			contamination = np.nan
			my_status = STATUS.ERROR
		elif len(target_in_mask) == 1 and self.catalog[target_in_mask][0]['starid'] == self.starid:
			contamination = 0
		else:
			# Calculate contamination metric as defined in Lund & Handberg (2014):
			mags_in_mask = self.catalog[target_in_mask]['tmag']
			mags_total = -2.5*np.log10(np.nansum(10**(-0.4*mags_in_mask)))
			contamination = 1.0 - 10**(0.4*(mags_total - self.target['tmag']))
			contamination = np.clip(contamination, 0, None) # Avoid stupid signs due to round-off errors

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
