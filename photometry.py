#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 17:52:09 2017

@author: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
from six.moves import range, zip
import numpy as np
import logging
from astropy.table import Table
import os.path
import sys
from BasePhotometry import BasePhotometry, PhotometryStatus
if not '/usr/users/kasoc/Preprocessing/' in sys.path:
	sys.path.append('/usr/users/kasoc/Preprocessing/')
import k2p2v2 as k2p2
from time import clock
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
class K2P2Photometry(BasePhotometry):
	"""Simple Aperture Photometry using K2P2 to define masks."""

	def __init__(self, starid):
		# Call the parent initializing:
		# This will set several default settings
		super(self.__class__, self).__init__(starid)

		# Here you could do other things that needs doing in the beginning
		# of the run on each target.
	
	#def default_stamp(self):
	#	Nrows = 50
	#	Ncolumns = 30
	#	return Nrows, Ncolumns

	def do_photometry(self):
		"""Perform photometry on the given target.

		This function needs to set
			* self.flux
			* self.flux_background
			* self.quality
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
						
			#plt.figure()
			#plt.imshow(np.log10(self.sumimage), origin='lower')
			#plt.scatter(catalog['row']+0.5, catalog['column']+0.5, s=100/catalog['tmag'], c='r')
			
			cat = np.column_stack((catalog['row'], catalog['column'], catalog['tmag']))
			
			logger.info("Creating new masks...")
			masks, background_bandwidth = k2p2.k2p2FixFromSum(SumImage, None, plot_folder=None, thresh=5, min_no_pixels_in_mask=4, catalog=cat)
			masks = np.asarray(masks, dtype='bool')
			
			if len(masks.shape) == 0:
				logger.error("No masks found")
				return PhotometryStatus.ERROR
			
			# Look at the central pixel where the target should be:
			indx_main = masks[:, target_pixel_row, target_pixel_column].flatten()
	
			if not np.any(indx_main):
				logger.error('No pixels')
				return PhotometryStatus.ERROR
			elif np.sum(indx_main) > 1:
				logger.error('Too many masks')
				return PhotometryStatus.ERROR
			
			# Mask of the main target:
			mask_main = masks[indx_main,:,:].reshape(SumImage.shape)
		
			resize_args = {}
			if np.any(mask_main[0,:]):
				resize_args['down'] = 10
			if np.any(mask_main[-1,:]):
				resize_args['up'] = 10
			if np.any(mask_main[:,0]):
				resize_args['left'] = 10
			if np.any(mask_main[:,-1]):
				resize_args['right'] = 10
			
			logger.info(resize_args)
		
			if resize_args:
				logger.error("Touching the edges! Retrying")
				self.resize_stamp(**resize_args)
				logger.info('-'*70)
			else:
				break
	
		# XY of pixels in frame
		Y, X = self.get_pixel_grid()
		members = np.column_stack((X[mask_main], Y[mask_main]))

		#
		for k, (img, bck) in enumerate(zip(self.images, self.backgrounds)):

			flux_in_cluster = img[mask_main] - bck[mask_main]

			# Calculate flux in mask:
			self.flux[k] = np.sum(flux_in_cluster)
			self.flux_background[k] = np.sum(bck[mask_main])

			# Calculate flux centroid:
			finite_vals = (flux_in_cluster > 0)
			self.pos_centroid[k, :] = np.average(members[finite_vals, :], weights=flux_in_cluster[finite_vals], axis=0)

		#
		self.final_mask = mask_main

		# Return whether you think it went well:
		return PhotometryStatus.OK

#------------------------------------------------------------------------------
if __name__ == '__main__':

	logging_level = logging.WARNING

	# Setup logging:
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	console = logging.StreamHandler()
	console.setFormatter(formatter)
	logger = logging.getLogger(__name__)
	logger.addHandler(console)
	logger.setLevel(logging_level)
	logger_parent = logging.getLogger('BasePhotometry')
	logger_parent.addHandler(console)
	logger_parent.setLevel(logging_level)
	
	cat = np.genfromtxt(r'input/catalog.txt.gz', skip_header=1, usecols=(4,5,6), dtype='float64')
	cat = np.column_stack((np.arange(1, cat.shape[0]+1, dtype='int64'), cat))
	catalog = Table(cat,
		names=('starid', 'x', 'y', 'tmag'),
		dtype=('int64', 'float32', 'float32', 'float32')
	)

	indx = (catalog['x'] > 0) & (catalog['x'] < 2048) & (catalog['y'] > 0) & (catalog['y'] < 2048)
	catalog = catalog[indx]
	catalog.sort('tmag')
	
	Ntests = 1000
	
	position_errors = np.zeros((Ntests, 2), dtype='float64') + np.nan
	for k, thisone in enumerate(catalog[:Ntests]):
		starid = thisone['starid']
		print(k, starid)
	
		with K2P2Photometry(starid) as pho:
			try:
				status = pho.do_photometry()
			except (KeyboardInterrupt, SystemExit):
				break
			except:
				status = PhotometryStatus.ERROR
				logger.error("Something happened")
			
			if status == PhotometryStatus.OK:
				#pho.save_lightcurve()
				
				extracted_pos = np.median(pho.pos_centroid, axis=0)
				real_pos = np.array([thisone['x'], thisone['y']])
							
				position_errors[k,:] = real_pos - extracted_pos
				
	fig = plt.figure()
	plt.scatter(position_errors[:,0], position_errors[:,1])
	fig.savefig('position_errors.png')
	plt.show()
