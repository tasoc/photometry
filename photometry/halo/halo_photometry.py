#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Halo Photometry.

.. codeauthor:: Tim White <white@phys.au.dk>
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
from six.moves import range, zip
import numpy as np
import logging
from .. import BasePhotometry, STATUS

#------------------------------------------------------------------------------
class HaloPhotometry(BasePhotometry):
	"""Simple Aperture Photometry using K2P2 to define masks.

	.. codeauthor:: Tim White <white@phys.au.dk>
	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	def __init__(self, *args, **kwargs):
		# Call the parent initializing:
		# This will set several default settings
		super(self.__class__, self).__init__(*args, **kwargs)

		# Here you could do other things that needs doing in the beginning
		# of the run on each target.


	def do_photometry(self):
		"""Perform photometry on the given target.

		This function needs to set
			* self.lightcurve
		"""

		# Start logger to use for print 
		logger = logging.getLogger(__name__)

		logger.info("starid: %d", self.starid)
		
		logger.info("Target position in stamp: (%f, %f)", self.target_pos_row_stamp, self.target_pos_column_stamp )

		# Get pixel grid:
		cols, rows = self.get_pixel_grid()
		
		# Loop through the images and backgrounds together:
		for k, (img, bck) in enumerate(zip(self.images, self.backgrounds)):
			# Fill out these things based on "img" and "bck"
			# self.lightcurve['flux'][k]
			# self.lightcurve['quality'][k]
			# self.lightcurve['pos_centroid'][k]
			pass
	
		# If something went seriouly wrong:
		#self.report_details(error='What the hell?!')
		#return STATUS.ERROR

		# If something is a bit strange:
		#self.report_details(error='')
		#return STATUS.WARNING
		
		# Save the mask to be stored in the outout file:
		#self.final_mask = mask

		# Add additional headers specific to this method:
		#self.additional_headers['HDR_KEY'] = (value, 'comment')

		# If some stars could be skipped:
		#self.report_details(skip_targets=skip_targets)

		# Return whether you think it went well:
		return STATUS.OK
