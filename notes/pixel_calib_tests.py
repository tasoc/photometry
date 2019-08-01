#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import sys
import os.path
import logging
from astropy.io import fits
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from photometry.pixel_calib import PixelCalibrator

#------------------------------------------------------------------------------
if __name__ == '__main__':

	logging_level = logging.INFO

	# Configure the standard console logger
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	console = logging.StreamHandler()
	console.setFormatter(formatter)

	# Configure this logger
	logger = logging.getLogger(__name__)
	logger.addHandler(console)
	logger.setLevel(logging_level)

	logger.info("What are we doing?")
	with PixelCalibrator(tpf=None) as cal:


		print(cal.flatfield)

		tpf_calibrated = cal.calibrate()

	"""
	files = ('test_data/pixel-data.pb.gz', )

	cal.build_collateral_library(files)

	for img_file in files: # TODO: Should be run in parallel
		img = CadenceImage(img_file)
		img_cal = cal.calibrate(img)
		img_cal.seperate_to_targets()
		# TODO: Store the calibrated data somewhere?!

	# TODO: Put all the calibrated data pertaining to one file together in one FITS file
	"""