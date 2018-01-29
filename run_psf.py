#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 10:46:09 2017

@author: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import logging
import os
from photometry import PSFPhotometry

if __name__ == '__main__':

	logging_level = logging.INFO

	# Setup logging:
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	console = logging.StreamHandler()
	console.setFormatter(formatter)
	logger = logging.getLogger(__name__)
	logger.addHandler(console)
	logger.setLevel(logging_level)
	logger_parent = logging.getLogger('photometry')
	logger_parent.addHandler(console)
	logger_parent.setLevel(logging_level)

	# Get input and output folder from enviroment variables:
	input_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), 'tests', 'input'))
	output_folder = os.environ.get('TESSPHOT_OUTPUT', os.path.abspath('.'))
	logger.info("Loading input data from '%s'", input_folder)
	logger.info("Putting output data in '%s'", output_folder)

	with PSFPhotometry(332979, input_folder) as pho:

		pho.photometry()

		print(pho.lightcurve['time'])

	plt.show()

	logger.info("Done")