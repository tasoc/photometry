#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 17:52:09 2017

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
from six.moves import range, zip
import os
import numpy as np
import logging
from astropy.table import Table
from time import clock
import matplotlib.pyplot as plt
from photometry import BasePhotometry, AperturePhotometry

#------------------------------------------------------------------------------
if __name__ == '__main__':

	logging_level = logging.WARNING

	input_folder = r'C:\Users\au195407\Documents\tess_data\input'
	output_folder = 'C:\Users\au195407\Documents\tess_data\output'

	# Setup logging:
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	console = logging.StreamHandler()
	console.setFormatter(formatter)
	logger = logging.getLogger(__name__)
	logger.addHandler(console)
	logger.setLevel(logging_level)
	for logger_name in ('photometry', 'photometry.AperturePhotometry.photometry'):
		logger_parent = logging.getLogger(logger_name)
		logger_parent.addHandler(console)
		logger_parent.setLevel(logging_level)

	cat = np.genfromtxt(os.path.join(input_folder, 'catalog.txt.gz'), skip_header=1, usecols=(4,5,6), dtype='float64')
	cat = np.column_stack((np.arange(1, cat.shape[0]+1, dtype='int64'), cat))
	catalog = Table(cat,
		names=('starid', 'x', 'y', 'tmag'),
		dtype=('int64', 'float32', 'float32', 'float32')
	)

	indx = (catalog['x'] > 0) & (catalog['x'] < 2048) & (catalog['y'] > 0) & (catalog['y'] < 2048)
	catalog = catalog[indx]
	catalog.sort('tmag')

	Ntests = 100

	position_errors = np.zeros((Ntests, 2), dtype='float64') + np.nan
	for k, thisone in enumerate(catalog[:Ntests]):
		starid = thisone['starid']
		print(k, starid)

		with AperturePhotometry(starid, input_folder) as pho:
			try:
				status = pho.do_photometry()
			except (KeyboardInterrupt, SystemExit):
				break
			except:
				status = BasePhotometry.STATUS_ERROR
				logger.exception("Something happened")

			if status == BasePhotometry.STATUS_OK:

				#pho.save_lightcurve(output_folder)

				extracted_pos = np.median(pho.lightcurve['pos_centroid'], axis=0)
				real_pos = np.array([thisone['x'], thisone['y']])

				position_errors[k,:] = real_pos - extracted_pos

	fig = plt.figure()
	plt.scatter(position_errors[:,0], position_errors[:,1])
	fig.savefig('position_errors.png')
	plt.show()
