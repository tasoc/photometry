#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, print_function, with_statement, absolute_import
import logging
from . import BasePhotometry, AperturePhotometry

#------------------------------------------------------------------------------
def tessphot(starid=None, method='aperture', input_folder=None, output_folder=None):
	"""
	Run the photometry pipeline on a single star.

	This function will run the specified photometry or perform the dynamical
	scheme of trying simple aperture photometry, evaluating its performance
	and if nessacery try another algorithm.
	"""

	logger = logging.getLogger(__name__)

	with AperturePhotometry(starid, input_folder) as pho:
		try:
			status = pho.photometry()
		except (KeyboardInterrupt, SystemExit):
			status = BasePhotometry.STATUS_ABORT
			logger.info("Stopped by user or system")
		except:
			status = BasePhotometry.STATUS_ERROR
			logger.exception("Something happened")

		if status == BasePhotometry.STATUS_OK:
			pho.save_lightcurve(output_folder)

	if status == BasePhotometry.STATUS_WARNING:
		logger.warning("Try something else?")

	logger.info("Done")
	return pho
