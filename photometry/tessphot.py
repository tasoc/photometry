#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, print_function, with_statement, absolute_import
import logging
from . import STATUS, AperturePhotometry

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
			pho.photometry()
			status = pho.status
		except (KeyboardInterrupt, SystemExit):
			status = STATUS.ABORT
			logger.info("Stopped by user or system")
		except:
			logger.exception("Something happened")
			status = STATUS.ERROR
			try:
				pho._status = STATUS.ERROR
			except:
				pass

		if status == STATUS.OK:
			pho.save_lightcurve(output_folder)

	if status == STATUS.WARNING:
		logger.warning("Try something else?")

	logger.info("Done")
	return pho
