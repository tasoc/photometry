#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, print_function, with_statement, absolute_import
import logging
from . import STATUS, AperturePhotometry, PSFPhotometry

#------------------------------------------------------------------------------
def _try_photometry(PhotClass, starid, input_folder, output_folder):
	logger = logging.getLogger(__name__)
	with PhotClass(starid, input_folder) as pho:
		try:
			pho.photometry()
			status = pho.status
		except (KeyboardInterrupt, SystemExit):
			status = STATUS.ABORT
			logger.info("Stopped by user or system")
			try:
				pho._status = STATUS.ABORT
			except:
				pass
		except:
			logger.exception("Something happened")
			status = STATUS.ERROR
			try:
				pho._status = STATUS.ERROR
			except:
				pass

		if status == STATUS.OK:
			pho.save_lightcurve(output_folder)

	return pho

#------------------------------------------------------------------------------
def tessphot(starid=None, method=None, input_folder=None, output_folder=None):
	"""
	Run the photometry pipeline on a single star.

	This function will run the specified photometry or perform the dynamical
	scheme of trying simple aperture photometry, evaluating its performance
	and if nessacery try another algorithm.
	"""

	logger = logging.getLogger(__name__)

	if method is None:
		pho = _try_photometry(AperturePhotometry, starid, input_folder, output_folder)

		if pho.status == STATUS.WARNING:
			logger.warning("Try something else?")
			# TODO: If too crowded:
			# pho = _try_photometry(PSFPhotometry, starid, input_folder, output_folder)

	elif method == 'aperture':
		pho = _try_photometry(AperturePhotometry, starid, input_folder, output_folder)

	elif method == 'psf':
		pho = _try_photometry(PSFPhotometry, starid, input_folder, output_folder)

	else:
		raise ValueError("Invalid method: '{0}'".format(method))

	logger.info("Done")
	return pho
