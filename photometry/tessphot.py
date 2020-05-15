#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import logging
import traceback
from . import STATUS, AperturePhotometry, PSFPhotometry, LinPSFPhotometry, HaloPhotometry
from .utilities import mag2flux

#--------------------------------------------------------------------------------------------------
class _PhotErrorDummy(object):
	def __init__(self, traceback, *args, **kwargs):
		self.status = STATUS.ERROR
		self.method = 'error'
		self._details = {'errors': traceback} if traceback else {}

#--------------------------------------------------------------------------------------------------
def _try_photometry(PhotClass, *args, **kwargs):
	logger = logging.getLogger(__name__)
	tbcollect = []
	try:
		with PhotClass(*args, **kwargs) as pho:
			pho.photometry()

			if pho.status in (STATUS.OK, STATUS.WARNING):
				pho.save_lightcurve()

	except (KeyboardInterrupt, SystemExit):
		logger.info("Stopped by user or system")
		try:
			pho._status = STATUS.ABORT
		except: # noqa: E722
			pass

	except: # noqa: E722
		logger.exception("Something happened")
		tb = traceback.format_exc().strip()
		try:
			pho._status = STATUS.ERROR
			pho.report_details(error=tb)
		except: # noqa: E722
			tbcollect.append(tb)

	try:
		return pho
	except UnboundLocalError:
		return _PhotErrorDummy(tbcollect, *args, **kwargs)

#--------------------------------------------------------------------------------------------------
def tessphot(method=None, *args, **kwargs):
	"""
	Run the photometry pipeline on a single star.

	This function will run the specified photometry or perform the dynamical
	scheme of trying simple aperture photometry, evaluating its performance
	and if nessacery try another algorithm.

	Parameters:
		method (string or None): Type of photometry to run.
			Can be ``'aperture'``, ``'halo'``, ``'psf'``, ``'linpsf'`` or ``None``.
		*args: Arguments passed on to the photometry class init-function.
		**kwargs: Keyword-arguments passed on to the photometry class init-function.

	Raises:
		ValueError: On invalid method.

	Returns:
		:py:class:`photometry.BasePhotometry`: Photometry object that inherits
			from :py:class:`photometry.BasePhotometry`.
	"""

	logger = logging.getLogger(__name__)

	if method is None:
		# Start out by trying simple aperture photometry:
		pho = _try_photometry(AperturePhotometry, *args, **kwargs)

		# If this is a bright target, and there are several pixels touching
		# the edge, let's switch to Halo photometry instead:
		if not isinstance(pho, _PhotErrorDummy):

			tmag_limit = 6.0 # Maximal Tmag to apply Halo photometry automatically
			flux_limit = 0.01
			EdgeFlux = pho._details.get('edge_flux')

			if pho.target['tmag'] <= tmag_limit and EdgeFlux is not None:
				ExpectedFlux = mag2flux(pho.target['tmag'])
				if EdgeFlux/ExpectedFlux > flux_limit:
					# There is significant flux on the edge, Halo should do a better job:
					logger.warning("Target is still touching the edge. Let us try Halo instead.")
					pho = _try_photometry(HaloPhotometry, *args, **kwargs)

					# Make sure to flag that we did automatic switching to Halo photometry,
					# and keep the edge_flux diagnostics from before, to keep the diagnostics
					# that led to the automatic switch:
					if not isinstance(pho, _PhotErrorDummy):
						pho.report_details('Automatically switched to Halo photometry')
						pho._details['edge_flux'] = EdgeFlux

		# TODO: If too crowded:
		# pho = _try_photometry(PSFPhotometry, starid, input_folder, output_folder, datasource, plot)

		# TODO: Still getting warning status. Maybe we should do something else?
		if pho.status == STATUS.WARNING:
			logger.warning("Do something else?")

	elif method == 'aperture':
		pho = _try_photometry(AperturePhotometry, *args, **kwargs)

	elif method == 'psf':
		pho = _try_photometry(PSFPhotometry, *args, **kwargs)

	elif method == 'linpsf':
		pho = _try_photometry(LinPSFPhotometry, *args, **kwargs)

	elif method == 'halo':
		pho = _try_photometry(HaloPhotometry, *args, **kwargs)

	else:
		raise ValueError("Invalid method: '{0}'".format(method))

	logger.info("Done")
	return pho
