#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download any missing data files to cache.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import logging
from astropy.time import Time
from astropy.utils import iers
from .spice import TESS_SPICE

#--------------------------------------------------------------------------------------------------
def download_cache(testing=False):
	"""
	Download any missing data files to cache.

	This will download all auxillary files used by astropy or our code itself
	to the cache. If all the nessacery files already exists, nothing will be done.
	It can be a good idea to call this function before starting the photometry
	in parallel on many machines sharing the same cache, in which case the processes
	will all attempt to download the cache files and may conflict with each other.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)

	# This will download IERS data needed for astropy.Time transformations:
	# https://docs.astropy.org/en/stable/utils/iers.html
	# Ensure that the auto_download config is enabled, otherwise nothing will be downloaded
	logger.info("Downloading IERS data...")
	oldval = iers.conf.auto_download
	try:
		iers.conf.auto_download = True
		iers.IERS_Auto().open()
	finally:
		iers.conf.auto_download = oldval

	# The TESS SPICE kernels should be downloaded, if they are not already.
	# We also make sure to unload any loaded kernels again,
	# to ensure that this function has zero effect.
	logger.info("Downloading SPICE kernels...")
	if testing:
		# When downloading for testing only, add the two time-intervals
		# corresponding to sectors 1 and 27, which are the ones needed
		# for the test-cases.
		intvs = [
			Time([1325.30104564163, 1326.68855796131], 2457000, format='jd', scale='tdb'), # Sector 1
			Time([2036.274561493, 2037.66210106632], 2457000, format='jd', scale='tdb') # Sector 27
		]
		for intv in intvs:
			with TESS_SPICE(intv=intv, download=True) as tsp:
				tsp.unload()
	else:
		with TESS_SPICE(download=True) as tsp:
			tsp.unload()

	logger.info("All cache data downloaded.")
