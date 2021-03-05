#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download any missing data files to cache.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import logging
from astropy.utils import iers
from .spice import TESS_SPICE

#--------------------------------------------------------------------------------------------------
def download_cache():
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
	with TESS_SPICE() as tsp:
		tsp.unload()

	logger.info("All cache data downloaded.")
