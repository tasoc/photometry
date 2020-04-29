#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Corrections of time offset present in early data releases of TESS Data (sectors 1-21).
"""

#---------------------------------------------------------------------------------------------------
def time_offset_should_be_fixed(sector, datarel, procver):
	"""

	Parameters:
		sector (integer): TESS Sector.
		datarel (integer): TESS Data Release number.
		procver (string):

	Returns:
		bool: Returns True if corrections to timestamps are needed, false otherwise.
	"""

	if sector <= 19 and datarel <= 26:
		return True

	elif sector == 20 and datarel == 27 \
		and procver in ('spoc-4.0.14-20200108', 'spoc-4.0.15-20200114', 'spoc-4.0.17-20200130'):
		return True

	elif sector == 21 and datarel == 29 \
		and procver in ('spoc-4.0.17-20200130', 'spoc-4.0.20-20200220', 'spoc-4.0.21-20200227'):
		return True

	return False

#---------------------------------------------------------------------------------------------------
def time_offset_apply(time, timepos='mid'):
	"""
	Apply time offset correction to array of timestamps.

	Parameters:
		time (ndarray): Array of timestamps in days.
		timepos (string, optional):

	Returns:
		ndarray: Returns True if corrections to timestamps are needed, false otherwise.

	Raises:
		ValueError: If invalid timepos.
	"""

	if timepos == 'mid':
		return time - (2.000 - 0.021) / 86400
	elif timepos == 'start':
		return time - (2.000 - 0.031) / 86400
	elif timepos == 'end':
		return time - (2.000 - 0.011) / 86400

	raise ValueError("")
