#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Corrections of time offset present in early data releases of TESS Data (sectors 1-21).

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

#--------------------------------------------------------------------------------------------------
def time_offset_should_be_fixed(header=None, datarel=None, procver=None):
	"""
	Should time offset correction be applied to this data?

	Parameters:
		header (dict, optional): Header from TPF, FFI or HDF5 file.
		datarel (int, optional): TESS Data Release number.
		procver (str, optional):

	Returns:
		bool: Returns True if corrections to timestamps are needed, false otherwise.

	Raises:
		ValueError: If wrong combination of input parameters.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	already_corrected = False
	if header is not None:
		datarel = int(header['DATA_REL'])
		procver = header.get('PROCVER', None)
		already_corrected = bool(header.get('TIME_OFFSET_CORRECTED', False))
	elif datarel is None:
		raise ValueError("Either HEADER or DATAREL must be provided.")

	# If correction already applied or for later data
	# releases no correction should be applied:
	if already_corrected or datarel > 29:
		return False

	# For these early data releases, a correction is needed:
	if datarel <= 26:
		return True

	# For these troublesome sectors, there were two data releases with the same
	# data release numbers. The only way of distinguishing between them is to use
	# the PROCVER header keyword, which unfortunately wasn't saved in the HDF5
	# files in earlier versions of the pipeline. In that case, we have to throw
	# an error and tell users to re-run prepare:
	if datarel in (27, 29) and procver is None:
		raise Exception("Nope")

	if datarel == 27 \
		and procver in ('spoc-4.0.14-20200108', 'spoc-4.0.15-20200114', 'spoc-4.0.17-20200130'):
		return True

	elif datarel == 29 \
		and procver in ('spoc-4.0.17-20200130', 'spoc-4.0.20-20200220', 'spoc-4.0.21-20200227'):
		return True

	return False

#--------------------------------------------------------------------------------------------------
def time_offset_apply(time, timepos='mid'):
	"""
	Apply time offset correction to array of timestamps.

	Parameters:
		time (ndarray): Array of timestamps in days.
		timepos (str, optional):

	Returns:
		ndarray: Returns True if corrections to timestamps are needed, false otherwise.

	Raises:
		ValueError: If invalid timepos.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	if timepos == 'mid':
		return time - (2.000 - 0.021) / 86400
	elif timepos == 'start':
		return time - (2.000 - 0.031) / 86400
	elif timepos == 'end':
		return time - (2.000 - 0.011) / 86400

	raise ValueError("Invalid TIMEPOS")
