#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Corrections of time offset present in early data releases of TESS Data (sectors 1-21).

* A correction to an error in calculation the start and end times of 2m and 30m data: these
  values were too high by 2.0 seconds in the original data products.

* The start times of integrations for every 2 minute and 30 minute cadence were shifted
  forward by 31 milliseconds, and the end times were shifted forward by 11 milliseconds.
  These offsets correct for effects in the focal plane electronics.

Assuming :math:`S`, :math:`M` and :math:`E` are the original start, mid and end-timestamps
respectively and :math:`S'`, :math:`M'` and :math:`E'` are the new (corrected) timestamps:

.. math::

	S' &= S - 2.000 + 0.031 = S - 1.969

	E' &= S' + 1.980

The end time used to be the start time + 2.000 seconds:

.. math::

	E = S + 2.000

Doing the math we find :math:`M' = (S'+E')/2 = (S+E)/2 - 1.979` and :math:`E'=E - 1.989`.
So the mid-time shifts by (2 - 0.021) seconds, which is the 11ms offset before readout
plus half of the 20ms readout time.

The correction should definitely be applied to all data releases <= 26, and possibly for
data releases 27 and 29 (sectors 20 and 21), depending on the headers in the files.
For the troublesome sectors 20 and 21, there were two data releases with the same
data release numbers. The only way of distinguishing between them is to use
the PROCVER header keyword, which unfortunately wasn't saved in the HDF5
files in earlier versions of the pipeline. In that case, the "prepare" stage will have
to be re-run and HDF5 file re-created..

.. seealso::

	Memos on revisions of TESS data releases 27 and 29
		https://tasoc.dk/docs/release_notes/tess_s20_dr27_data_product_revision_memo_v01.pdf
		https://tasoc.dk/docs/release_notes/tess_s21_dr29_data_product_revision_memo_v01.pdf

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

#--------------------------------------------------------------------------------------------------
def time_offset_should_be_fixed(header=None, datarel=None, procver=None):
	"""
	Should time offset correction be applied to this data?

	Parameters:
		header (dict, optional): Header from TPF, FFI or HDF5 file.
		datarel (int, optional): TESS Data Release number.
		procver (str, optional): PROCVER header value. Indicates processing pipeline version.

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
		raise Exception("""The timestamps of these data may need to be corrected,
			but the PROCVER header is not present. HDF5 files may need to be re-created.""")

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
		timepos (str, optional): At what time during exposure are times indicating?
			Choices are ``'mid'``, ``'start'`` and ``'end'``. Default is ``'mid'``.

	Returns:
		ndarray: Corrected timestamps in days.

	Raises:
		ValueError: If invalid timepos.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	if timepos == 'mid':
		return time - 1.979 / 86400 # 2.000 - 0.021
	elif timepos == 'start':
		return time - 1.969 / 86400 # 2.000 - 0.031
	elif timepos == 'end':
		return time - 1.989 / 86400 # 2.000 - 0.011

	raise ValueError("Invalid TIMEPOS")
