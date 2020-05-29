#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Corrections of time offset present in early data releases of TESS Data (sectors 1-21).
This involved the following corrections to the timestamps:

Staggered readouts of the four cameras:
	The two-second integrations in the cameras are offset by 0.5 seconds,
	in the order camera 1, camera 3, camera 4, camera 2. This applies to FFIs only.

Staggered readouts of the four CCDs within a camera:
	The readouts of the four CCDs are staggered by 0.020 seconds,
	in the order CCD 1, CCD 2, CCD 3, CCD 4. This applies to FFIs only.

Off-by-one error in cadence counting:
	A correction to an error in calculation the start and end times of 2m and 30m data: these
	values were too high by 2.0 seconds in the original data products.

Detailed correction to start and end times:
	The start times of integrations for every 2 minute and 30 minute cadence were shifted
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
to be re-run and HDF5 file re-created.

.. seealso::

	Sector 18 data release notes 25, section 3.3:
		https://tasoc.dk/docs/release_notes/tess_sector_18_drn25_v02.pdf

	Memos on revisions of TESS data releases 27 and 29
		https://tasoc.dk/docs/release_notes/tess_s20_dr27_data_product_revision_memo_v02.pdf
		https://tasoc.dk/docs/release_notes/tess_s21_dr29_data_product_revision_memo_v02.pdf

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import logging

#--------------------------------------------------------------------------------------------------
def time_offset(time, header, datatype='ffi', timepos='mid', return_flag=False):
	"""
	Apply time offset correction to array of timestamps.

	Parameters:
		time (ndarray): Array of timestamps in days.
		header (dict): Header from TPF, FFI or HDF5 file.
		datatype (str, optional): Data product to correct. Choices are ``'ffi'`` or ``'tpf'``.
			Default is ``'ffi'``.
		timepos (str, optional): At what time during exposure are times indicating?
			Choices are ``'mid'``, ``'start'`` and ``'end'``. Default is ``'mid'``.
		return_flag (bool, optional): Also return the flag indication whether the
			timestamps were corrected.

	Returns:
		tuple:
		- ndarray: Corrected timestamps in days.
		- bool: True if corrections to timestamps were needed, false otherwise.
			Only returned if ``return_flag`` parameter is set.

	Raises:
		ValueError: If invalid timepos.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)

	datarel = int(header['DATA_REL'])
	procver = header.get('PROCVER', None)
	already_corrected = bool(header.get('TIME_OFFSET_CORRECTED', False))

	if timepos not in ('start', 'mid', 'end'):
		raise ValueError("Invalid TIMEPOS")

	datarel27_first_release = False
	# If correction already applied or for later data
	# releases no correction should be applied:
	if already_corrected or datarel > 29:
		apply_correction = False

	# For these early data releases, a correction is needed:
	elif datarel <= 26:
		apply_correction = True

	# For these troublesome sectors, there were two data releases with the same
	# data release numbers. The only way of distinguishing between them is to use
	# the PROCVER header keyword, which unfortunately wasn't saved in the HDF5
	# files in earlier versions of the pipeline. In that case, we have to throw
	# an error and tell users to re-run prepare:
	elif datarel in (27, 29) and procver is None:
		raise Exception("""The timestamps of these data may need to be corrected,
			but the PROCVER header is not present. HDF5 files may need to be re-created.""")

	elif datarel == 27 \
		and procver in ('spoc-4.0.14-20200108', 'spoc-4.0.15-20200114', 'spoc-4.0.17-20200130'):
		datarel27_first_release = True
		apply_correction = True

	elif datarel == 29 \
		and procver in ('spoc-4.0.17-20200130', 'spoc-4.0.20-20200220', 'spoc-4.0.21-20200227'):
		apply_correction = True

	else:
		apply_correction = False

	if apply_correction:
		logger.debug("Fixes: Applying time offset correction")

		# Early releases of FFIs (sectors 1-20, DR 1-27) suffer from
		# differences between timestamps depending on camera:
		staggered_readout = 0
		if datatype == 'ffi' and (datarel <= 26 or datarel27_first_release):
			camera = int(header['CAMERA'])
			ccd = int(header['CCD'])

			staggered_readout = {
				1: 0.000,
				2: 1.500,
				3: 0.500,
				4: 1.000
			}[camera]

			staggered_readout += {
				1: 0.000,
				2: 0.020,
				3: 0.040,
				4: 0.060
			}[ccd]

		if timepos == 'mid':
			time = time + (staggered_readout - 2.000 + 0.021) / 86400

		elif timepos == 'start':
			time = time + (staggered_readout - 2.000 + 0.031) / 86400

		elif timepos == 'end':
			time = time + (staggered_readout - 2.000 + 0.011) / 86400

	else:
		logger.debug("Fixes: Not applying time offset correction")

	if return_flag:
		return time, apply_correction
	else:
		return time
