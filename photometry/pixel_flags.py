#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import logging
from scipy.ndimage.filters import median_filter

#------------------------------------------------------------------------------
def pixel_manual_exclude(img, hdr):
	"""
	Manual Exclude of individual pixels in Full Frame Images.

	Parameters:
		img (ndarray): Image of which to create manual exclude mask.
		hdr (dict of fits.Header): FITS Header for the images.

	Returns:
		ndarray: Boolean image with the same dimentions as ``img``, containg true
			for pixels that have been marked as manual excludes.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)

	# Create manual exclude mask array:
	mask = np.zeros_like(img, dtype='bool')

	# Check if this is real TESS data:
	# Could proberly be done more elegant, but if it works, it works...
	if hdr.get('TELESCOP') == 'TESS' and hdr.get('NAXIS1') == 2136 and hdr.get('NAXIS2') == 2078:
		is_tess = True
		time = 0.5*(hdr['TSTART'] + hdr['TSTOP'])
		cadenceno = hdr.get('FFIINDEX', np.inf)
	else:
		is_tess = False
		time = np.NaN
		cadenceno = np.inf

	# Mars falls in output channel D of camera 1, CCD 4 in the beginning of
	# TESS Sector 1, which floods the registers and messes up the images
	if is_tess and hdr['CAMERA'] == 1 and hdr['CCD'] == 4 and (cadenceno <= 4724 or hdr['TSTART'] <= 1325.881282301840):
		logger.debug("Manual Exclude: Register overflow due to Mars in FOV")
		mask[:, 1536:] = True

	elif is_tess and hdr['CAMERA'] == 1 and (11354 <= cadenceno <= 11366 or 1464.0158778 <= time <= 1464.265871):
		logger.debug("Manual Exclude: Excessive Earth-shine")
		mask[:, :] = True

	# Specific problems sometimes found where the whole image is zero:
	# One example is in Sector 6 (DR8), camera 2, ccd 1.
	if is_tess and np.all(img == 0):
		logger.debug("Manual Exclude: Whole image is zero")
		mask[:, :] = True

	return mask

#------------------------------------------------------------------------------
def pixel_background_shenanigans(img, SumImage=None):
	"""

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""
	# Subtract reference image:
	if SumImage is not None:
		flux0 = img - SumImage

	# Run median filter to get rid of residuals from individual stars:
	flux0 = median_filter(flux0, size=15)

	return flux0
