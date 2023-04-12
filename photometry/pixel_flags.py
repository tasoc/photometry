#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import logging
from scipy.ndimage.filters import median_filter
from scipy.ndimage import label
from bottleneck import nanmedian

#--------------------------------------------------------------------------------------------------
def pixel_manual_exclude(img):
	"""
	Manual Exclude of individual pixels in Full Frame Images.

	Parameters:
		img (:class:`io.FFIImage`): Image of which to create manual exclude mask.

	Returns:
		ndarray: Boolean image with the same dimentions as ``img``, containg true
			for pixels that have been marked as manual excludes.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)

	# Create manual exclude mask array:
	mask = np.zeros_like(img, dtype='bool')

	# Check if this is real TESS data:
	hdr = img.header
	if img.is_tess:
		time = 0.5*(hdr['TSTART'] + hdr['TSTOP'])
		cadenceno = hdr.get('FFIINDEX', np.inf)
	else:
		time = np.NaN
		cadenceno = np.inf

	# Mars falls in output channel D of camera 1, CCD 4 in the beginning of
	# TESS Sector 1, which floods the registers and messes up the images
	if img.is_tess and hdr['CAMERA'] == 1 and hdr['CCD'] == 4 and (cadenceno <= 4724 or hdr['TSTART'] <= 1325.881282301840):
		logger.debug("Manual Exclude: Register overflow due to Mars in FOV")
		mask[:, 1536:] = True

	elif img.is_tess and hdr['CAMERA'] == 1 and (11354 <= cadenceno <= 11366 or 1464.0158778 <= time <= 1464.265871):
		logger.debug("Manual Exclude: Excessive Earth-shine")
		mask[:, :] = True

	# Specific problems sometimes found where the whole image is zero:
	# One example is in Sector 6 (DR8), camera 2, ccd 1.
	if img.is_tess and np.all(img.data == 0):
		logger.debug("Manual Exclude: Whole image is zero")
		mask[:, :] = True

	return mask

#--------------------------------------------------------------------------------------------------
def pixel_background_shenanigans(img, SumImage=None):
	"""

	Parameters:
		img (:class:`io.FFIImage`): Image of which to create mask.
		SumImage (ndarray): Average image to compare with.

	Returns:
		:func:`np.ndarray`: Boolean mask indicating pixels with background shenanigans.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""
	# Subtract reference image:
	flux0 = (img - SumImage) if SumImage is not None else img

	# Run median filter to get rid of residuals from individual stars:
	flux0 = median_filter(flux0, size=15)

	return flux0

#--------------------------------------------------------------------------------------------------
def pixel_detect_bad_smear_columns(img):
	"""

	Parameters:
		img (:class:`io.FFIImage`): Image of which to create mask.

	Returns:
		:func:`np.ndarray`: Boolean mask indicating bad columns in the image.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	mask = np.zeros(img.shape, dtype='bool')
	if not img.is_tess:
		return mask

	# Check the median of the virtual smear column.
	# This should not be very large, unless a bright star has contaminated it.
	ms = nanmedian(img.vsmear, axis=0)
	indx1 = (ms > 250)

	# Also check the image itself for large negative columns:
	ms2 = nanmedian(img, axis=0)
	indx2 = (ms2 < -1000)

	# Usually the effect of the bad smear is best detected in the virtual smear,
	# but the image itself shows which columns were actually affected.
	# Therefore, mark columns as bad if they either of the indicies above are true,
	# but require that the virtual smear condition is true in the cluster of columns:
	lab_both, num_both = label(indx1 | indx2)
	for k in range(num_both):
		indx_this = (lab_both == k+1)
		if np.any(indx1 & indx_this):
			mask[:, indx_this] = True

	return mask
