#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Estimation of sky background in TESS Full Frame Images.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
import six
import numpy as np
from astropy.stats import SigmaClip
from photutils import Background2D, SExtractorBackground
from photometry.utilities import load_ffi_fits

def fit_background(image, catalog=None, flux_cutoff=8e4):
	"""
	Estimate background in Full Frame Image.

	Parameters:
		image (ndarray or string): Either the image as 2D ndarray or a path to FITS or NPY file where to load image from.
		catalog (`astropy.table.Table` object): Catalog of stars in the image. Is not yet being used for anything.
		flux_cutoff (float): Flux value at which any pixel above will be masked out of the background estimation.

	Returns:
		ndarray: Estimated background with the same size as the input image.
		ndarray: Boolean array specifying which pixels was used to estimate the background (``True`` if pixel was used).

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	# Load file:
	if isinstance(image, np.ndarray):
		img = image
	elif isinstance(image, six.string_types):
		if image.endswith('.npy'):
			img = np.load(image)
		else:
			img = load_ffi_fits(image)
	else:
		raise ValueError("Input image must be either 2D ndarray or path to file.")

	# Create mask
	# TODO: Use the known locations of bright stars
	mask = ~np.isfinite(img)
	mask |= (img > flux_cutoff)

	# Estimate the background:
	sigma_clip = SigmaClip(sigma=3.0, maxiters=5)
	bkg_estimator = SExtractorBackground()
	bkg = Background2D(img, (64, 64),
		filter_size=(3, 3),
		sigma_clip=sigma_clip,
		bkg_estimator=bkg_estimator,
		mask=mask,
		exclude_percentile=50)

	return bkg.background, mask
