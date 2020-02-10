#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Estimation of sky background in TESS Full Frame Images.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import logging
import numpy as np
from astropy.stats import SigmaClip
from scipy.stats import binned_statistic
from scipy.interpolate import InterpolatedUnivariateSpline
from photutils import Background2D, SExtractorBackground, BackgroundBase
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
from .utilities import load_ffi_fits, move_median_central
from .pixel_flags import pixel_manual_exclude

#------------------------------------------------------------------------------
def _reduce_mode(x):
	if len(x) == 0:
		return np.NaN
	x = np.asarray(x, dtype=np.float64)
	kde = KDE(x)
	kde.fit(gridsize=2000)
	return kde.support[np.argmax(kde.density)]

#------------------------------------------------------------------------------
def _mode(data):
	modes = np.zeros([data.shape[0]])
	for i in range(data.shape[0]):
		kde = KDE(data[i,:])
		kde.fit(gridsize=2000)
		modes[i] = kde.support[np.argmax(kde.density)]
	return modes

#------------------------------------------------------------------------------
class ModeBackground(BackgroundBase):
	def calc_background(self, data, axis=None):
		if self.sigma_clip is not None:
			data = self.sigma_clip(data, axis=axis)
		bkg = np.atleast_1d(_mode(np.asarray(data, dtype=np.float64)))
		return bkg

#------------------------------------------------------------------------------
def fit_background(image, catalog=None, flux_cutoff=8e4,
		bkgiters=3, radial_cutoff=2400, radial_pixel_step=15, radial_smooth=3):
	"""
	Estimate background in Full Frame Image.

	The background is estimated using a combination of a 2D estimate of the mode
	of the images (using background estimator from SExtractor), and a radial
	component to account for the corner-glow that is present in TESS FFIs.

	Parameters:
		image (ndarray or string): Either the image as 2D ndarray or a path to FITS or NPY file where to load image from.
		catalog (`astropy.table.Table` object): Catalog of stars in the image. Is not yet being used for anything.
		flux_cutoff (float): Flux value at which any pixel above will be masked out of the background estimation.
		bkgiters (integer): Number of times to iterate the background components. Default=3.
		radial_cutoff (float): Radial distance in pixels from camera centre to start using radial component. Default=2400.
		radial_pixel_step (integer): Step sizes to use in radial component. Default=15.
		radial_smooth (integer): Width of median smoothing on radial profile. Default=3.

	Returns:
		ndarray: Estimated background with the same size as the input image.
		ndarray: Boolean array specifying which pixels was used to estimate the background (``True`` if pixel was not used).

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)

	# Load file:
	hdr = {}
	if isinstance(image, np.ndarray):
		img0 = image
	elif isinstance(image, str):
		if image.endswith('.npy'):
			img0 = np.load(image)
		else:
			img0, hdr = load_ffi_fits(image, return_header=True)
	else:
		raise ValueError("Input image must be either 2D ndarray or path to file.")

	# Check if this is real TESS data:
	# Could proberly be done more elegant, but if it works, it works...
	is_tess = (hdr.get('TELESCOP') == 'TESS' and hdr.get('NAXIS1') == 2136 and hdr.get('NAXIS2') == 2078)

	# Create mask
	# TODO: Use the known locations of bright stars
	mask = ~np.isfinite(img0)
	mask |= (img0 > flux_cutoff)
	mask |= (img0 < 0)

	# Mask out pixels marked for manual exclude:
	mask |= pixel_manual_exclude(img0, hdr)

	# If the entire image has been masked out,
	# we should just stop now and return NaNs:
	if np.all(mask):
		return np.full_like(img0, np.NaN), mask

	# Setup background estimator:
	sigma_clip = SigmaClip(sigma=3.0, maxiters=5)
	bkg_estimator = SExtractorBackground(sigma_clip)

	# Create distance-image with distances (in pixels) from the camera centre:
	use_radial_component = True
	if is_tess:
		# Background estimator function to be evaluated in radial coordinates:
		#stat = lambda x: bkg_estimator.calc_background(x)
		stat = _reduce_mode

		# Lookup table for the pixel coordinates of the camera centre with respect
		# to each CCD in TESS.
		# This table is created using the average from the WCS in Sector 1 FFIs.
		# TODO: Isn't it possible to get this in a better/more accurate way?
		camera = hdr.get('CAMERA')
		ccd = hdr.get('CCD')
		xycen = {
			(1, 1): [2158.222313, 2099.523364],
			(1, 2): [-5.653058, 2098.018608],
			(1, 3): [2141.511437, 2099.868226],
			(1, 4): [-22.406442, 2100.116443],
			(2, 1): [2148.588316, 2094.033024],
			(2, 2): [-16.806140, 2095.810070],
			(2, 3): [2151.351646, 2105.747100],
			(2, 4): [-13.118570, 2105.982211],
			(3, 1): [2152.175481, 2092.337442],
			(3, 2): [-10.494413, 2093.108135],
			(3, 3): [2145.029218, 2107.883573],
			(3, 4): [-17.374782, 2105.296746],
			(4, 1): [2149.259760, 2091.433315],
			(4, 2): [-12.906931, 2093.350054],
			(4, 3): [2148.906766, 2110.730620],
			(4, 4): [-14.629676, 2111.341670],
		}.get((camera, ccd))
		if xycen is None:
			raise ValueError("Invalid CAMERA or CCD in header: CAMERA=%s, CCD=%s" % (camera, ccd))

		# Create radial coordinates:
		# Note that these are in "real" coordinates like the ones in the WCS,
		# but zero-based.
		xx, yy = np.meshgrid(
			np.arange(44, img0.shape[1]+44, 1),
			np.arange(0, img0.shape[0], 1)
		)
		r = np.sqrt((xx - xycen[0])**2 + (yy - xycen[1])**2)

		# Create the bins in which to evaluate the background estimator:
		radial_max = np.max(r) + radial_pixel_step
		bins = np.arange(radial_cutoff, radial_max, radial_pixel_step)
		bin_center = bins[1:] - radial_pixel_step/2
	else:
		use_radial_component = False
		bkgiters = 1

	# Iterate the radial and square background components:
	img_bkg_radial = 0
	img_bkg_square = 0
	for iters in range(bkgiters):
		if use_radial_component:
			# Remove the square component from image for next iteration:
			img = img0 - img_bkg_square

			# Evaluate the background estimator in radial rings:
			# We are working in logarithmic units since the mode estimator seems to
			# work better in that case.
			s2, _, _ = binned_statistic(
				r[~mask].flatten(),
				np.log10(img[~mask].flatten()),
				statistic=stat,
				bins=bins
			)

			# Optionally smooth the radial profile:
			if radial_smooth:
				s2 = move_median_central(s2, radial_smooth)

			# Interpolate the radial curve and reshape back onto the 2D image:
			indx = ~np.isnan(s2)
			Ngood = np.sum(indx)
			if Ngood >= 3: # The required number of points for qubic spline
				try:
					intp = InterpolatedUnivariateSpline(bin_center[indx], s2[indx], k=3, ext=3)
					img_bkg_radial = 10**intp(r)
				except ValueError:
					logger.exception("Background interpolation failed (N=%d).", Ngood)
					img_bkg_radial = 0
			else:
				logger.warning("Not enough points for radial interpolation (N=%d).", Ngood)
				img_bkg_radial = 0

		# Run 2D square tiles background estimation:
		bkg = Background2D(img0 - img_bkg_radial, (64, 64),
			filter_size=(3, 3),
			sigma_clip=sigma_clip,
			bkg_estimator=bkg_estimator,
			mask=mask,
			exclude_percentile=50)
		img_bkg_square = bkg.background

	# Total background image:
	img_bkg = img_bkg_radial + img_bkg_square

	return img_bkg, mask
