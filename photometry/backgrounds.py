#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Estimation of sky background in TESS Full Frame Images.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
import six
import warnings
import numpy as np
from astropy.utils.exceptions import AstropyDeprecationWarning
warnings.filterwarnings('ignore', category=AstropyDeprecationWarning)
from astropy.wcs import WCS
from astropy.stats import SigmaClip
from scipy.stats import binned_statistic
from scipy.interpolate import InterpolatedUnivariateSpline
from photutils import Background2D, SExtractorBackground, BackgroundBase
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
from .utilities import load_ffi_fits
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
def fit_background(image, camera_centre=None, catalog=None, flux_cutoff=8e4,
		bkgiters=3, radial_cutoff=2400, radial_pixel_step=15):
	"""
	Estimate background in Full Frame Image.

	The background is estimated using a combination of a 2D estimate of the mode
	of the images (using background estimator from SExtractor), and a radial
	component to account for the corner-glow that is present in TESS FFIs.

	Parameters:
		image (ndarray or string): Either the image as 2D ndarray or a path to FITS or NPY file where to load image from.
		camera_centre (array-like): RA and DEC of camera centre from which to calculate the radial background component.
		catalog (`astropy.table.Table` object): Catalog of stars in the image. Is not yet being used for anything.
		flux_cutoff (float): Flux value at which any pixel above will be masked out of the background estimation.
		bkgiters (integer): Number of times to iterate the background components. Default=3.
		radial_cutoff (float): Radial distance in pixels from camera centre to start using radial component. Default=2400.
		radial_pixel_step (integer): Step sizes to use in radial component. Default=15.

	Returns:
		ndarray: Estimated background with the same size as the input image.
		ndarray: Boolean array specifying which pixels was used to estimate the background (``True`` if pixel was used).

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	# Load file:
	if isinstance(image, np.ndarray):
		img0 = image
	elif isinstance(image, six.string_types):
		if image.endswith('.npy'):
			img0 = np.load(image)
		else:
			img0, hdr = load_ffi_fits(image, return_header=True)
	else:
		raise ValueError("Input image must be either 2D ndarray or path to file.")

	# World Coordinate System solution:
	wcs = WCS(header=hdr, relax=True)

	# Create mask
	# TODO: Use the known locations of bright stars
	mask = ~np.isfinite(img0)
	mask |= (img0 > flux_cutoff)
	mask |= (img0 < 0)

	# Mask out pixels marked for manual exclude:
	mask |= pixel_manual_exclude(img0, hdr)

	# Setup background estimator:
	sigma_clip = SigmaClip(sigma=3.0, maxiters=5)
	bkg_estimator = SExtractorBackground(sigma_clip)

	# Create distance-image with distances (in pixels) from the camera centre:
	if camera_centre is not None:
		# Background estimator function to be evaluated in radial coordinates:
		#stat = lambda x: bkg_estimator.calc_background(x)
		stat = _reduce_mode

		# Create radial coordinates:
		xx, yy = np.meshgrid(
			np.arange(0, img0.shape[1], 1),
			np.arange(0, img0.shape[0], 1)
		)
		xycen = wcs.all_world2pix(np.atleast_2d(camera_centre), 0, ra_dec_order=True)
		xcen = xycen[0,0]
		ycen = xycen[0,1]
		r = np.sqrt((xx - xcen)**2 + (yy - ycen)**2)

		# Create the bins in which to evaluate the background estimator:
		radial_max = np.max(r) + radial_pixel_step
		bins = np.arange(radial_cutoff, radial_max, radial_pixel_step)
		bin_center = bins[1:] - radial_pixel_step/2
	else:
		bkgiters = 1

	# Iterate the radial and square background components:
	img_bkg_radial = 0
	img_bkg_square = 0
	for iters in range(bkgiters):
		if camera_centre is not None:
			# Remove the square component from image for next iteration:
			img = img0 - img_bkg_square

			# Evaluate the background estimator in radial rings:
			# We are working in logarithmic units since the mode estimator seems to
			# work better in that case.
			s2, _, _ = binned_statistic(r[~mask].flatten(), np.log10(img[~mask].flatten()),
				statistic=stat,
				bins=bins
			)

			# Optionally smooth the radial profile:
			#s2 = move_median_central(s2, 5)

			# Interpolate the radial curve and reshape back onto the 2D image:
			indx = ~np.isnan(s2)
			intp = InterpolatedUnivariateSpline(bin_center[indx], s2[indx], ext=3)
			img_bkg_radial = 10**intp(r)

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
