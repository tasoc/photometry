#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, unicode_literals
import six
import numpy as np
import matplotlib.pyplot as plt
import h5py
from plots import plot_image
from utilities import move_median_central
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import SigmaClip
from scipy.stats import binned_statistic
from scipy.interpolate import interp1d
from photutils import Background2D, SExtractorBackground
import warnings
warnings.filterwarnings('ignore', module='scipy', category=FutureWarning)


if __name__ == '__main__':

	plt.close('all')

	# Settings:
	flux_cutoff = 8e4 # Absolute flux cutoff
	pixel_step = 15 # Radial size of bins to use in radial background
	radial_cutoff = 2400 # Distance where the radial background kicks in
	bkgiters = 1 # Number of background iterations

	k = 500 # The timestamp to run

	# Centre of sector 1, camera 1
	# TODO: Get this from the catalog file
	camera_center = np.array([[324.566998914166, -33.172999301379]])

	# Just loading the FFI images from the HDF5 files:
	with h5py.File(r'E:\tess_data\S01_DR01\sector001_camera1_ccd2.hdf5', 'r') as hdf:
		img0 = np.asarray(hdf['images']['%04d' % k])
		img0 += np.asarray(hdf['backgrounds']['%04d' % k])

		# World Coordinate System solution:
		hdr_string = hdf['wcs']['%04d' % k][0]
		if not isinstance(hdr_string, six.string_types): hdr_string = hdr_string.decode("utf-8") # For Python 3
		wcs = WCS(header=fits.Header().fromstring(hdr_string), relax=True) # World Coordinate system solution.

		#sumimage = np.asarray(hdf['sumimage'])
		#img -= sumimage

	# Create mask
	# TODO: Use the known locations of bright stars
	mask = ~np.isfinite(img0)
	mask |= (img0 > flux_cutoff)
	mask |= (img0 < 0)

	# Setup background estimator:
	sigma_clip = SigmaClip(sigma=3.0, iters=5)
	bkg_estimator = SExtractorBackground(sigma_clip)
	stat = lambda x: bkg_estimator.calc_background(x)

	# Create distance-image with distances (in pixels) from the camera centre:
	xx, yy = np.meshgrid(
		np.arange(0, img0.shape[1], 1),
		np.arange(0, img0.shape[0], 1)
	)
	xycen = wcs.all_world2pix(camera_center, 0, ra_dec_order=True)
	xcen = xycen[0,0]
	ycen = xycen[0,1]
	r = np.sqrt((xx - xcen)**2 + (yy - ycen)**2)

	plt.figure()
	h = plot_image(img0, percentile=90)
	plt.colorbar(h)

	m = np.max(r) + pixel_step
	rs = np.arange(np.min(r), np.max(r) + pixel_step/5, pixel_step/5)

	img = np.copy(img0)
	for iters in range(bkgiters):
		# Evaluate the background estimator in radial rings:
		s2, bin_edges, _ = binned_statistic(r[~mask].flatten(), img[~mask].flatten(),
			statistic=stat,
			bins=np.arange(radial_cutoff, m, pixel_step)
		)
		bin_center = bin_edges[1:] - pixel_step/2

		# Optionally smooth the radial profile:
		s2 = move_median_central(s2, 5)

		# Interpolate the radial curve and reshape back onto the 2D image:
		indx = ~np.isnan(s2)
		intp = interp1d(bin_center[indx], s2[indx],
			kind='linear',
			bounds_error=False,
			fill_value=(s2[indx][0], s2[indx][-1]),
			assume_sorted=True
		)
		img_bkg_radial = intp(r)

		# Plot radial profile:
		plt.figure()
		#plt.scatter(r[~mask].flatten(), img[~mask].flatten(), alpha=0.3)
		plt.scatter(bin_center, s2, c='r')
		plt.plot(rs, intp(rs), 'r-')
		plt.xlim(xmin=radial_cutoff - 100)

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

		pkw = {'xlabel': None, 'ylabel': None}
		plt.figure()
		plt.subplot(151)
		h = plot_image(img, title='Original image', **pkw)
		plt.subplot(152)
		plot_image(img_bkg_radial, percentile=100, title='Radial background', **pkw)
		plt.subplot(153)
		plot_image(img0 - img_bkg_radial, title='Original - Radial', **pkw)
		plt.subplot(154)
		plot_image(img_bkg_square, percentile=100, title='Square background', **pkw)
		plt.subplot(155)
		plot_image(img0 - img_bkg_square, title='Original - Square', **pkw)

		# Remove the square component from image for next iteration:
		img = img0 - img_bkg_square

	# Plot final background:
	plt.figure()
	plt.subplot(121)
	plot_image(img_bkg, title='Final background')
	plt.subplot(122)
	plot_image(img0 - img_bkg, title='Original - Final background')
	plt.show()
