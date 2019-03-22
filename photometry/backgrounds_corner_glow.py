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
from scipy.interpolate import interp1d,  InterpolatedUnivariateSpline
import matplotlib.animation as animation
from photutils import Background2D, SExtractorBackground, BackgroundBase
import warnings
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
warnings.filterwarnings('ignore', module='scipy', category=FutureWarning)
import sys
from bottleneck import allnan

def reduce_mode(x):
	if len(x) == 0:
		return np.NaN
	x = np.asarray(x, dtype=np.float64)
	kde = KDE(x)
	kde.fit(gridsize=2000)
	return kde.support[np.argmax(kde.density)]

def _mode(data):
	modes = np.zeros([data.shape[0]])
	for i in range(data.shape[0]):
		kde = KDE(data[i,:])
		kde.fit(gridsize=2000)

		pdf = kde.density
		xx = kde.support
		modes[i] = xx[np.argmax(pdf)]
	return modes

class ModeBackground(BackgroundBase):
	def calc_background(self, data, axis=None):
		if self.sigma_clip is not None:
			data = self.sigma_clip(data, axis=axis)
		bkg = np.atleast_1d(_mode(np.asarray(data, dtype=np.float64)))
		return bkg

def pixel_manual_exclude(img, hdr):
	mask = np.zeros_like(img, dtype='bool')

	#
	if hdr['SECTOR'] == 1 and hdr['CAMERA'] == 1 and hdr['CCD'] == 4 and hdr['FFIINDEX'] <= 4724:
		mask[:, 1536:] = True

	return mask


if __name__ == '__main__':

	plt.close('all')

	# Settings:
	flux_cutoff = 8e4 # Absolute flux cutoff
	pixel_step = 15 # Radial size of bins to use in radial background
	radial_cutoff = 2400 # Distance where the radial background kicks in
	bkgiters = 3 # Number of background iterations

	# Centre of sector 1, camera 1
	# TODO: Get this from the catalog file
	camera_center = np.array([[324.566998914166, -33.172999301379]])

	fig_radial_profile = plt.figure()
	axradial = fig_radial_profile.add_subplot(111)
	axradial.set_ylim(2.2, 4.0)
	axradial.set_xlim(radial_cutoff-pixel_step/5, 2963)
	ims = []

	for k in range(27, 29): # The timestamp to run

		# Just loading the FFI images from the HDF5 files:
		with h5py.File(r'E:\tess_data\S01_DR01\sector001_camera1_ccd4.hdf5', 'r') as hdf:
			img0 = np.asarray(hdf['images']['%04d' % k])
			img0 += np.asarray(hdf['backgrounds']['%04d' % k])

			# World Coordinate System solution:
			hdr_string = hdf['wcs']['%04d' % k][0]
			if not isinstance(hdr_string, six.string_types): hdr_string = hdr_string.decode("utf-8") # For Python 3
			wcs = WCS(header=fits.Header().fromstring(hdr_string), relax=True) # World Coordinate system solution.

			hdr = dict(hdf['images'].attrs)
			hdr['SECTOR'] = 1
			hdr['CAMERA'] = 1
			hdr['CCD'] = 4
			hdr['FFIINDEX'] = hdf['cadenceno'][k]
			print(hdr['FFIINDEX'])

			print(hdf['time'][k])

			#sumimage = np.asarray(hdf['sumimage'])
			#img -= sumimage

		# Create mask
		# TODO: Use the known locations of bright stars
		mask = ~np.isfinite(img0)
		mask |= (img0 > flux_cutoff)
		mask |= (img0 < 0)

		mask |= pixel_manual_exclude(img0, hdr)

		# Setup background estimator:
		sigma_clip = SigmaClip(sigma=3.0, iters=5)
		bkg_estimator = SExtractorBackground(sigma_clip)
	#	stat = lambda x: bkg_estimator.calc_background(x)
		stat = reduce_mode

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
		plt.subplot(121)
		h = plot_image(img0, percentile=90)
		plt.subplot(122)
		plot_image(mask, percentile=100, scale='linear')
		plt.colorbar(h)
		plt.show()

		m = np.max(r) + pixel_step
		print(m)
		rs = np.arange(radial_cutoff - pixel_step/5, np.max(r) + pixel_step/5, pixel_step/5)

		img = np.copy(img0)

		#figitt, axitt = plt.subplots(1,bkgiters, figsize=(18,6))
		#figitt1, axitt1 = plt.subplots(1,bkgiters, figsize=(18,6))

		#figittcomp = plt.figure()
		#axittcomp = figittcomp.add_subplot(111)

		for iters in range(bkgiters):
			print('iteration', iters)

			# Evaluate the background estimator in radial rings:
			s2, bin_edges, _ = binned_statistic(r[~mask].flatten(), np.log10(img[~mask].flatten()),
				statistic=stat,
				bins=np.arange(radial_cutoff, m, pixel_step)
			)
			bin_center = bin_edges[1:] - pixel_step/2

			# Optionally smooth the radial profile:
			#s2 = move_median_central(s2, 5)

			# Interpolate the radial curve and reshape back onto the 2D image:
			indx = ~np.isnan(s2)
			intp = InterpolatedUnivariateSpline(bin_center[indx], s2[indx], ext=3)
	#		intp = interp1d(bin_center[indx], s2[indx],
	#			kind='linear',
	#			bounds_error=False,
	#			fill_value=(s2[indx][0], s2[indx][-1]),
	#			assume_sorted=True
	#		)
			img_bkg_radial = 10**intp(r)

			# Plot radial profile:
	#		plt.figure()
	#		plt.scatter(r[(~mask) & (r>2300)].flatten(), img[(~mask) & (r>2300)].flatten(), facecolors='None', edgecolors='k', alpha=0.1)
	#		plt.scatter(bin_center, 10**s2, c='r')
	#		plt.plot(rs, intp(rs), 'r-')
	#		plt.xlim(xmin=radial_cutoff - 100)

			# Run 2D square tiles background estimation:
			bkg = Background2D(img0 - img_bkg_radial, (64, 64),
				filter_size=(3, 3),
				sigma_clip=sigma_clip,
				bkg_estimator=bkg_estimator,
				mask=mask,
				exclude_percentile=50)
			img_bkg_square = bkg.background

			"""
			#if iters==0:
			#	img_bkg000 = img_bkg_radial + img_bkg_square
#
#			else:
			#	diff = img_bkg - (img_bkg_radial + img_bkg_square)
			#	plot_image(diff, ax = axitt[iters-1], scale='log',percentile=100)
#
#				kde = KDE(diff[(~mask) & (r>2300)])
#				kde.fit(gridsize=2000)
##
	#			pdf = kde.density
	#			xx = kde.support
	#			axitt1[iters-1].plot(xx, pdf)


	#			axittcomp.scatter(iters-1, np.nanpercentile(diff, 50), color='k', marker='o')
	#			axittcomp.scatter(iters-1, np.nanpercentile(diff, 5), color='b', marker='o')
	#			axittcomp.scatter(iters-1, np.nanpercentile(diff, 95), color='b', marker='o')
	#			axittcomp.scatter(iters-1, np.nanmedian(np.abs(diff-np.nanmedian(diff))), color='r', marker='o')
			"""

			# Total background image:
			img_bkg = img_bkg_radial + img_bkg_square

			"""
			pkw = {'xlabel': None, 'ylabel': None}
			plt.figure()
			plt.subplot(151)
			imglog = np.log10(img + np.abs(np.nanmin(img)) + 1)
			h = plot_image(imglog, title='Original image', **pkw)
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
			"""

		axradial.plot(rs, intp(rs), '-', color='0.8', zorder=1)
		ims.append(axradial.plot(rs, intp(rs), 'r-', zorder=2))

		#plot_image(img_bkg000 - img_bkg, ax = axitt[iters], percentile=100)
		#diff = img_bkg000 - img_bkg
		#kde = KDE(diff[(~mask) & (r>2300)])
		#kde.fit(gridsize=2000)

		#pdf = kde.density
		#xx = kde.support
		#axitt1[iters].plot(xx, pdf)

		# Plot final background:
		#plt.figure()
		#plt.subplot(121)
		#plot_image(img_bkg, title='Final background')
		#plt.subplot(122)
		#plot_image(img0 - img_bkg, title='Original - Final background')

	# Radial profile animation:
	im_ani = animation.ArtistAnimation(fig_radial_profile, ims, interval=200) # , blit=True
	plt.show()
