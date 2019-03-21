#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, unicode_literals
import argparse
import logging
import numpy as np
import h5py
import os.path
from photometry.plots import plt, plot_image
from matplotlib import animation
from tqdm import trange
from photometry.quality import PixelQualityFlags

#------------------------------------------------------------------------------
def _animate(k, imgs, ax, hdf_file):

	with h5py.File(hdf_file, 'r', libver='latest') as hdf:
		dset_name = '%04d' % k
		flux0 = np.asarray(hdf['images/' + dset_name])
		bkg = np.asarray(hdf['backgrounds/' + dset_name])
		img = np.asarray(hdf['pixel_flags/' + dset_name])

	# Plot background:
	imgs[0].set_data(flux0 + bkg)
	ax[0].set_title('Original Image - ' + dset_name)
	imgs[1].set_data(bkg)
	imgs[2].set_data(flux0)
	imgs[3].set_data(img & PixelQualityFlags.BackgroundShenanigans != 0)

	return imgs

#------------------------------------------------------------------------------
def make_movie(hdf_file):

	logger = logging.getLogger(__name__)
	logger.info("Processing '%s'", hdf_file)

	logger.info("Calculating image scales...")
	with h5py.File(hdf_file, 'r', libver='latest') as hdf:
		numfiles = len(hdf['images'])
		dummy_img = np.asarray(hdf['images/0000'])
		dummy_bkg = np.asarray(hdf['backgrounds/0000'])

		vmax = -np.inf
		vmin = np.inf
		for k in trange(numfiles):
			vmin1, vmax1 = np.nanpercentile(hdf['backgrounds/%04d' % k], [0, 100])
			vmin = min(vmin, vmin1)
			vmax = max(vmax, vmax1)

	logger.info("Creating movie...")
	fig, ax = plt.subplots(1, 4, figsize=(20, 6))

	imgs = [0,0,0,0]
	imgs[0] = plot_image(dummy_bkg, ax=ax[0], scale='sqrt', vmin=vmin, vmax=vmax, title='Original Image - 0000', xlabel=None, ylabel=None, cmap=plt.cm.viridis, make_cbar=True)
	imgs[1] = plot_image(dummy_bkg, ax=ax[1], scale='sqrt', vmin=vmin, vmax=vmax, title='Background', xlabel=None, ylabel=None, cmap=plt.cm.Blues, make_cbar=True)
	imgs[2] = plot_image(dummy_img, ax=ax[2], scale='sqrt', vmin=0, title='Background subtracted', xlabel=None, ylabel=None, cmap=plt.cm.viridis, make_cbar=True)
	imgs[3] = plot_image(dummy_img, ax=ax[3], scale='linear', vmin=0, vmax=1, title='Background Shenanigans', xlabel=None, ylabel=None, cmap=plt.cm.Reds, make_cbar=True, clabel='Flags')

	for a in ax:
		a.set_xticks([])
		a.set_yticks([])

	plt.tight_layout()

	ani = animation.FuncAnimation(fig, _animate, trange(numfiles), fargs=(imgs, ax, hdf_file), repeat=False, blit=True)
	ani.save(os.path.splitext(hdf_file)[0] + '.mp4', fps=10)
	plt.close(fig)

#------------------------------------------------------------------------------
if __name__ == '__main__':

	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Create movie of TESS camera.')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('files', help='HDF5 file to create movie from.', nargs='+')
	args = parser.parse_args()

	logging_level = logging.INFO
	if args.quiet:
		logging_level = logging.WARNING
	elif args.debug:
		logging_level = logging.DEBUG

	# Setup logging:
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	logger = logging.getLogger(__name__)
	logger.setLevel(logging_level)
	console = logging.StreamHandler()
	console.setFormatter(formatter)
	logger_parent = logging.getLogger('photometry')
	logger_parent.setLevel(logging_level)
	if not logger.hasHandlers(): logger.addHandler(console)
	if not logger_parent.hasHandlers(): logger_parent.addHandler(console)

	for fname in args.files:
		make_movie(fname)
