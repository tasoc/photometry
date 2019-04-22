#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create movie of FFIs and extracted backgrounds.

This program will create a MP4 movie file with an animation of the extracted
backgrounds and flags from an HDF5 file created by the photometry pipeline.

Example:
	To create a MP4 movie for a specific file, run the program with the HDF5 file as input:

	>>> python run_ffimovie.py path/to/file/sector01_camera1_ccd1.hdf5

Example:
	Multiple files can be processed at a time. They will be processed in parallel.

	>>> python run_ffimovie.py file1.hdf5 file2.hdf5

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, unicode_literals
from six.moves import map
import argparse
import logging
import numpy as np
import h5py
import os.path
import multiprocessing
from photometry.plots import plt, plot_image
from matplotlib import animation
from tqdm import trange
from photometry.quality import PixelQualityFlags

#------------------------------------------------------------------------------
def _animate(k, imgs, ax, hdf):

	dset_name = '%04d' % k
	flux0 = np.asarray(hdf['images/' + dset_name])
	bkg = np.asarray(hdf['backgrounds/' + dset_name])

	# Plot original image, background and new image:
	imgs[0].set_data(flux0 + bkg)
	ax[0].set_title('Original Image - ' + dset_name)
	imgs[1].set_data(bkg)
	imgs[2].set_data(flux0)

	# Background Shenanigans flags, if available:
	if 'pixel_flags/' + dset_name in hdf:
		img = np.asarray(hdf['pixel_flags/' + dset_name])
		imgs[3].set_data(img & PixelQualityFlags.BackgroundShenanigans != 0)

	return imgs

#------------------------------------------------------------------------------
def make_movie(hdf_file, fps=15, overwrite=True):
	"""
	Create animation of the contents of a HDF5 files produced by the photometry pipeline.

	The function will create a MP4 movie file with the same name as the input file,
	placed in the same directory, containing the animation.

	Parameters:
		hdf_file (string): Path to the HDF5 file to produce movie from.
		fps (integer): Frames per second of generated movie. Default=15.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)

	logger.info("Processing '%s'", hdf_file)

	output_file = os.path.splitext(hdf_file)[0] + '.mp4'
	if os.path.exists(output_file):
		if overwrite:
			logger.debug("Deleting existing output file")
			os.remove(output_file)
		else:
			logger.info("Movie file already exists")
			return output_file

	# Open HDF5 file for reading:
	with h5py.File(hdf_file, 'r', libver='latest') as hdf:
		numfiles = len(hdf['images'])
		dummy_img = np.asarray(hdf['images/0000'])
		dummy_bkg = np.asarray(hdf['backgrounds/0000'])

		# Calculate scales to use for plotting the images:
		logger.info("Calculating image scales...")
		vmax = np.empty(numfiles)
		vmin = np.empty(numfiles)
		vmax2 = np.empty(numfiles)
		vmin2 = np.empty(numfiles)
		for k in trange(numfiles):
			vmin[k], vmax[k] = np.nanpercentile(hdf['backgrounds/%04d' % k], [1.0, 99.0])
			vmin2[k], vmax2[k] = np.nanpercentile(hdf['images/%04d' % k], [1.0, 99.0])

		vmin = np.nanpercentile(vmin, 25.0)
		vmax = np.nanpercentile(vmax, 75.0)
		vmin2 = np.nanpercentile(vmin2, 25.0)
		vmax2 = np.nanpercentile(vmax2, 75.0)

		logger.info("Creating movie...")
		fig, ax = plt.subplots(1, 4, figsize=(20, 6))

		imgs = [0,0,0,0]
		imgs[0] = plot_image(dummy_bkg, ax=ax[0], scale='sqrt', vmin=vmin, vmax=vmax, title='Original Image - 0000', xlabel=None, ylabel=None, cmap=plt.cm.viridis, make_cbar=True)
		imgs[1] = plot_image(dummy_bkg, ax=ax[1], scale='sqrt', vmin=vmin, vmax=vmax, title='Background', xlabel=None, ylabel=None, cmap=plt.cm.viridis, make_cbar=True)
		imgs[2] = plot_image(dummy_img, ax=ax[2], scale='sqrt', vmin=vmin2, vmax=vmax2, title='Background subtracted', xlabel=None, ylabel=None, cmap=plt.cm.viridis, make_cbar=True)
		imgs[3] = plot_image(dummy_img, ax=ax[3], scale='linear', vmin=0, vmax=1, title='Background Shenanigans', xlabel=None, ylabel=None, cmap=plt.cm.Reds, make_cbar=True, clabel='Flags')

		for a in ax:
			a.set_xticks([])
			a.set_yticks([])

		fig.set_tight_layout('tight')

		ani = animation.FuncAnimation(fig, _animate, trange(numfiles), fargs=(imgs, ax, hdf), repeat=False, blit=True)
		ani.save(output_file, fps=fps) # writer='ffmpeg'
		plt.close(fig)

	return output_file

#------------------------------------------------------------------------------
if __name__ == '__main__':
	multiprocessing.freeze_support() # for Windows support

	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Create movie of TESS camera.')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('-j', '--jobs', help='Maximal number of jobs to run in parallel.', type=int, default=None, nargs='?')
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

	# Get the number of processes we can spawn in case it is needed for calculations:
	threads = args.jobs
	if threads is None:
		threads = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
	threads = min(threads, len(args.files))
	logger.info("Using %d processes.", threads)

	# Start pool of workers:
	if threads > 1:
		pool = multiprocessing.Pool(threads)
		m = pool.imap
	else:
		m = map

	for fname in m(make_movie, args.files):
		logger.info("Created movie: %s", fname)

	# Close workers again:
	if threads > 1:
		pool.close()
		pool.join()