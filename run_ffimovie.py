#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create movie of FFIs and extracted backgrounds.

This program will create a MP4 movie file with an animation of the extracted
backgrounds and flags from an HDF5 file created by the photometry pipeline.

This program requires the program `FFmpeg <https://ffmpeg.org/>`_ to be installed.

Example:
	To create a MP4 movie for a specific file, run the program with the HDF5 file as input:

	>>> python run_ffimovie.py path/to/file/sector01_camera1_ccd1.hdf5

Example:
	If you wish to change the the frame-rate or the size of the generated movie,
	you can use the ``fps`` and ``dpi`` settings:

	>>> python run_ffimovie.py --fps=15 --dpi=100 file.hdf5

Example:
	Multiple files can be processed at a time. Default behavior is to process
	them one at a time, but can also be processed in parallel by specifying
	the number of processes to run vis the ``--jobs`` option:

	>>> python run_ffimovie.py --jobs=2 file1.hdf5 file2.hdf5

	If number of processes is set to zero (``--jobs=0``), the number of processes
	will be set to the number of available CPUs.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import argparse
import logging
import numpy as np
import h5py
import os.path
import functools
import multiprocessing
from photometry.plots import plt, plot_image
from matplotlib import animation
from tqdm import tqdm
from photometry.quality import PixelQualityFlags

#------------------------------------------------------------------------------
def make_movie(hdf_file, fps=15, dpi=100, overwrite=False):
	"""
	Create animation of the contents of a HDF5 files produced by the photometry pipeline.

	The function will create a MP4 movie file with the same name as the input file,
	placed in the same directory, containing the animation.

	Parameters:
		hdf_file (string): Path to the HDF5 file to produce movie from.
		fps (integer): Frames per second of generated movie. Default=15.
		dpi (integer): DPI of the movie. Default=100.
		overwrite (boolean): Overwrite existing MP4 files? Default=False.

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
		dummy_img = np.zeros_like(hdf['images/0000'])

		# Calculate scales to use for plotting the images:
		logger.info("Calculating image scales...")
		vmax = np.empty(numfiles)
		vmin = np.empty(numfiles)
		vmax2 = np.empty(numfiles)
		vmin2 = np.empty(numfiles)
		for k in range(numfiles):
			vmin[k], vmax[k] = np.nanpercentile(hdf['backgrounds/%04d' % k], [1.0, 99.0])
			vmin2[k], vmax2[k] = np.nanpercentile(hdf['images/%04d' % k], [1.0, 99.0])

		vmin = np.nanpercentile(vmin, 25.0)
		vmax = np.nanpercentile(vmax, 75.0)
		vmin2 = np.nanpercentile(vmin2, 25.0)
		vmax2 = np.nanpercentile(vmax2, 75.0)

		logger.info("Creating movie...")
		with plt.style.context('dark_background'):
			fig, ax = plt.subplots(1, 4, figsize=(20, 6))

			imgs = [0,0,0,0]
			imgs[0] = plot_image(dummy_img, ax=ax[0], scale='sqrt', vmin=vmin, vmax=vmax, title='Original Image - 0000', xlabel=None, ylabel=None, cmap=plt.cm.viridis, make_cbar=True)
			imgs[1] = plot_image(dummy_img, ax=ax[1], scale='sqrt', vmin=vmin, vmax=vmax, title='Background', xlabel=None, ylabel=None, cmap=plt.cm.viridis, make_cbar=True)
			imgs[2] = plot_image(dummy_img, ax=ax[2], scale='sqrt', vmin=vmin2, vmax=vmax2, title='Background subtracted', xlabel=None, ylabel=None, cmap=plt.cm.viridis, make_cbar=True)
			imgs[3] = plot_image(dummy_img, ax=ax[3], scale='linear', vmin=0, vmax=1, title='Background Shenanigans', xlabel=None, ylabel=None, cmap=plt.cm.Reds, make_cbar=True, clabel='Flags')

			for a in ax:
				a.set_xticks([])
				a.set_yticks([])

			fig.set_tight_layout('tight')

			writer = animation.FFMpegWriter(fps=fps)
			with writer.saving(fig, output_file, dpi):
				for k in range(numfiles):

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

					writer.grab_frame()

			plt.close(fig)

	return output_file

#------------------------------------------------------------------------------
if __name__ == '__main__':
	multiprocessing.freeze_support() # for Windows support

	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Create movie of TESS camera.')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('-o', '--overwrite', help='Overwrite existing files.', action='store_true')
	parser.add_argument('-j', '--jobs', help='Maximal number of jobs to run in parallel.', type=int, default=1, nargs='?')
	parser.add_argument('--fps', help='Frames per second of generated movie.', type=int, default=15, nargs='?')
	parser.add_argument('--dpi', help='DPI of generated movie.', type=int, default=100, nargs='?')
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

	tqdm_settings = {
		'disable': not logger.isEnabledFor(logging.INFO),
		'total': len(args.files)
	}

	# Get the number of processes we can spawn in case it is needed for calculations:
	threads = args.jobs
	if threads <= 0:
		threads = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
	threads = min(threads, len(args.files))
	logger.info("Using %d processes.", threads)

	# Start pool of workers:
	if threads > 1:
		pool = multiprocessing.Pool(threads)
		m = pool.imap
	else:
		m = map

	# Make wrapper function with all settings:
	make_movie_wrapper = functools.partial(make_movie,
		fps=args.fps,
		dpi=args.dpi,
		overwrite=args.overwrite
	)

	# Process the files on at a time, in parallel if needed:
	for fname in tqdm(m(make_movie_wrapper, args.files), **tqdm_settings):
		logger.info("Created movie: %s", fname)

	# Close workers again:
	if threads > 1:
		pool.close()
		pool.join()