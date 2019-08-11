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
	To create movies of all HDF5 files in a given directory, simply pass the
	the path to the directory as the input:

	>>> python run_ffimovie.py path/to/directory/

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
import re
from tqdm import tqdm, trange
from photometry.plots import plt, plot_image
from matplotlib import animation
from matplotlib.colors import ListedColormap
from photometry.quality import PixelQualityFlags
from photometry.utilities import find_hdf5_files, TqdmLoggingHandler

#------------------------------------------------------------------------------
def set_copyright(fig, xpos=0.01, ypos=0.99, fontsize=12):
	plt.text(ypos, xpos, 'Created by TASOC',
		verticalalignment='bottom', horizontalalignment='right',
        transform=fig.transFigure,
        color='0.3', fontsize=fontsize)

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
	tqdm_settings = {'disable': not logger.isEnabledFor(logging.INFO)}
	logger.info("Processing '%s'", hdf_file)

	# File to be created:
	output_file = os.path.splitext(hdf_file)[0] + '.mp4'
	if os.path.exists(output_file):
		if overwrite:
			logger.debug("Deleting existing output file")
			os.remove(output_file)
		else:
			logger.info("Movie file already exists")
			return output_file

	# Open HDF5 file:
	# We need to have write-privaledges because we are going to updated some attributes
	with h5py.File(hdf_file, 'r+', libver='latest') as hdf:
		numfiles = len(hdf['images'])
		dummy_img = np.zeros_like(hdf['images/0000'])
		time = np.asarray(hdf['time'])
		cadenceno = np.asarray(hdf['cadenceno'])
		sector = hdf['images'].attrs.get('SECTOR')
		camera = hdf['images'].attrs.get('CAMERA')
		ccd = hdf['images'].attrs.get('CCD')

		# Load the image scales if they have already been calculated:
		vmin = hdf['backgrounds'].attrs.get('movie_vmin')
		vmax = hdf['backgrounds'].attrs.get('movie_vmax')
		vmin2 = hdf['images'].attrs.get('movie_vmin')
		vmax2 = hdf['images'].attrs.get('movie_vmax')

		# Calculate scales to use for plotting the images:
		if not vmin:
			logger.info("Calculating image scales...")
			vmax = np.empty(numfiles)
			vmin = np.empty(numfiles)
			vmax2 = np.empty(numfiles)
			vmin2 = np.empty(numfiles)
			for k in trange(numfiles, **tqdm_settings):
				vmin[k], vmax[k] = np.nanpercentile(hdf['backgrounds/%04d' % k], [1.0, 99.0])
				vmin2[k], vmax2[k] = np.nanpercentile(hdf['images/%04d' % k], [1.0, 99.0])

			vmin = np.nanpercentile(vmin, 25.0)
			vmax = np.nanpercentile(vmax, 75.0)
			vmin2 = np.nanpercentile(vmin2, 25.0)
			vmax2 = np.nanpercentile(vmax2, 75.0)

			# Save image scales to HDF5 file:
			hdf['backgrounds'].attrs['movie_vmin'] = vmin
			hdf['backgrounds'].attrs['movie_vmax'] = vmax
			hdf['images'].attrs['movie_vmin'] = vmin2
			hdf['images'].attrs['movie_vmax'] = vmax2
			hdf.flush()


		logger.info("Creating movie...")
		with plt.style.context('dark_background'):
			fig, ax = plt.subplots(1, 4, figsize=(20, 6.8))

			# Colormap to use for FFIs:
			cmap = plt.get_cmap('viridis')
			cmap.set_bad('k', 1.0) # FIXME: Does not work with plot_image

			# Colormap for Flags:
			viridis = plt.get_cmap('Dark2')
			newcolors = viridis(np.linspace(0, 1, 4))
			newcolors[:1, :] = np.array([1, 1, 1, 1])
			cmap_flags = ListedColormap(newcolors)

			imgs = [0,0,0,0]
			imgs[0] = plot_image(dummy_img, ax=ax[0], scale='sqrt', vmin=vmin, vmax=vmax, title='Original Image', xlabel=None, ylabel=None, cmap=cmap, make_cbar=True)
			imgs[1] = plot_image(dummy_img, ax=ax[1], scale='sqrt', vmin=vmin, vmax=vmax, title='Background', xlabel=None, ylabel=None, cmap=cmap, make_cbar=True)
			imgs[2] = plot_image(dummy_img, ax=ax[2], scale='sqrt', vmin=vmin2, vmax=vmax2, title='Background subtracted', xlabel=None, ylabel=None, cmap=cmap, make_cbar=True)
			imgs[3] = plot_image(dummy_img, ax=ax[3], scale='linear', vmin=-0.5, vmax=3.5, title='Pixel Flags', xlabel=None, ylabel=None, cmap=cmap_flags, make_cbar=True, clabel='Flags', cbar_ticks=[0,1,2,3], cbar_ticklabels=['None','Not used','Man Excl','Shenan'])

			for a in ax:
				a.set_xticks([])
				a.set_yticks([])

			fig.suptitle("to come\nt=???????", fontsize=15)
			fig.set_tight_layout('tight')
			fig.subplots_adjust(top=0.85)
			set_copyright(fig)

			writer = animation.FFMpegWriter(fps=fps)
			with writer.saving(fig, output_file, dpi):
				for k in trange(numfiles, **tqdm_settings):
					dset_name = '%04d' % k
					flux0 = np.asarray(hdf['images/' + dset_name])
					bkg = np.asarray(hdf['backgrounds/' + dset_name])

					# Plot original image, background and new image:
					imgs[0].set_data(flux0 + bkg)
					imgs[1].set_data(bkg)
					imgs[2].set_data(flux0)

					# Background Shenanigans flags, if available:
					if 'pixel_flags/' + dset_name in hdf:
						img = np.asarray(hdf['pixel_flags/' + dset_name])

						flags = np.zeros_like(img, dtype='uint8')
						flags[img & PixelQualityFlags.NotUsedForBackground != 0] = 1
						flags[img & PixelQualityFlags.ManualExclude != 0] = 2
						flags[img & PixelQualityFlags.BackgroundShenanigans != 0] = 3

						imgs[3].set_data(flags)

					# Update figure title with cadence information;
					fig.suptitle("Sector {sector:d}, Camera {camera:d}, CCD {ccd:d}\ndset={dset:s}, cad={cad:d}, t={time:.6f}".format(
						sector=sector,
						camera=camera,
						ccd=ccd,
						dset=dset_name,
						cad=cadenceno[k],
						time=time[k]
					), fontsize=15)

					writer.grab_frame()

			plt.close(fig)

	return output_file

#------------------------------------------------------------------------------
def make_combined_movie(input_dir, fps=15, dpi=100, overwrite=False):
	"""
	Create animation of the combined contents of all HDF5 files in a directoru,
	produced by the photometry pipeline.

	Parameters:
		input_dir (string): Path to the directory with HDF5 files to produce movie from.
		fps (integer): Frames per second of generated movie. Default=15.
		dpi (integer): DPI of the movie. Default=100.
		overwrite (boolean): Overwrite existing MP4 files? Default=False.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)
	logger.info("Processing '%s'", input_dir)

	camccd = [
		(1,1), (1,2), (2,1), (2,2), (3,1), (3,2), (4,1), (4,2),
		(1,4), (1,3), (2,4), (2,3), (3,4), (3,3), (4,4), (4,3)
	]

	# Find the sectors that are available:
	# TODO: Could we change this so we don't have to parse the filenames?
	sectors = []
	for fname in find_hdf5_files(input_dir):
		m = re.match(r'^sector(\d+)_camera\d_ccd\d\.hdf5$', os.path.basename(fname))
		if int(m.group(1)) not in sectors:
			sectors.append(int(m.group(1)))

	# Create one movie per found sector:
	for sector in sectors:
		# Define the output file, and overwrite it if needed:
		output_file = os.path.join(input_dir, 'sector{sector:03d}_combined.mp4'.format(sector=sector))
		if os.path.exists(output_file):
			if overwrite:
				logger.debug("Deleting existing output file")
				os.remove(output_file)
			else:
				logger.info("Movie file already exists")
				return output_file


		try:
			hdf = [None]*16
			vmin = np.full(16, np.NaN)
			vmax = np.full(16, np.NaN)
			for k, (camera, ccd) in enumerate(camccd):
				hdf_file = find_hdf5_files(input_dir, sector=sector, camera=camera, ccd=ccd)
				if hdf_file:
					hdf[k] = h5py.File(hdf_file[0], 'r', libver='latest')

					numfiles = len(hdf[k]['images'])
					dummy_img = np.full_like(hdf[k]['images/0000'], np.NaN)
					time = np.asarray(hdf[k]['time'])
					cadenceno = np.asarray(hdf[k]['cadenceno'])

					# Load the image scales if they have already been calculated:
					vmin[k] = hdf[k]['images'].attrs.get('movie_vmin', 0)
					vmax[k] = hdf[k]['images'].attrs.get('movie_vmax', 500)

			# Summarize the different CCDs into common values:
			vmin = np.nanmedian(vmin)
			vmax = np.nanmedian(vmax)

			logger.info("Creating movie...")
			with plt.style.context('dark_background'):

				fig, axes = plt.subplots(2, 8, figsize=(25, 6.8))

				cmap = plt.get_cmap('viridis')
				cmap.set_bad('k', 1.0)

				imgs = [None]*16
				for k, ax in enumerate(axes.flatten()):
					imgs[k] = plot_image(dummy_img, ax=ax, scale='sqrt', vmin=vmin, vmax=vmax, xlabel=None, ylabel=None, cmap=cmap, make_cbar=False)
					ax.set_xticks([])
					ax.set_yticks([])

				fig.suptitle("to come\nt=???????", fontsize=15)
				fig.set_tight_layout('tight')
				fig.subplots_adjust(top=0.85)
				set_copyright(fig)

				writer = animation.FFMpegWriter(fps=fps)
				with writer.saving(fig, output_file, dpi):
					for k in trange(numfiles):
						dset_name = '%04d' % k

						for k in range(16):
							if hdf[k] is None: continue
							img = np.asarray(hdf[k]['images/' + dset_name])
							img += np.asarray(hdf[k]['backgrounds/' + dset_name])

							# TODO: This can't always be right!
							img = np.rot90(img)

							imgs[k].set_data(img)

						# Update figure title with cadence information;
						fig.suptitle("Sector {sector:d}, Camera {camera:d}, CCD {ccd:d}\ndset={dset:s}, cad={cad:d}, t={time:.6f}".format(
							sector=sector,
							camera=camera,
							ccd=ccd,
							dset=dset_name,
							cad=cadenceno[k],
							time=time[k]
						), fontsize=15)

						writer.grab_frame()

				plt.close(fig)

		except:
			raise

		finally:
			for k in range(16):
				if hdf[k] is not None:
					hdf[k].close()

	return output_file

#------------------------------------------------------------------------------
if __name__ == '__main__':
	multiprocessing.freeze_support() # for Windows support

	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Create movie of TESS camera.')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('-o', '--overwrite', help='Overwrite existing files.', action='store_true')
	parser.add_argument('-j', '--jobs', help='Maximal number of jobs to run in parallel.', type=int, default=0, nargs='?')
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
	console = TqdmLoggingHandler()
	console.setFormatter(formatter)
	logger_parent = logging.getLogger('photometry')
	logger_parent.setLevel(logging_level)
	if not logger.hasHandlers(): logger.addHandler(console)
	if not logger_parent.hasHandlers(): logger_parent.addHandler(console)

	# If the user provided the path to a single directory,
	# find all the HDF5 files in that directory and process them:
	run_full_directory = None
	if len(args.files) == 1 and os.path.isdir(args.files[0]):
		run_full_directory = args.files[0]
		args.files = find_hdf5_files(run_full_directory)
		logger.info("Found %d HDF5 files in directory '%s'", len(args.files), run_full_directory)


	tqdm_settings = {
		'disable': not logger.isEnabledFor(logging.INFO),
		'total': len(args.files),
		'dynamic_ncols': True
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

	if run_full_directory:
		fname = make_combined_movie(run_full_directory, overwrite=args.overwrite, fps=args.fps, dpi=args.dpi)
		logger.info("Created movie: %s", fname)
