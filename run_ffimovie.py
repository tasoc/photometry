#!/usr/bin/env python3
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
import copy
from tqdm import tqdm, trange
from photometry.plots import plt, plot_image
from matplotlib import animation
from matplotlib.colors import ListedColormap
from photometry.quality import PixelQualityFlags
from photometry.utilities import find_hdf5_files, TqdmLoggingHandler, to_tuple

#--------------------------------------------------------------------------------------------------
def set_copyright(fig, xpos=0.01, ypos=0.99, fontsize=12):
	plt.text(ypos, xpos, 'Created by TASOC',
		verticalalignment='bottom', horizontalalignment='right',
		transform=fig.transFigure,
		color='0.3', fontsize=fontsize)

#--------------------------------------------------------------------------------------------------
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
	tqdm_settings = {'disable': None if logger.isEnabledFor(logging.INFO) else True}
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
	save_image_scales = False
	with h5py.File(hdf_file, 'r') as hdf:
		# Load the image scales if they have already been calculated:
		vmin = hdf['backgrounds'].attrs.get('movie_vmin')
		vmax = hdf['backgrounds'].attrs.get('movie_vmax')
		vmin2 = hdf['images'].attrs.get('movie_vmin')
		vmax2 = hdf['images'].attrs.get('movie_vmax')

		# Calculate scales to use for plotting the images:
		if not vmin:
			logger.info("Calculating image scales...")
			numfiles = len(hdf['images'])
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
			save_image_scales = True

	# If needed, reopen the file for saving the attributes:
	if save_image_scales:
		with h5py.File(hdf_file, 'r+') as hdf:
			# Save image scales to HDF5 file:
			hdf['backgrounds'].attrs['movie_vmin'] = vmin
			hdf['backgrounds'].attrs['movie_vmax'] = vmax
			hdf['images'].attrs['movie_vmin'] = vmin2
			hdf['images'].attrs['movie_vmax'] = vmax2
			hdf.flush()

	# We should now be ready for creating the movie, reopen the file as readonly:
	logger.info("Creating movie...")
	with h5py.File(hdf_file, 'r') as hdf:
		numfiles = len(hdf['images'])
		dummy_img = np.full_like(hdf['images/0000'], np.NaN)
		time = np.asarray(hdf['time'])
		cadenceno = np.asarray(hdf['cadenceno'])
		sector = hdf['images'].attrs.get('SECTOR')
		camera = hdf['images'].attrs.get('CAMERA')
		ccd = hdf['images'].attrs.get('CCD')

		with plt.style.context('dark_background'):
			plt.rc('axes', titlesize=15)

			fig, ax = plt.subplots(1, 4, figsize=(20, 6.8), dpi=dpi)

			# Colormap to use for FFIs:
			cmap = copy.copy(plt.get_cmap('viridis'))
			cmap.set_bad('k', 1.0)

			# Colormap for Flags:
			viridis = plt.get_cmap('Dark2')
			newcolors = viridis(np.linspace(0, 1, 4))
			newcolors[:1, :] = np.array([1, 1, 1, 1])
			cmap_flags = ListedColormap(newcolors)

			imgs = [None]*4
			imgs[0] = plot_image(dummy_img, ax=ax[0], scale='sqrt', vmin=vmin, vmax=vmax, title='Original Image', cmap=cmap, cbar='bottom', cbar_pad=0.05)
			imgs[1] = plot_image(dummy_img, ax=ax[1], scale='sqrt', vmin=vmin, vmax=vmax, title='Background', cmap=cmap, cbar='bottom', cbar_pad=0.05)
			imgs[2] = plot_image(dummy_img, ax=ax[2], scale='sqrt', vmin=vmin2, vmax=vmax2, title='Background subtracted', cmap=cmap, cbar='bottom', cbar_pad=0.05)
			imgs[3] = plot_image(dummy_img, ax=ax[3], scale='linear', vmin=-0.5, vmax=3.5, title='Pixel Flags', cmap=cmap_flags, cbar='bottom', cbar_pad=0.05, clabel='Flags', cbar_ticks=[0,1,2,3], cbar_ticklabels=['None','Not used','Man Excl','Shenan'])

			for a in ax:
				a.set_xticks([])
				a.set_yticks([])

			figtext = fig.suptitle("to come\nt=???????", fontsize=16)
			fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.03, wspace=0.05)
			set_copyright(fig)

			metadata = {
				'title': f'TESS Sector {sector:d}, Camera {camera:d}, CCD {ccd:d}',
				'artist': 'TASOC'
			}

			# Set up the writer (FFMpeg)
			WriterClass = animation.writers['ffmpeg']
			writer = WriterClass(fps=fps, codec='h264', bitrate=-1, metadata=metadata)
			with writer.saving(fig, output_file, dpi):
				for k in trange(numfiles, **tqdm_settings):
					dset = f'{k:04d}'
					flux0 = np.asarray(hdf['images/' + dset])
					bkg = np.asarray(hdf['backgrounds/' + dset])

					# Plot original image, background and new image:
					imgs[0].set_data(flux0 + bkg)
					imgs[1].set_data(bkg)
					imgs[2].set_data(flux0)

					# Background Shenanigans flags, if available:
					if 'pixel_flags/' + dset in hdf:
						img = np.asarray(hdf['pixel_flags/' + dset])

						flags = np.zeros_like(img, dtype='uint8')
						flags[img & PixelQualityFlags.NotUsedForBackground != 0] = 1
						flags[img & PixelQualityFlags.ManualExclude != 0] = 2
						flags[img & PixelQualityFlags.BackgroundShenanigans != 0] = 3

						imgs[3].set_data(flags)

					# Update figure title with cadence information;
					figtext.set_text(f"Sector {sector:d}, Camera {camera:d}, CCD {ccd:d}\ndset={dset:s}, cad={cadenceno[k]:d}, t={time[k]:.6f}")

					writer.grab_frame()

			plt.close(fig)

	return output_file

#--------------------------------------------------------------------------------------------------
def make_combined_movie(input_dir, mode='images', sectors=None, fps=15, dpi=100, overwrite=False):
	"""
	Create animation of the combined contents of all HDF5 files in a directory,
	produced by the photometry pipeline.

	Parameters:
		input_dir (str): Path to the directory with HDF5 files to produce movie from.
		mode (str): Which images to show.
			Choices are `'originals'`, `'images'`, `'backgrounds'` or `'flags'`.
			Default=images.
		sectors: Sector or list of sectors to generate combined movies for.
		fps (int): Frames per second of generated movie. Default=15.
		dpi (int): DPI of the movie. Default=100.
		overwrite (bool): Overwrite existing MP4 files? Default=False.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	# Basic input checks:
	if mode not in ('originals', 'images', 'backgrounds', 'flags'):
		raise ValueError("Invalid MODE specified")

	logger = logging.getLogger(__name__)
	tqdm_settings = {'disable': None if logger.isEnabledFor(logging.INFO) else True}
	logger.info("Processing '%s'", input_dir)

	camccdrot = [
		(1,3,1), (1,2,3), (2,3,1), (2,2,3), (3,1,1), (3,4,3), (4,1,1), (4,4,3),
		(1,4,1), (1,1,3), (2,4,1), (2,1,3), (3,2,1), (3,3,3), (4,2,1), (4,3,3)
	]

	# Find the sectors that are available:
	if sectors is None:
		sectors = []
		for fname in find_hdf5_files(input_dir):
			# Load the sector number from HDF5 file attributes:
			with h5py.File(fname, 'r') as hdf:
				s = hdf['images'].attrs.get('SECTOR')

			if s is not None and int(s) not in sectors:
				sectors.append(int(s))
			else:
				# If the attribute doesn't exist try to find it from
				# parsing the file name:
				m = re.match(r'^sector(\d+)_camera\d_ccd\d\.hdf5$', os.path.basename(fname))
				if int(m.group(1)) not in sectors:
					sectors.append(int(m.group(1)))

	# Create one movie per found sector:
	for sector in sectors:
		# Define the output file, and overwrite it if needed:
		output_file = os.path.join(input_dir, f'sector{sector:03d}_combined_{mode:s}.mp4')
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
			for k, (camera, ccd, rot) in enumerate(camccdrot):
				hdf_file = find_hdf5_files(input_dir, sector=sector, camera=camera, ccd=ccd)
				if hdf_file:
					hdf[k] = h5py.File(hdf_file[0], 'r')

					numfiles = len(hdf[k]['images'])
					dummy_img = np.full_like(hdf[k]['images/0000'], np.NaN)
					time = np.asarray(hdf[k]['time'])
					cadenceno = np.asarray(hdf[k]['cadenceno'])

					# Load the image scales if they have already been calculated:
					if mode == 'backgrounds':
						vmin[k] = hdf[k]['backgrounds'].attrs.get('movie_vmin', 0)
						vmax[k] = hdf[k]['backgrounds'].attrs.get('movie_vmax', 500)
					elif mode == 'images' or mode == 'originals':
						vmin[k] = hdf[k]['images'].attrs.get('movie_vmin', 0)
						vmax[k] = hdf[k]['images'].attrs.get('movie_vmax', 500)

			# Summarize the different CCDs into common values:
			vmin = np.nanpercentile(vmin, 25.0)
			vmax = np.nanpercentile(vmax, 75.0)

			logger.info("Creating combined %s movie...", mode)
			with plt.style.context('dark_background'):
				fig, axes = plt.subplots(2, 8, figsize=(25, 6.8), dpi=dpi)

				cmap = copy.copy(plt.get_cmap('viridis'))
				cmap.set_bad('k', 1.0)

				# Colormap for Flags:
				viridis = plt.get_cmap('Dark2')
				newcolors = viridis(np.linspace(0, 1, 4))
				newcolors[:1, :] = np.array([1, 1, 1, 1])
				cmap_flags = ListedColormap(newcolors)

				imgs = [None]*16
				for k, ax in enumerate(axes.flatten()):
					if mode == 'flags':
						imgs[k] = plot_image(dummy_img, ax=ax, scale='linear', vmin=-0.5, vmax=4.5, cmap=cmap_flags)
					else:
						imgs[k] = plot_image(dummy_img, ax=ax, scale='sqrt', vmin=vmin, vmax=vmax, cmap=cmap)
					ax.set_xticks([])
					ax.set_yticks([])

				figtext = fig.suptitle("to come\nt=???????", fontsize=16)
				fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.05, wspace=0.05, hspace=0.05)
				set_copyright(fig)

				metadata = {
					'title': f'TESS Sector {sector:d}, {mode:s}',
					'artist': 'TASOC'
				}

				# Set up the writer (FFMpeg)
				WriterClass = animation.writers['ffmpeg']
				writer = WriterClass(fps=fps, codec='h264', bitrate=-1, metadata=metadata)
				with writer.saving(fig, output_file, dpi):
					for i in trange(numfiles, **tqdm_settings):
						dset = f'{i:04d}'

						for k in range(16):
							if hdf[k] is None:
								continue

							# Background Shenanigans flags, if available:
							if mode == 'flags':
								flags = np.asarray(hdf[k]['pixel_flags/' + dset])
								img = np.zeros_like(flags, dtype='uint8')
								img[flags & PixelQualityFlags.NotUsedForBackground != 0] = 1
								img[flags & PixelQualityFlags.ManualExclude != 0] = 2
								img[flags & PixelQualityFlags.BackgroundShenanigans != 0] = 3
							elif mode == 'originals':
								img = np.asarray(hdf[k]['images/' + dset])
								img += np.asarray(hdf[k]['backgrounds/' + dset])
							else:
								img = np.asarray(hdf[k][mode + '/' + dset])

							# Rotate the image:
							cam, ccd, rot = camccdrot[k]
							img = np.rot90(img, rot)

							# Update the image:
							imgs[k].set_data(img)

						# Update figure title with cadence information:
						figtext.set_text(f"Sector {sector:d} - {mode:s}\ndset={dset:s}, cad={cadenceno[i]:d}, t={time[i]:.6f}")

						writer.grab_frame()

				plt.close(fig)

		except: # noqa: E722
			raise

		finally:
			for k in range(16):
				if hdf[k] is not None:
					hdf[k].close()

	return output_file

#--------------------------------------------------------------------------------------------------
def main():
	multiprocessing.freeze_support() # for Windows support

	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Create movies of TESS cameras.')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('-o', '--overwrite', help='Overwrite existing files.', action='store_true')
	parser.add_argument('-j', '--jobs', help='Maximal number of jobs to run in parallel.', type=int, default=0, nargs='?')
	parser.add_argument('--fps', help='Frames per second of generated movie.', type=int, default=15, nargs='?')
	parser.add_argument('--dpi', help='DPI of generated movie.', type=int, default=100, nargs='?')
	parser.add_argument('--sector', type=int, default=None, action='append', help='TESS Sector. Default is to run all sectors.')
	parser.add_argument('files', help='Directory or HDF5 file to create movie from.', nargs='+')
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
	if not logger.hasHandlers():
		logger.addHandler(console)
	if not logger_parent.hasHandlers():
		logger_parent.addHandler(console)

	# If the user provided the path to a single directory,
	# find all the HDF5 files in that directory and process them:
	run_full_directory = None
	if len(args.files) == 1 and os.path.isdir(args.files[0]):
		run_full_directory = args.files[0]
		args.files = find_hdf5_files(run_full_directory, sector=to_tuple(args.sector))
		logger.info("Found %d HDF5 files in directory '%s'", len(args.files), run_full_directory)

	tqdm_settings = {
		'disable': None if logger.isEnabledFor(logging.INFO) else True,
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
	make_movie_wrapper = functools.partial(
		make_movie,
		fps=args.fps,
		dpi=args.dpi,
		overwrite=args.overwrite
	)

	# Process the files on at a time, in parallel if needed:
	for fname in tqdm(m(make_movie_wrapper, args.files), **tqdm_settings):
		logger.info("Created movie: %s", fname)

	if run_full_directory and len(args.files) > 0:
		# Make wrapper function with all settings:
		make_combined_movie_wrapper = functools.partial(
			make_combined_movie,
			run_full_directory,
			sectors=args.sector,
			fps=args.fps,
			dpi=args.dpi,
			overwrite=args.overwrite
		)

		for fname in m(make_combined_movie_wrapper, ('backgrounds', 'originals', 'images', 'flags')):
			logger.info("Created movie: %s", fname)

	# Close workers again:
	if threads > 1:
		pool.close()
		pool.join()

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	main()
