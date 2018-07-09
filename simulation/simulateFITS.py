#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 17:21:56 2018

.. codeauthor:: Jonas Svenstrup Hansen <jonas.svenstrup@gmail.com>
"""

import os
import numpy as np
import random
from astropy.io import fits
from astropy.table import Table, Column
from astropy.wcs import WCS
from copy import deepcopy
import sqlite3
from multiprocessing import Pool

# Import stuff from the photometry directory:
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__))))

from photometry.psf import PSF
from photometry.utilities import mag2flux, add_proper_motion
#from photometry.plots import plot_image


class simulateFITS(object):
	def __init__(self, Nstars = 2, Ntimes = 2,
			save_images=True, overwrite_images=True,
			include_jitter=True, include_noise=True,
			include_bkg=True, inaccurate_catalog=False,
			Nvariables=1, multiprocess=True):
		"""
		Simulate FITS images with stars, background and noise.

		The purpose of this code is not to replace SPyFFI, but to supplement it
		in making simulated images simpler and more customizable. The aim is to
		supply simulated images that can illustrate the performance of various
		photometry methods in the photometry pipeline.

		Parameters:
			Nstars (int): Number of stars in image. Default is 2.
			Ntimes (int): Number of time steps in timeseries. Default is 2.
			save_images (boolean): True if images and catalog should be saved.
			Default is True.
			overwrite_images (boolean): True if image and catalog files should
			be overwritten. Default is True.
			include_jitter (boolean): True if jitter is to be used (default).
			include_noise (boolean): True if noise is to be added to the image (default).
			include_bkg (boolean): True if background is to be added to the image (default).
			inaccurate_catalog (boolean): True if catalog positional inaccuracies at Gaia level should be included (default is False).
			Nvariables (int): Number of stars that are variable (default is 1).
			multiprocess (boolean): True if multiprocessing is to be used (default).

		Output:
			The output FITS images are saved to a subdirectory images in the
			parent directory specified by the environment variable
			TESSPHOT_INPUT. An ASCII file named catalog.txt.gz with the
			simulated catalog, prepared in the format read by
			`prepare_photometry`, is written to this parent directory.

		Example:
			Default use. Write 5 FITS images of shape 200x200px with 5 stars in
			them to 5 separate files in a subdirectory called images in the
			directory specified by the TESSPHOT_INPUT environment variable:

			>>> sim = simulateFITS()

			Print catalog. This call does not save images or a catalog file,
			but will just print the catalog.

			>>> sim = simulateFITS(save_images=False)
			      ra             decl      prop_mot_ra prop_mot_dec      row           col        tmag
			-------------- --------------- ----------- ------------ ------------- ------------- -------
			0.029851440263   0.68646339125         0.0          0.0   117.6794385 5.11738975937 12.5627
			 0.42553055972   1.00578707012         0.0          0.0 172.420640592  72.948095952 14.4416
			 1.32855128151  0.643677266712         0.0          0.0 110.344674294 227.751648259 11.5697
			 1.23209768011 0.0831070155292         0.0          0.0 14.2469169479 211.216745161 13.4081
			0.451164111667  0.512332559648         0.0          0.0 87.8284387967  77.342419143 7.69913


		.. codeauthor:: Jonas Svenstrup Hansen <jonas.svenstrup@gmail.com>
		"""
		self.Nstars = np.int(Nstars) # Number of stars in image
		self.Ntimes = np.int(Ntimes) # Number of images in time series
		self.save_images = save_images # True if images+catalog should be saved
		self.overwrite_images = overwrite_images # True if overwrite in saving
		self.include_jitter = include_jitter
		self.include_noise = include_noise
		self.include_bkg = include_bkg
		self.inaccurate_catalog = inaccurate_catalog
		self.Nvariables = Nvariables # Number of oscillating stars
		self.multiprocess = multiprocess


		# Get output directory from enviroment variable:
		self.output_folder = os.environ.get('TESSPHOT_INPUT',
									os.path.abspath('.'))

		# Set image parameters:
		self.pixel_scale = 21.0 # Size of single pixel in arcsecs
		self.Nrows = 256
		self.Ncols = 256
		self.stamp = (0,self.Nrows,0,self.Ncols)
		self.coord_zero_point = [0.,0.] # Zero point
		self.sector = 0
		self.camera = 1
		self.ccd = 1
		self.TmagLow = 7.
		self.TmagHigh = 16.
		self.gain = 5.3 # electrons/count, CCD output gain (ETE6 estimate)

		# Define time stamps:
		self.exposure_time = 1800. # 30 min
		self.times = self.make_times(cadence=self.exposure_time)
		self.reference_time = 2457000.+np.mean(self.times)
		self.epoch = self.reference_time - 2455200.5

		# Make WCS solution parameters:
		self.w = WCS(naxis=2)
		self.w.wcs.crpix = [0,0]
		self.w.wcs.cdelt = [self.pixel_scale/3600, self.pixel_scale/3600]
		self.w.wcs.crval = self.coord_zero_point # [0.,0.]
		self.w.wcs.ctype = ["RA---AIR", "DEC--AIR"]
		self.header = self.w.to_header()

		# Set random number generator seeds:
		random.seed(0)
		np.random.seed(0)

		# Make catalog:
		self.catalog = self.make_catalog()

		# Save catalog to txt.gz file:
		self.make_catalog_file(self.catalog)

		# Generate sqlite catalog file from catalog.txt.gzfile:
		self.create_catalog(self.sector, self.camera, self.ccd)

		# Generate TODO list for each photometry method:
		for method in ['linpsf','aperture','psf']:
			self.create_todo(sector=self.sector, method=method, filename='todo_'+method+'.sqlite')

		# Apply time-independent changes to catalog:
		if self.inaccurate_catalog:
			self.catalog = self.apply_inaccurate_catalog(self.catalog)
		master_catalog = deepcopy(self.catalog)

		# Generate jitter array:
		if include_jitter:
			self.jitter = self.apply_jitter()
			print("Jitter included.")
		else:
			self.jitter = np.zeros([self.Ntimes, 2])
			print("Jitter excluded. Jitter array set to 0.")
		print("Saving jitter array to jitter.txt")
		np.savetxt(os.path.join(self.output_folder, "jitter.txt"), self.jitter)

		# Generate light curves:
		if self.Nvariables > 0:
			self.light_curves, params = self.generate_light_curves()
		else:
			self.light_curves = np.zeros([self.Nstars, self.Ntimes])
			params = ''
			print("No variable stars included.")
		print("Saving light curve parameters to light_curve_params")
		np.savetxt(os.path.join(self.output_folder, "light_curve_params.txt"), params)
		print("Saving light curves to light_curves.txt")
		np.savetxt(os.path.join(self.output_folder, "light_curves.txt"), self.light_curves)

		# Loop through the time steps:
		for t, timestep in enumerate(self.times):
			print("Making time step "+str(t)+" out of "+str(self.Ntimes))

			# Set catalog to master_catalog:
			self.catalog = master_catalog

			# Make stars from catalog:
			stars = self.make_stars(t)

			# Make uniform background:
			bkg = self.make_background()

			# Make Gaussian noise:
			noise = self.make_noise()

			# Sum image from its parts:
			img = stars
			if self.include_noise:
				img += noise
			if self.include_bkg:
				img += bkg

			if self.save_images:
				# Write img to FITS file:
				self.make_fits(img, timestep, t)


	def make_times(self, cadence = 1800.0):
		"""
		Make the time steps.

		Parameters:
			cadence (float): Time difference between frames. Default is 1800
			seconds corresponding the 30 minutes in long cadence data from
			TESS.

		Returns:
			times (numpy array): Timestamps of all images to be made.
		"""
		# Define time stamps:
		times = np.arange(0, cadence*self.Ntimes, cadence)

		# Force correct number of time steps:
		# (this is necessary because cadence is not an int)
		if len(times) > self.Ntimes:
			times = times[0:10]

		return times


	def make_catalog(self):
		"""
		Make catalog of stars in the current image.

		The table contains the following columns:
		 * starid: Identifier. Starts at 0.
		 * row:    Pixel row in image.
		 * col:    Pixel column in image.
		 * tmag:   TESS magnitude.

		Returns:
			catalog (`astropy.table.Table`): Table with stars in the current
			image.
		"""
		# Set star identification:
		starids = np.arange(self.Nstars, dtype=int)

		# Set buffer pixel size around edge where not to put stars:
		bufferpx = 3

		# Draw uniform row positions:
		starrows =  np.random.uniform(bufferpx, self.Nrows-bufferpx, self.Nstars)

		# Draw uniform column positions:
		starcols =  np.random.uniform(bufferpx, self.Ncols-bufferpx, self.Nstars)

		# Draw stellar magnitudes:
		starmag = np.random.triangular(self.TmagLow, self.TmagHigh, self.TmagHigh, self.Nstars)

		# Collect star parameters in list for catalog:
		cat = [starids, starrows, starcols, starmag]

		# Make astropy table with catalog:
		return Table(
			cat,
			names=('starid', 'row', 'col', 'tmag'),
			dtype=('int64', 'float64', 'float64', 'float32')
		)


	def make_catalog_file(self, catalog, fname='catalog', compress=True):
		"""
		Write simulated catalog to an ASCII file in the format used by
		`prepare_photometry`.

		The name of each column in the catalog is written as a header in the
		first line of the catalog file. The following columns will be written:
		 * ra:            Right ascension coordinate.
		 * decl:          Declination coordinate.
		 * prop_mot_ra:   Proper motion in right ascension. Is set to 0.
		 * prop_mot_decl: Proper motion in declination. Is set to 0.
		 * row:           Pixel row in full frame image.
		 * col:           Pixel column in full frame image.
		 * tmag:          TESS magnitude.

		Parameters:
			catalog (`astropy.table.Table`): Table with stars in the current
			image. Columns must be starid, row, col, tmag.
			fname (string): Filename of catalog. Default is catalog.
			compress (boolean): True if catalog txt file is to be compressed.
			Default is True.
		"""

		# Remove starid in input catalog:
		catalog.remove_column('starid')

		# Set ra and dec from pixel coordinates using WCS solution:
		ra, decl = self.w.all_pix2world(catalog['col'],catalog['row'],0,ra_dec_order=True)

		# Set proper motion to zero:
		prop_mot_ra = np.zeros_like(catalog['tmag'])
		prop_mot_dec = np.zeros_like(catalog['tmag'])

		# Define extra columns:
		Col_ra = Column(data=ra, name='ra', dtype=np.float64)
		Col_decl = Column(data=decl, name='decl', dtype=np.float64)
		Col_prop_mot_ra = Column(data=prop_mot_ra, name='prop_mot_ra',
							dtype=np.float64)
		Col_prop_mot_decl = Column(data=prop_mot_dec, name='prop_mot_dec',
							dtype=np.float64)

		# Add extra columns to catalog:
		catalog.add_columns([Col_ra, Col_decl,
							Col_prop_mot_ra, Col_prop_mot_decl],
							indexes=[0,0,0,0])

		if self.save_images:
			# Convert catalog to numpy array:
			catalog_out = np.asarray(catalog)

			if self.overwrite_images:
				# Directory with filename of catalog output file:
				if compress:
					fextension = '.txt.gz'
				else:
					fextension = '.txt'
				txtfiledir = os.path.join(self.output_folder, fname+fextension)

				# Write catalog to txt file:
				print("Writing catalog to file: "+txtfiledir)
				np.savetxt(txtfiledir, catalog_out,
							delimiter='\t',
							header='    '.join(catalog.colnames))
			else:
				pass
		else:
			pass

		# Print the catalog:
		print(catalog)


	def create_catalog(self, sector, camera, ccd):
		""" Original function by Rasmus Handberg. Create sqlite file """
		input_folder = os.environ['TESSPHOT_INPUT']

		# We need a list of when the sectors are in time:
		sector_reference_time = 2457827.0 + 13.5

		# Load the catalog from file:
		cat = np.genfromtxt(os.path.join(input_folder, 'catalog.txt.gz'), skip_header=1, usecols=(0,1,2,3,6), dtype='float64')
		if cat.ndim == 1:
			cat = np.expand_dims(cat, axis=0)
		cat = np.column_stack((np.arange(1, cat.shape[0]+1, dtype='int64'), cat))

		# Create SQLite file:
		catalog_file = os.path.join(input_folder, 'catalog_camera{0:d}_ccd{1:d}.sqlite'.format(camera, ccd))
		if os.path.exists(catalog_file): os.remove(catalog_file)
		conn = sqlite3.connect(catalog_file)
		conn.row_factory = sqlite3.Row
		cursor = conn.cursor()

		cursor.execute("""CREATE TABLE catalog (
			starid BIGINT PRIMARY KEY NOT NULL,
			ra DOUBLE PRECISION NOT NULL,
			decl DOUBLE PRECISION NOT NULL,
			ra_J2000 DOUBLE PRECISION NOT NULL,
			decl_J2000 DOUBLE PRECISION NOT NULL,
			tmag REAL NOT NULL
		);""")

		for row in cat:
			# Add the proper motion to each coordinate:
			ra, dec = add_proper_motion(row[1], row[2], row[3], row[4], sector_reference_time, epoch=2000.0)

			# Save the coordinates in SQLite database:
			cursor.execute("INSERT INTO catalog (starid,ra,decl,ra_J2000,decl_J2000,tmag) VALUES (?,?,?,?,?,?);", (
				int(row[0]),
				ra,
				dec,
				row[1],
				row[2],
				row[5]
			))

		cursor.execute("CREATE UNIQUE INDEX starid_idx ON catalog (starid);")
		cursor.execute("CREATE INDEX ra_dec_idx ON catalog (ra, decl);")

		# Add settings table:
		cursor.execute("""CREATE TABLE settings (
			sector INT NOT NULL,
			camera INT NOT NULL,
			ccd INT NOT NULL,
			reference_time DOUBLE PRECISION NOT NULL,
			epoch DOUBLE PRECISION NOT NULL,
			coord_buffer DOUBLE PRECISION NOT NULL,
			footprint TEXT NOT NULL
		);""")

		# Fill out settings table using arbitrary values for coord_buffer and footprint:
		cursor.execute("INSERT INTO SETTINGS (sector,camera,ccd,reference_time,epoch,coord_buffer,footprint) VALUES (?,?,?,?,?,?,?);", (
			self.sector,
			self.camera,
			self.ccd,
			self.reference_time,
			self.epoch,
			0.,
			"{256.99505428967007,6.459825950686631,255.0210912526594,-5.037571570922653,242.81585702674383,-2.6813989943029086,245.1973064137359,9.154168271446911}"
		))

		conn.commit()
		cursor.close()
		conn.close()


	def create_todo(self, sector, method, filename):
		"""Create the TODO list which is used by the pipeline to keep track of the
		targets that needs to be processed.

		Will create the file `todo.sqlite` in the `TESSPHOT_INPUT` directory.
		It will be overwritten if it already exists.

		Parameters:
			sector (integer): The TESS observing sector.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		input_folder = os.environ['TESSPHOT_INPUT']

		cat = np.genfromtxt(os.path.join(input_folder, 'catalog.txt.gz'), skip_header=1, usecols=(4,5,6), dtype='float64')
		if cat.ndim == 1:
			cat = np.expand_dims(cat, axis=0)
		cat = np.column_stack((np.arange(1, cat.shape[0]+1, dtype='int64'), cat))
		# Convert data to astropy table for further use:
		cat = Table(
			data=cat,
			names=('starid', 'x', 'y', 'tmag'),
			dtype=('int64', 'float64', 'float64', 'float32')
		)

		todo_file = os.path.join(input_folder, filename)
		if os.path.exists(todo_file): os.remove(todo_file)
		conn = sqlite3.connect(todo_file)
		cursor = conn.cursor()

		cursor.execute("""CREATE TABLE todolist (
			priority BIGINT NOT NULL,
			starid BIGINT NOT NULL,
			datasource TEXT NOT NULL DEFAULT 'ffi',
			camera INT NOT NULL,
			ccd INT NOT NULL,
			method TEXT DEFAULT NULL,
			status INT DEFAULT NULL,
			elaptime REAL DEFAULT NULL,
			x REAL,
			y REAL,
			tmag REAL
		);""")

		indx = (cat['tmag'] <= 20) & (cat['x'] > 0) & (cat['x'] < 2048) & (cat['y'] > 0) & (cat['y'] < 2048)
		cat = cat[indx]
		cat.sort('tmag')
		for pri, row in enumerate(cat):
			starid = int(row['starid'])
			tmag = float(row['tmag'])
			cursor.execute("INSERT INTO todolist (priority,starid,camera,ccd,method,x,y,tmag) VALUES (?,?,?,?,?,?,?,?);", (pri+1, starid, 1, 1, method, row['x'], row['y'], tmag))

		cursor.execute("CREATE UNIQUE INDEX priority_idx ON todolist (priority);")
		cursor.execute("CREATE INDEX status_idx ON todolist (status);")
		cursor.execute("CREATE INDEX starid_idx ON todolist (starid);")
		conn.commit()
		cursor.close()
		conn.close()

	#	logger.info("TODO done.")


	def apply_inaccurate_catalog(self, catalog):
		"""
		Modify the input catalog to simulate inaccurate catalog information
		independent of time.

		It is assumed that the right ascension and declination uncertainties
		apply directly to pixel row and column positions. Thus, the spacial
		transformation from spherical coordinates is neglected.

		Parameters:
			catalog (`astropy.table.Table`): Table with stars in the current
			image. Columns must be starid, row, col, tmag.

		Returns:
			catalog (`astropy.table.Table`): Table formatted like the catalog
			parameter, but with changes to its entries.
		"""

#		# Scatter of Gaia band to TESS band calibration (Stassun, 28 Jun 2017):
#		sigma_tmag = 0.015 # (magnitudes)
#
#		# Median RA std. in Gaia DR1 (Lindegren, 29 June 2016, Table 1):
#		sigma_RA = 0.254 # (milliarcsec)
#		sigma_col = self.pixel_scale * sigma_RA / 1e3
#
#		# Median DEC std. in Gaia DR1 (Lindegren, 29 June 2016, Table 1):
#		sigma_DEC = 0.233 # (milliarcsec)
#		sigma_row = self.pixel_scale * sigma_DEC / 1e3

#		# Scatter of Gaia band to TESS band calibration (Stassun, 28 Jun 2017):
#		sigma_tmag = 0.015 # (magnitudes)

		# Median RA std. in Gaia DR1 (Lindegren, 29 June 2016, Table 1):
		sigma_RA = 0.1 # (milliarcsec)
		sigma_col = self.pixel_scale * sigma_RA / 1e3

		# Median DEC std. in Gaia DR1 (Lindegren, 29 June 2016, Table 1):
		sigma_DEC = 0.1 # (milliarcsec)
		sigma_row = self.pixel_scale * sigma_DEC / 1e3

		# Loop through each star in the catalog:
		for star in range(len(catalog['tmag'])):
#			# Modify TESS magnitude:
#			catalog['tmag'][star] += random.gauss(0, sigma_tmag)

			# Modify column pixel positions:
			catalog['col'][star] += random.gauss(0, sigma_col)

			# Modify row pixel positions:
			catalog['row'][star] += random.gauss(0, sigma_row)

		return catalog


	def generate_light_curves(self):
		"""
		Generate light curves to scale the stellar flux with. Half of all will be constant.

		Returns:
			light_curves (numpy array, 2D): Light curves formatted to be multiplied with the constant star flux value, (Nstars,Ntimes).
			params (list): List of oscillation parameters used to generate the light curve, ([amplitudes, frequency, phase] for each star).
		"""
		# Indices of half the stars that are not constant:
		osc_star_indices = np.random.choice(self.Nstars, self.Nvariables)

		# Preallocate light curves array:
		light_curves = np.ones([self.Nstars, self.Ntimes])

		# Set oscillation parameters:
		amplitudes = 0.1*np.ones(self.Nvariables)
		frequencies = 3*1e-3*np.ones(self.Nvariables) # RGB star, solarlike osc.
		phases = np.array([random.uniform(0, self.exposure_time)
				for star in range(self.Nvariables)])
		params = [[amplitude, frequency, phase] \
				for amplitude, frequency, phase \
				in zip(amplitudes, frequencies, phases)]

		k = 0
		for star in range(self.Nstars):
			if star in osc_star_indices:
				amplitude = params[k][0]
				frequency = params[k][1]
				phase = params[k][2]
				light_curves[star,:] += amplitude*np.sin(frequency*self.times + phase)
				k += 1

		return light_curves, params


	def apply_variable_magnitudes(self, catalog, timestamp):
		"""
		Modify the input catalog to simulate variable stars.

		Parameters:
			catalog (`astropy.table.Table`): Table with stars in the current
			image. Columns must be starid, row, col, tmag.

		Returns:
			catalog (`astropy.table.Table`): Table formatted like the catalog
			parameter, but with changes to its entries.
		"""

		# TODO: Introduce some variation in the TESS magnitude here

		return catalog


	def apply_jitter(self, var=None):
		"""
		Apply crude large-scale jitter by adjusting catalog position.

		Parameters:
			var (float): Variance of 2D Gaussian distribution in pixels.
			Default is None which sets the variance to the value 1 arcsec from
			Sullivan et al (2015).


		Returns:
			catalog (numpy array): Jitter changes in each timestep (dim. 1) and
			dimension (dim. 2).
		"""

		# Set jitter scale:
		jitter_scale = 0.1

		# Define distribution properties:
		if var is None:
			var = jitter_scale*self.pixel_scale/3600
		cov_mat = np.array([[var, 0.],[0., var]])

		# Create jitter, the same for all stars:
		jitter = np.random.multivariate_normal([0,0], cov_mat, self.Ntimes)

		return jitter


	def make_stars(self, t, camera=1, ccd=1):
		"""
		Make stars for the image and append catalog with flux column.

		Parameters:
			t (int): Time loop index.
			camera (int): Kepler camera. Used to get PSF. Default is 1.
			ccd (int): Kepler CCD. Used to get PSF. Default is 1.


		Returns:
			stars (numpy array): Summed PRFs of stars in the image of the same
			shape as image.
		"""

		# Create PSF class instance:
		kpsf = PSF(camera=camera, ccd=ccd, stamp=self.stamp)

		# Make list with parameter numpy arrays for the pixel integrater:
		params = [
					np.array(
						[self.catalog['row'][i] + self.jitter[t][0],
						self.catalog['col'][i] + self.jitter[t][1],
						self.light_curves[i][t]*mag2flux(self.catalog['tmag'][i])]
					)
				for i in range(self.Nstars)
				]

		# Integrate stars to image:
		if self.multiprocess:
			img = np.zeros([self.Nrows, self.Ncols], dtype='float64')
			with Pool(8) as p:
				for i in p.imap_unordered(kpsf.integrate_single, params):
					img += i
			return img
		else:
			return kpsf.integrate_to_image(params, cutoff_radius=20)


	def make_background(self, bkg_level=337000.):
		"""
		Make a background for the image.

		Parameters:
			bkg_level (float): Background level of uniform background. Default
			is the median of an ETE6 image.

		Returns:
			bkg (numpy array): Background array of the same shape as image.
		"""

		# Apply background level by multiplying:
		return bkg_level * np.ones([self.Nrows, self.Ncols]) * self.gain


	def make_noise(self, sigma=52000.):
		"""
		Make Gaussian noise uniformily across the image.

		Parameters:
			sigma (float): Sigma parameter of Gaussian distribution for noise.
			Default is the standard deviation of an upper boundary sigma
			clipped (6 sigma) ETE6 image.

		Returns:
			noise (numpy array): Noise array of the same shape as image.
		"""

		# Preallocate noise array:
		noise = np.zeros([self.Nrows, self.Ncols])

		# Loop over each pixel:
		for row in range(self.Nrows):
			for col in range(self.Ncols):
				# Draw a random value from a Gaussian (normal) distribution:
				noise[row,col] = random.gauss(mu=0, sigma=sigma) * self.gain

		return noise


	def make_fits(self, img, timestamp, i, outdir=None):
		"""
		Write image to FITS file.

		Parameters:
			img (numpy array): Image to write to file.
			timestamp (float): Timestamp in seconds of image.
			i (int): Timestamp index that is used in filename.
		"""

		# Instantiate primary header data unit:
		hdu = fits.PrimaryHDU(data=img, header=self.header)

		# Add info to header as used by prepare_photometry.py:
		hdu.header['BJDREFI'] = (int(self.reference_time), 'integer part of BTJD reference date')
		hdu.header['BJDREFF'] = (np.asscalar(self.reference_time % 1), 'fraction of the day in BTJD reference date')
		hdu.header['TSTART'] = ((timestamp-0.5*self.exposure_time)/3600/24, 'time in days (arb. starting point)')
		hdu.header['TSTOP'] = ((timestamp+0.5*self.exposure_time)/3600/24, 'time in days (arb. starting point)')
		hdu.header['NAXIS'] = (2, 'Number of data dimension')
		hdu.header['NAXIS1'] = (self.Ncols, 'Number of pixel columns')
		hdu.header['NAXIS2'] = (self.Nrows, 'Number of pixel rows')
		hdu.header['DQUALITY'] = (0, 'Data quality')
		hdu.header['NUM_FRM'] = (900, 'Number of frames added (true: 1)')
		hdu.header['GAIN'] = (self.gain, 'Gain. Arbitrary value')
		hdu.header['READNOIS'] = (10, 'Readnoise. Arbitrary value')

		# Specify output directory:
		if outdir is None:
			outdir = os.path.join(self.output_folder, 'images')

		# Remove any previous hdf5 file made by prepare_photometry:
		try:
			hdf5filename = 'camera1_ccd1.hdf5'
			os.remove(os.path.join(self.output_folder,hdf5filename))
		except:
			pass

		# Create image files directory if it doesn't already exist:
		if not os.path.exists(outdir):
			os.makedirs(outdir)

		# Write FITS file to output directory:
		hdu.writeto(os.path.join(outdir, 'tess{time:011d}-{camera:d}-{ccd:d}-0000-s_ffic.fits'.format(time=i, camera=1, ccd=1)),
					overwrite=self.overwrite_images)



if __name__ == '__main__':
	sim = simulateFITS(save_images=False)
