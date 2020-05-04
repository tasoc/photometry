#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The basic photometry class for the TASOC Photometry pipeline.
All other specific photometric algorithms will inherit from BasePhotometry.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import h5py
import sqlite3
import logging
import datetime
import os.path
import glob
import contextlib
import warnings
from copy import deepcopy
from astropy._erfa.core import ErfaWarning
from astropy.io import fits
from astropy.table import Table, Column
from astropy import units
import astropy.coordinates as coord
from astropy.time import Time
from astropy.wcs import WCS
import enum
from bottleneck import nanmedian, nanvar, nanstd, allnan
from .image_motion import ImageMovementKernel
from .quality import TESSQualityFlags, PixelQualityFlags, CorrectorQualityFlags
from .utilities import (find_tpf_files, find_hdf5_files, find_catalog_files, rms_timescale,
	find_nearest, ListHandler)
from .catalog import catalog_sqlite_search_footprint
from .plots import plot_image, plt, save_figure
from .spice import TESS_SPICE
from .version import get_version
from . import fixes

# Filter out annoying warnings:
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=ErfaWarning, module="astropy")

__version__ = get_version()

__docformat__ = 'restructuredtext'

hdf5_cache = {}

#--------------------------------------------------------------------------------------------------
@enum.unique
class STATUS(enum.Enum):
	"""
	Status indicator of the status of the photometry.
	"""
	UNKNOWN = 0 #: The status is unknown. The actual calculation has not started yet.
	STARTED = 6 #: The calculation has started, but not yet finished.
	OK = 1      #: Everything has gone well.
	ERROR = 2   #: Encountered a catastrophic error that I could not recover from.
	WARNING = 3 #: Something is a bit fishy. Maybe we should try again with a different algorithm?
	ABORT = 4   #: The calculation was aborted.
	SKIPPED = 5 #: The target was skipped because the algorithm found that to be the best solution.

#--------------------------------------------------------------------------------------------------
class BasePhotometry(object):
	"""
	The basic photometry class for the TASOC Photometry pipeline.
	All other specific photometric algorithms will inherit from this.

	Attributes:
		starid (int): TIC number of star being processed.
		input_folder (str): Root directory where files are loaded from.
		output_folder (str): Root directory where output files are saved.
		plot (bool): Indicates wheter plots should be created as part of the output.
		plot_folder (str): Directory where plots are saved to.

		sector (int): TESS observing sector.
		camera (int): TESS camera (1-4).
		ccd (int): TESS CCD (1-4).
		data_rel (int): Data release number.
		n_readout (int): Number of frames co-added in each timestamp.
		header (dict-like): Primary header, either from TPF or HDF5 files.

		target (dict): Dictionary with information about primary target.
		target_mag (float): TESS magnitude of the main target.
		target_pos_ra (float): Right ascension of the main target at time of observation.
		target_pos_dec (float): Declination of the main target at time of observation.
		target_pos_ra_J2000 (float): Right ascension of the main target at J2000.
		target_pos_dec_J2000 (float): Declination of the main target at J2000.
		target_pos_column (flat): Main target CCD column position.
		target_pos_row (float): Main target CCD row position.
		target_pos_column_stamp (float): Main target CCD column position in stamp.
		target_pos_row_stamp (float): Main target CCD row position in stamp.
		wcs (:py:class:`astropy.wcs.WCS`): World Coordinate system solution.

		lightcurve (``astropy.table.Table`` object): Table to be filled with an extracted lightcurve.
		final_phot_mask (numpy.ndarray): Mask indicating which pixels were used in extraction of
			lightcurve. ``True`` if used, ``False`` otherwise.
		final_position_mask (numpy.ndarray): Mask indicating which pixels were used in extraction
			of positions. ``True`` if used, ``False`` otherwise.
		additional_headers (dict): Additional headers to be included in FITS files.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	#----------------------------------------------------------------------------------------------
	def __init__(self, starid, input_folder, output_folder, datasource='ffi',
		sector=None, camera=None, ccd=None, plot=False, cache='basic', version=5):
		"""
		Initialize the photometry object.

		Parameters:
			starid (int): TIC number of star to be processed.
			input_folder (string): Root directory where files are loaded from.
			output_folder (string): Root directory where output files are saved.
			datasource (string, optional): Source of the data. Options are ``'ffi'`` or ``'tpf'``.
				Default is ``'ffi'``.
			plot (boolean, optional): Create plots as part of the output. Default is ``False``.
			camera (integer, optional): TESS camera (1-4) to load target from (Only used for FFIs).
			ccd (integer, optional): TESS CCD (1-4) to load target from (Only used for FFIs).
			cache (string, optional): Optional values are ``'none'``, ``'full'``
				or ``'basic'`` (Default).
			version (integer): Data release number to be added to headers. Default=5.

		Raises:
			OSError: If starid could not be found in catalog.
			FileNotFoundError: If input file (HDF5, TPF, Catalog) could not be found.
			ValueError: On invalid datasource.
			ValueError: If ``camera`` and ``ccd`` is not provided together with ``datasource='ffi'``.
		"""

		logger = logging.getLogger(__name__)

		if datasource != 'ffi' and not datasource.startswith('tpf'):
			raise ValueError("Invalid datasource: '%s'" % datasource)
		if cache not in ('basic', 'none', 'full'):
			raise ValueError("Invalid cache: '%s'" % cache)

		# Store the input:
		self.starid = starid
		self.input_folder = input_folder
		self.output_folder_base = os.path.abspath(output_folder)
		self.plot = plot
		self.datasource = datasource
		self.version = version

		logger.info('STARID = %d, DATASOURCE = %s', self.starid, self.datasource)

		self._status = STATUS.UNKNOWN
		self._details = {}
		self.tpf = None
		self.hdf = None
		self._MovementKernel = None
		self._images_cube_full = None
		self._images_err_cube_full = None
		self._backgrounds_cube_full = None
		self._pixelflags_cube_full = None
		self._sumimage_full = None

		# Add a ListHandler to the logging of the corrections module.
		# This is needed to catch any errors and warnings made by the correctors
		# for ultimately storing them in the TODO-file.
		# https://stackoverflow.com/questions/36408496/python-logging-handler-to-append-to-list
		self.message_queue = []
		handler = ListHandler(message_queue=self.message_queue, level=logging.WARNING)
		formatter = logging.Formatter('%(levelname)s: %(message)s')
		handler.setFormatter(formatter)
		logging.getLogger('photometry').addHandler(handler)

		# Directory where output files will be saved:
		self.output_folder = os.path.join(
			self.output_folder_base,
			self.datasource[:3], # Only three first characters for cases with "tpf:XXXXXX"
			'{0:011d}'.format(self.starid)[:5]
		)

		# Set directory where diagnostics plots should be saved to:
		self.plot_folder = None
		if self.plot:
			self.plot_folder = os.path.join(self.output_folder, 'plots', '{0:011d}'.format(self.starid))
			os.makedirs(self.plot_folder, exist_ok=True)

		# Init table that will be filled with lightcurve stuff:
		self.lightcurve = Table()

		if self.datasource == 'ffi':
			# The camera and CCD should also come as input
			# They will be needed to find the correct input files
			if sector is None or camera is None or ccd is None:
				raise ValueError("SECTOR, CAMERA and CCD keywords must be provided for FFI targets.")

			self.sector = sector # TESS observing sector.
			self.camera = camera # TESS camera.
			self.ccd = ccd # TESS CCD.

			logger.debug('SECTOR = %s', self.sector)
			logger.debug('CAMERA = %s', self.camera)
			logger.debug('CCD = %s', self.ccd)

			# Load stuff from the common HDF5 file:
			filepath_hdf5 = find_hdf5_files(input_folder, sector=self.sector, camera=self.camera, ccd=self.ccd)
			if len(filepath_hdf5) != 1:
				raise FileNotFoundError("HDF5 File not found. SECTOR=%d, CAMERA=%d, CCD=%d" % (self.sector, self.camera, self.ccd))
			filepath_hdf5 = filepath_hdf5[0]
			self.filepath_hdf5 = filepath_hdf5

			logger.debug("CACHE = %s", cache)
			load_into_cache = False
			if cache == 'none':
				load_into_cache = True
			else:
				global hdf5_cache
				if filepath_hdf5 not in hdf5_cache:
					hdf5_cache[filepath_hdf5] = {}
					load_into_cache = True
				elif cache == 'full' and hdf5_cache[filepath_hdf5].get('_images_cube_full') is None:
					load_into_cache = True

			# Open the HDF5 file for reading if we are not holding everything in memory:
			if load_into_cache or cache != 'full':
				self.hdf = h5py.File(filepath_hdf5, 'r')

			if load_into_cache:
				logger.debug('Loading basic data into cache...')
				attrs = {}

				# Just a shorthand for the attributes we use as "headers":
				hdr = dict(self.hdf['images'].attrs)
				attrs['header'] = hdr
				attrs['data_rel'] = hdr['DATA_REL'] # Data release number

				# Start filling out the basic vectors:
				self.lightcurve['time'] = Column(self.hdf['time'], description='Time', dtype='float64', unit='TBJD')
				N = len(self.lightcurve['time'])
				self.lightcurve['cadenceno'] = Column(self.hdf['cadenceno'], description='Cadence number', dtype='int32')
				self.lightcurve['quality'] = Column(self.hdf['quality'], description='Quality flags', dtype='int32')
				if 'timecorr' in self.hdf:
					self.lightcurve['timecorr'] = Column(self.hdf['timecorr'], description='Barycentric time correction', unit='days', dtype='float32')
				else:
					self.lightcurve['timecorr'] = Column(np.zeros(N, dtype='float32'), description='Barycentric time correction', unit='days', dtype='float32')

				# Correct timestamp offset that was in early data releases:
				if fixes.time_offset_should_be_fixed(header=hdr):
					logger.debug("Fixes: Applying time offset correction")
					self.lightcurve['time'] = fixes.time_offset_apply(self.lightcurve['time'])
				else:
					logger.debug("Fixes: Not applying time offset correction")

				attrs['lightcurve'] = self.lightcurve

				# World Coordinate System solution:
				if isinstance(self.hdf['wcs'], h5py.Group):
					refindx = self.hdf['wcs'].attrs['ref_frame']
					hdr_string = self.hdf['wcs']['%04d' % refindx][0]
				else:
					hdr_string = self.hdf['wcs'][0]
				if not isinstance(hdr_string, str): hdr_string = hdr_string.decode("utf-8") # For Python 3
				self.wcs = WCS(header=fits.Header().fromstring(hdr_string), relax=True) # World Coordinate system solution.
				attrs['wcs'] = self.wcs

				# Get shape of sumimage from hdf5 file:
				attrs['_max_stamp'] = (0, self.hdf['sumimage'].shape[0], 0, self.hdf['sumimage'].shape[1])
				attrs['pixel_offset_row'] = hdr.get('PIXEL_OFFSET_ROW', 0)
				attrs['pixel_offset_col'] = hdr.get('PIXEL_OFFSET_COLUMN', 44) # Default for TESS data

				# Get info for psf fit Gaussian statistic:
				attrs['readnoise'] = hdr.get('READNOIS', 10)
				attrs['gain'] = hdr.get('GAIN', 100)
				attrs['num_frm'] = hdr.get('NUM_FRM', 900) # Number of frames co-added in each timestamp (Default=TESS).
				attrs['n_readout'] = hdr.get('NREADOUT', int(attrs['num_frm']*(1-2/hdr.get('CRBLKSZ', np.inf)))) # Number of frames co-added in each timestamp (Default=TESS).

				# Load MovementKernel into memory:
				attrs['_MovementKernel'] = self.MovementKernel

				# The full sum-image:
				attrs['_sumimage_full'] = np.asarray(self.hdf['sumimage'])

				# Store attr in global variable:
				hdf5_cache[filepath_hdf5] = deepcopy(attrs)

				# If we are doing a full cache (everything in memory) load the image cubes as well.
				# Note that this will take up A LOT of memory!
				if cache == 'full':
					logger.warning('Loading full image cubes into cache...')
					hdf5_cache[filepath_hdf5]['_images_cube_full'] = np.empty((attrs['_max_stamp'][1], attrs['_max_stamp'][3], N), dtype='float32')
					hdf5_cache[filepath_hdf5]['_images_err_cube_full'] = np.empty((attrs['_max_stamp'][1], attrs['_max_stamp'][3], N), dtype='float32')
					hdf5_cache[filepath_hdf5]['_backgrounds_cube_full'] = np.empty((attrs['_max_stamp'][1], attrs['_max_stamp'][3], N), dtype='float32')
					hdf5_cache[filepath_hdf5]['_pixelflags_cube_full'] = np.empty((attrs['_max_stamp'][1], attrs['_max_stamp'][3], N), dtype='uint8')
					for k in range(N):
						hdf5_cache[filepath_hdf5]['_images_cube_full'][:, :, k] = self.hdf['images/%04d' % k]
						hdf5_cache[filepath_hdf5]['_images_err_cube_full'][:, :, k] = self.hdf['images_err/%04d' % k]
						hdf5_cache[filepath_hdf5]['_backgrounds_cube_full'][:, :, k] = self.hdf['backgrounds/%04d' % k]
						hdf5_cache[filepath_hdf5]['_pixelflags_cube_full'][:, :, k] = self.hdf['pixelflags/%04d' % k]

					# We dont need the file anymore!
					self.hdf.close()
					self.hdf = None
			else:
				logger.debug('Loaded data from cache!')
				attrs = hdf5_cache[filepath_hdf5] # Pointer to global variable

			# Set all the attributes from the cache:
			# TODO: Does this create copies of data? - if so we should mayde delete "attrs" again?
			for key, value in attrs.items():
				setattr(self, key, value)

		elif self.datasource.startswith('tpf'):
			# If the datasource was specified as 'tpf:starid' it means
			# that we should load from the specified starid instead of
			# the starid of the current main target.
			if self.datasource.startswith('tpf:'):
				starid_to_load = int(self.datasource[4:])
				self.datasource = 'tpf'
			else:
				starid_to_load = self.starid

			# Find the target pixel file for this star:
			fname = find_tpf_files(input_folder, sector=sector, starid=starid_to_load)
			if len(fname) == 1:
				fname = fname[0]
			elif len(fname) == 0:
				raise FileNotFoundError("Target Pixel File not found")
			elif len(fname) > 1:
				raise OSError("Multiple Target Pixel Files found matching pattern")

			# Open the FITS file:
			self.tpf = fits.open(fname, mode='readonly', memmap=True)

			# Load sector, camera and CCD from the FITS header:
			self.header = self.tpf[0].header
			self.sector = self.tpf[0].header['SECTOR']
			self.camera = self.tpf[0].header['CAMERA']
			self.ccd = self.tpf[0].header['CCD']
			self.data_rel = self.tpf[0].header['DATA_REL'] # Data release number

			# Fix for timestamps that are not defined. Simply remove them from the table:
			# This is seen in some file from sector 1.
			indx_good_times = np.isfinite(self.tpf['PIXELS'].data['TIME'])
			self.tpf['PIXELS'].data = self.tpf['PIXELS'].data[indx_good_times]

			# Extract the relevant information from the FITS file:
			self.lightcurve['time'] = Column(self.tpf['PIXELS'].data['TIME'], description='Time', dtype='float64', unit='TBJD')
			self.lightcurve['timecorr'] = Column(self.tpf['PIXELS'].data['TIMECORR'], description='Barycentric time correction', unit='days', dtype='float32')
			self.lightcurve['cadenceno'] = Column(self.tpf['PIXELS'].data['CADENCENO'], description='Cadence number', dtype='int32')
			self.lightcurve['quality'] = Column(self.tpf['PIXELS'].data['QUALITY'], description='Quality flags', dtype='int32')

			# World Coordinate System solution:
			self.wcs = WCS(header=self.tpf['APERTURE'].header, relax=True)

			# Get the positions of the stamp from the FITS header:
			self._max_stamp = (
				self.tpf['APERTURE'].header['CRVAL2P'] - 1,
				self.tpf['APERTURE'].header['CRVAL2P'] - 1 + self.tpf[2].header['NAXIS2'],
				self.tpf['APERTURE'].header['CRVAL1P'] - 1,
				self.tpf['APERTURE'].header['CRVAL1P'] - 1 + self.tpf[2].header['NAXIS1']
			)
			self.pixel_offset_row = self.tpf['APERTURE'].header['CRVAL2P'] - 1
			self.pixel_offset_col = self.tpf['APERTURE'].header['CRVAL1P'] - 1

			logger.debug(
				'Max stamp size: (%d, %d)',
				self._max_stamp[1] - self._max_stamp[0],
				self._max_stamp[3] - self._max_stamp[2]
			)

			# Get info for psf fit Gaussian statistic:
			self.readnoise = self.tpf['PIXELS'].header.get('READNOIA', 10) # FIXME: This only loads readnoise from channel A!
			self.gain = self.tpf['PIXELS'].header.get('GAINA', 100) # FIXME: This only loads gain from channel A!
			self.num_frm = self.tpf['PIXELS'].header.get('NUM_FRM', 60) # Number of frames co-added in each timestamp.
			self.n_readout = self.tpf['PIXELS'].header.get('NREADOUT', 48) # Number of frames co-added in each timestamp.

			# Load stuff from the common HDF5 file:
			filepath_hdf5 = find_hdf5_files(input_folder, sector=self.sector, camera=self.camera, ccd=self.ccd)
			if len(filepath_hdf5) != 1:
				raise FileNotFoundError("HDF5 File not found. SECTOR=%d, CAMERA=%d, CCD=%d" % (self.sector, self.camera, self.ccd))
			filepath_hdf5 = filepath_hdf5[0]
			self.hdf = h5py.File(filepath_hdf5, 'r', libver='latest')

			# Correct timestamp offset that was in early data releases:
			if fixes.time_offset_should_be_fixed(header=self.tpf[0].header):
				logger.debug("Fixes: Applying time offset correction")
				self.lightcurve['time'] = fixes.time_offset_apply(self.lightcurve['time'])
			else:
				logger.debug("Fixes: Not applying time offset correction")

		else:
			raise ValueError("Invalid datasource: '%s'" % self.datasource)

		# The file to load the star catalog from:
		self.catalog_file = find_catalog_files(input_folder, sector=self.sector, camera=self.camera, ccd=self.ccd)
		self._catalog = None
		logger.debug('Catalog file: %s', self.catalog_file)
		if len(self.catalog_file) != 1:
			raise FileNotFoundError("Catalog file not found: SECTOR=%s, CAMERA=%s, CCD=%s" % (self.sector, self.camera, self.ccd))
		self.catalog_file = self.catalog_file[0]

		# Load information about main target:
		with contextlib.closing(sqlite3.connect(self.catalog_file)) as conn:
			conn.row_factory = sqlite3.Row
			cursor = conn.cursor()
			cursor.execute("SELECT ra,decl,ra_J2000,decl_J2000,pm_ra,pm_decl,tmag,teff FROM catalog WHERE starid={0:d};".format(self.starid))
			target = cursor.fetchone()
			if target is None:
				raise OSError("Star could not be found in catalog: {0:d}".format(self.starid))
			self.target = dict(target) # Dictionary of all main target properties.
			self.target_tmag = target['tmag'] # TESS magnitude of the main target.
			self.target_pos_ra = target['ra'] # Right ascension of the main target at time of observation.
			self.target_pos_dec = target['decl'] # Declination of the main target at time of observation.
			self.target_pos_ra_J2000 = target['ra_J2000'] # Right ascension of the main target at J2000.
			self.target_pos_dec_J2000 = target['decl_J2000'] # Declination of the main target at J2000.
			cursor.execute("SELECT sector,reference_time,ticver FROM settings LIMIT 1;")
			target = cursor.fetchone()
			if target is not None:
				self._catalog_reference_time = target['reference_time']
				self.ticver = target['ticver']
			cursor.close()

		# Define the columns that have to be filled by the do_photometry method:
		self.Ntimes = len(self.lightcurve['time'])
		self.lightcurve['flux'] = Column(length=self.Ntimes, description='Flux', dtype='float64')
		self.lightcurve['flux_err'] = Column(length=self.Ntimes, description='Flux Error', dtype='float64')
		self.lightcurve['flux_background'] = Column(length=self.Ntimes, description='Background flux', dtype='float64')
		self.lightcurve['pos_centroid'] = Column(length=self.Ntimes, shape=(2,), description='Centroid position', unit='pixels', dtype='float64')
		self.lightcurve['pos_corr'] = Column(length=self.Ntimes, shape=(2,), description='Position correction', unit='pixels', dtype='float64')

		# Correct timestamps for light-travel time in FFIs:
		# http://docs.astropy.org/en/stable/time/#barycentric-and-heliocentric-light-travel-time-corrections
		if self.datasource == 'ffi':
			# Coordinates of the target as astropy SkyCoord object:
			star_coord = coord.SkyCoord(
				ra=self.target['ra'],
				dec=self.target['decl'],
				unit=units.deg,
				frame='icrs'
			)

			# Use the SPICE kernels to get accurate positions of TESS, to be used in calculating
			# the light-travel-time corrections:
			with TESS_SPICE() as knl:
				# Change the timestamps back to uncorrected JD (TDB) in the TESS frame:
				time_nocorr = np.asarray(self.lightcurve['time'] - self.lightcurve['timecorr'])

				# Use SPICE kernels to get new barycentric time correction for the stars coordinates:
				tm, tc = knl.barycorr(time_nocorr + 2457000, star_coord)
				self.lightcurve['time'] = tm - 2457000
				self.lightcurve['timecorr'] = tc

		# Init arrays that will be filled with lightcurve stuff:
		self.final_phot_mask = None # Mask indicating which pixels were used in extraction of lightcurve.
		self.final_position_mask = None # Mask indicating which pixels were used in extraction of position.
		self.additional_headers = {} # Additional headers to be included in FITS files.

		# Project target position onto the pixel plane:
		self.target_pos_column, self.target_pos_row = self.wcs.all_world2pix(self.target['ra'], self.target['decl'], 0, ra_dec_order=True)
		if self.datasource.startswith('tpf'):
			self.target_pos_column += self.pixel_offset_col
			self.target_pos_row += self.pixel_offset_row
		logger.info("Target column: %f", self.target_pos_column)
		logger.info("Target row: %f", self.target_pos_row)

		# Store the jitter at the target position:
		# TODO: TPF and FFI may end up with slightly different zero-points.
		if self.datasource.startswith('tpf'):
			self.lightcurve['pos_corr'][:] = np.column_stack((self.tpf[1].data['POS_CORR1'], self.tpf[1].data['POS_CORR2']))
		else:
			self.lightcurve['pos_corr'][:] = self.MovementKernel.jitter(self.lightcurve['time'] - self.lightcurve['timecorr'], self.target_pos_column, self.target_pos_row)

		# Init the stamp:
		self._stamp = None
		self.target_pos_column_stamp = None # Main target CCD column position in stamp.
		self.target_pos_row_stamp = None # Main target CCD row position in stamp.
		self._set_stamp()
		self._sumimage = None
		self._images_cube = None
		self._images_err_cube = None
		self._backgrounds_cube = None
		self._pixelflags_cube = None
		self._aperture = None

	#----------------------------------------------------------------------------------------------
	def __enter__(self):
		return self

	#----------------------------------------------------------------------------------------------
	def __exit__(self, *args):
		self.close()

	#----------------------------------------------------------------------------------------------
	def __del__(self):
		self.close()

	#----------------------------------------------------------------------------------------------
	def close(self):
		"""Close photometry object and close all associated open file handles."""
		if hasattr(self, 'hdf') and self.hdf:
			self.hdf.close()
		if hasattr(self, 'tpf') and self.tpf:
			self.tpf.close()

	#----------------------------------------------------------------------------------------------
	def clear_cache(self):
		"""Clear internal cache"""
		global hdf5_cache
		hdf5_cache = {}

	#----------------------------------------------------------------------------------------------
	@property
	def status(self):
		"""The status of the photometry. From :py:class:`STATUS`."""
		return self._status

	#----------------------------------------------------------------------------------------------
	def default_stamp(self):
		"""
		The default size of the stamp to use.

		The stamp will be centered on the target star position, with
		a width and height specified by this function. The stamp can
		later be resized using :py:func:`resize_stamp`.

		Returns:
			int: Number of rows
			int: Number of columns

		Note:
			This function is only used for FFIs. For postage stamps
			the default stamp is the entire available postage stamp.

		See Also:
			:py:func:`resize_stamp`
		"""
		# Decide how many pixels to use based on lookup tables as a function of Tmag:
		tmag = np.array([0.0, 0.52631579, 1.05263158, 1.57894737, 2.10526316,
			2.63157895, 3.15789474, 3.68421053, 4.21052632, 4.73684211,
			5.26315789, 5.78947368, 6.31578947, 6.84210526, 7.36842105,
			7.89473684, 8.42105263, 8.94736842, 9.47368421, 10.0, 13.0])

		height = np.array([831.98319063, 533.58494422, 344.0840884, 223.73963332,
			147.31365728, 98.77856016, 67.95585074, 48.38157414,
			35.95072974, 28.05639497, 23.043017, 19.85922009,
			17.83731732, 16.5532873, 15.73785092, 15.21999971,
			14.89113301, 14.68228285, 14.54965042, 14.46542084, 14.0])

		width = np.array([157.71602062, 125.1238281, 99.99440209, 80.61896267,
			65.6799962, 54.16166547, 45.28073365, 38.4333048,
			33.15375951, 28.05639497, 23.043017, 19.85922009,
			17.83731732, 16.5532873, 15.73785092, 15.21999971,
			14.89113301, 14.68228285, 14.54965042, 14.46542084, 14.0])

		Ncolumns = np.interp(self.target_tmag, tmag, width)
		Nrows = np.interp(self.target_tmag, tmag, height)

		# Round off and make sure we have minimum 15 pixels:
		Nrows = np.maximum(np.ceil(Nrows), 15)
		Ncolumns = np.maximum(np.ceil(Ncolumns), 15)
		return Nrows, Ncolumns

	#----------------------------------------------------------------------------------------------
	def resize_stamp(self, down=None, up=None, left=None, right=None, width=None, height=None):
		"""
		Resize the stamp in a given direction.

		Parameters:
			down (int, optional): Number of pixels to extend downwards.
			up (int, optional): Number of pixels to extend upwards.
			left (int, optional): Number of pixels to extend left.
			right (int, optional): Number of pixels to extend right.
			width (int, optional): Set the width of the stamp to this number of pixels.
				This takes presendence over ``left`` and ``right`` if they are also provided.
			height (int, optional): Set the height of the stamp to this number of pixels.
				This takes presendence over ``up`` and ``down`` if they are also provided.

		Returns:
			bool: `True` if the stamp could be resized, `False` otherwise.
		"""

		old_stamp = self._stamp

		self._stamp = list(self._stamp)
		if up:
			self._stamp[1] += up
		if down:
			self._stamp[0] -= down
		if left:
			self._stamp[2] -= left
		if right:
			self._stamp[3] += right
		if height:
			self._stamp[0] = int(np.round(self.target_pos_row)) - height//2
			self._stamp[1] = int(np.round(self.target_pos_row)) + height//2 + 1
		if width:
			self._stamp[2] = int(np.round(self.target_pos_column)) - width//2
			self._stamp[3] = int(np.round(self.target_pos_column)) + width//2 + 1
		self._stamp = tuple(self._stamp)

		# Set stamp and check if the stamp actually changed:
		stamp_changed = self._set_stamp(compare_stamp=old_stamp)

		# Count the number of times that we are resizing the stamp:
		if stamp_changed:
			self._details['stamp_resizes'] = self._details.get('stamp_resizes', 0) + 1

		# Return if the stamp actually changed:
		return stamp_changed

	#----------------------------------------------------------------------------------------------
	def _set_stamp(self, compare_stamp=None):
		"""
		The default size of the stamp to use.

		The stamp will be centered on the target star position, with
		a width and height specified by this function. The stamp can
		later be resized using :py:func:`resize_stamp`.

		Parameters:
			compare_stamp (tuple): Stamp to compare against whether anything changed.

		Returns:
			bool: `True` if ``compare_stamp`` is set and has changed. If ``compare_stamp``
			is not provided, always returns `True`.

		See Also:
			:py:func:`resize_stamp`

		Note:
			Stamp is zero-based counted from the TOP of the image.
		"""

		logger = logging.getLogger(__name__)

		if not self._stamp:
			if self.datasource == 'ffi':
				Nrows, Ncolumns = self.default_stamp()
				logger.info("Setting default stamp with sizes (%d,%d)", Nrows, Ncolumns)
				self._stamp = (
					int(np.round(self.target_pos_row)) - Nrows//2,
					int(np.round(self.target_pos_row)) + Nrows//2 + 1,
					int(np.round(self.target_pos_column)) - Ncolumns//2,
					int(np.round(self.target_pos_column)) + Ncolumns//2 + 1
				)
			else:
				Nrows = self._max_stamp[1] - self._max_stamp[0]
				Ncolumns = self._max_stamp[3] - self._max_stamp[2]
				logger.info("Setting default stamp with sizes (%d,%d)", Nrows, Ncolumns)
				self._stamp = self._max_stamp

		# Limit the stamp to not go outside the limits of the images:
		# TODO: We really should have a thourgh cleanup in the self._stamp, self._maxstamp and self.pixel_offset_* mess!
		self._stamp = list(self._stamp)
		if self.datasource == 'ffi':
			self._stamp[0] = int(np.maximum(self._stamp[0], self._max_stamp[0] + self.pixel_offset_row))
			self._stamp[1] = int(np.minimum(self._stamp[1], self._max_stamp[1] + self.pixel_offset_row))
			self._stamp[2] = int(np.maximum(self._stamp[2], self._max_stamp[2] + self.pixel_offset_col))
			self._stamp[3] = int(np.minimum(self._stamp[3], self._max_stamp[3] + self.pixel_offset_col))
		else:
			self._stamp[0] = int(np.maximum(self._stamp[0], self._max_stamp[0]))
			self._stamp[1] = int(np.minimum(self._stamp[1], self._max_stamp[1]))
			self._stamp[2] = int(np.maximum(self._stamp[2], self._max_stamp[2]))
			self._stamp[3] = int(np.minimum(self._stamp[3], self._max_stamp[3]))
		self._stamp = tuple(self._stamp)

		# Sanity checks:
		if self._stamp[0] > self._stamp[1] or self._stamp[2] > self._stamp[3]:
			raise ValueError("Invalid stamp selected")

		# Store the stamp in details:
		self._details['stamp'] = self._stamp

		# Check if the stamp actually changed:
		if self._stamp == compare_stamp:
			return False

		# Calculate main target position in stamp:
		self.target_pos_row_stamp = self.target_pos_row - self._stamp[0]
		self.target_pos_column_stamp = self.target_pos_column - self._stamp[2]

		# Force sum-image and catalog to be recalculated next time:
		self._sumimage = None
		self._catalog = None
		self._images_cube = None
		self._backgrounds_cube = None
		self._pixelflags_cube = None
		self._aperture = None
		return True

	#----------------------------------------------------------------------------------------------
	def get_pixel_grid(self):
		"""
		Returns mesh-grid of the pixels (1-based) in the stamp.

		Returns:
			tuple(cols, rows): Meshgrid of pixel coordinates in the current stamp.
		"""
		return np.meshgrid(
			np.arange(self._stamp[2]+1, self._stamp[3]+1, 1, dtype='int32'),
			np.arange(self._stamp[0]+1, self._stamp[1]+1, 1, dtype='int32')
		)

	#----------------------------------------------------------------------------------------------
	@property
	def stamp(self):
		"""
		Tuple indicating the stamps position within the larger image.

		Returns:
			tuple: Tuple of (row_min, row_max, col_min, col_max).
		"""
		return self._stamp

	#----------------------------------------------------------------------------------------------
	def _load_cube(self, tpf_field='FLUX', hdf_group='images', full_cube=None):
		"""
		Load data cube into memory from TPF and HDF5 files depending on datasource.
		"""
		if self.datasource == 'ffi':
			ir1 = self._stamp[0] - self.pixel_offset_row
			ir2 = self._stamp[1] - self.pixel_offset_row
			ic1 = self._stamp[2] - self.pixel_offset_col
			ic2 = self._stamp[3] - self.pixel_offset_col
			if full_cube is None:
				# We dont have an in-memory version of the full cube, so let us
				# create the cube by loading the cutouts of each image:
				cube = np.empty((ir2-ir1, ic2-ic1, self.Ntimes), dtype='float32')
				if hdf_group in self.hdf:
					for k in range(self.Ntimes):
						cube[:, :, k] = self.hdf[hdf_group + '/%04d' % k][ir1:ir2, ic1:ic2]
				else:
					cube[:, :, :] = np.NaN
			else:
				# We have an in-memory version of the full cube.
				# TODO: Will this create copy of data in memory?
				cube = full_cube[ir1:ir2, ic1:ic2, :]
		else:
			ir1 = self._stamp[0] - self._max_stamp[0]
			ir2 = self._stamp[1] - self._max_stamp[0]
			ic1 = self._stamp[2] - self._max_stamp[2]
			ic2 = self._stamp[3] - self._max_stamp[2]
			cube = np.empty((ir2-ir1, ic2-ic1, self.Ntimes), dtype='float32')
			for k in range(self.Ntimes):
				cube[:, :, k] = self.tpf['PIXELS'].data[tpf_field][k][ir1:ir2, ic1:ic2]

		return cube

	#----------------------------------------------------------------------------------------------
	@property
	def images_cube(self):
		"""
		Image cube containing all the images as a function of time.

		Returns:
			ndarray: Three dimentional array with shape ``(rows, cols, times)``, where
				``rows`` is the number of rows in the image, ``cols`` is the number
				of columns and ``times`` is the number of timestamps.

		Note:
			The images has had the large-scale background subtracted. If needed
			the backgrounds can be added again from :py:meth:`backgrounds`
			or :py:meth:`backgrounds_cube`.

		Example:

			>>> pho = BasePhotometry(starid)
			>>> print(pho.images_cube.shape)
			>>>   (10, 10, 1399)

		See Also:
			:py:meth:`images`, :py:meth:`backgrounds`, :py:meth:`backgrounds_cube`
		"""
		if self._images_cube is None:
			self._images_cube = self._load_cube(tpf_field='FLUX', hdf_group='images', full_cube=self._images_cube_full)
		return self._images_cube

	#----------------------------------------------------------------------------------------------
	@property
	def images_err_cube(self):
		"""
		Image cube containing all the uncertainty images as a function of time.

		Returns:
			ndarray: Three dimentional array with shape ``(rows, cols, times)``, where
				``rows`` is the number of rows in the image, ``cols`` is the number
				of columns and ``times`` is the number of timestamps.

		Example:

			>>> pho = BasePhotometry(starid)
			>>> print(pho.images_err_cube.shape)
			>>>   (10, 10, 1399)

		See Also:
			:py:meth:`images`, :py:meth:`backgrounds`, :py:meth:`backgrounds_cube`
		"""
		if self._images_err_cube is None:
			self._images_err_cube = self._load_cube(tpf_field='FLUX_ERR', hdf_group='images_err', full_cube=self._images_err_cube_full)
		return self._images_err_cube

	#----------------------------------------------------------------------------------------------
	@property
	def backgrounds_cube(self):
		"""
		Image cube containing all the background images as a function of time.

		Returns:
			ndarray: Three dimentional array with shape ``(rows, cols, times)``, where
				``rows`` is the number of rows in the image, ``cols`` is the number
				of columns and ``times`` is the number of timestamps.

		Example:

			>>> pho = BasePhotometry(starid)
			>>> print(pho.backgrounds_cube.shape):
			>>>   (10, 10, 1399)

		See Also:
			:py:meth:`backgrounds`, :py:meth:`images_cube`, :py:meth:`images`
		"""
		if self._backgrounds_cube is None:
			self._backgrounds_cube = self._load_cube(tpf_field='FLUX_BKG', hdf_group='backgrounds', full_cube=self._backgrounds_cube_full)
		return self._backgrounds_cube

	#----------------------------------------------------------------------------------------------
	@property
	def pixelflags_cube(self):
		"""
		Cube containing all pixel flag images as a function of time.

		Returns:
			ndarray: Three dimentional array with shape ``(rows, cols, ffi_times)``, where
				``rows`` is the number of rows in the image, ``cols`` is the number
				of columns and ``ffi_times`` is the number of timestamps in the FFIs.

		Note:
			This function will only return flags on the timestamps of the FFIs, even though
			an TPF is being processed.

		Example:

			>>> pho = BasePhotometry(starid)
			>>> print(pho.pixelflags_cube.shape):
			>>>   (10, 10, 1399)

		See Also:
			:py:meth:`pixelflags`, :py:meth:`backgrounds_cube`, :py:meth:`images_cube`.
		"""
		if self._pixelflags_cube is None:
			# We can't used the _loac_cube function here, since we always have
			# to load from the HDF5 file, even though we are running an TPF.
			if self._pixelflags_cube_full is None:
				ir1 = self._stamp[0] - self.hdf['images'].attrs.get('PIXEL_OFFSET_ROW', 0)
				ir2 = self._stamp[1] - self.hdf['images'].attrs.get('PIXEL_OFFSET_ROW', 0)
				ic1 = self._stamp[2] - self.hdf['images'].attrs.get('PIXEL_OFFSET_COLUMN', 44)
				ic2 = self._stamp[3] - self.hdf['images'].attrs.get('PIXEL_OFFSET_COLUMN', 44)

				# We dont have an in-memory version of the full cube, so let us
				# create the cube by loading the cutouts of each image:
				cube = np.empty((ir2-ir1, ic2-ic1, len(self.hdf['time'])), dtype='uint8')
				if 'pixel_flags' in self.hdf:
					for k in range(len(self.hdf['time'])):
						cube[:, :, k] = self.hdf['pixel_flags/%04d' % k][ir1:ir2, ic1:ic2]
				else:
					cube[:, :, :] = 0
			else:
				# We have an in-memory version of the full cube.
				# TODO: Will this create copy of data in memory?
				cube = self._pixelflags_cube_full[ir1:ir2, ic1:ic2, :]

			self._pixelflags_cube = cube

		return self._pixelflags_cube

	#----------------------------------------------------------------------------------------------
	@property
	def pixelflags(self):
		"""
		Iterator that will loop through the pixel flag images.

		Returns:
			iterator: Iterator which can be used to loop through the pixel flags images.

		Example:

			>>> pho = BasePhotometry(starid)
			>>> for img in pho.pixelflags:
			>>> 	print(img)

		See Also:
			:py:meth:`pixelflags_cube`, :py:meth:`images`, :py:meth:`backgrounds`
		"""
		# Yield slices from the data-cube as an iterator:
		if self.datasource == 'ffi':
			for k in range(self.Ntimes):
				yield self.pixelflags_cube[:, :, k]
		else:
			hdf_times = np.asarray(self.hdf['time']) - np.asarray(self.hdf['timecorr'])
			for k in range(self.Ntimes):
				indx = find_nearest(hdf_times, self.lightcurve['time'][k] - self.lightcurve['timecorr'][k])
				yield self.pixelflags_cube[:, :, indx]

	#----------------------------------------------------------------------------------------------
	@property
	def images(self):
		"""
		Iterator that will loop through the image stamps.

		Returns:
			iterator: Iterator which can be used to loop through the image stamps.

		Note:
			The images has had the large-scale background subtracted. If needed
			the backgrounds can be added again from :py:meth:`backgrounds`.

		Note:
			For each image, this function will actually load the necessary
			data from disk, so don't loop through it more than you absolutely
			have to to save I/O.

		Example:

			>>> pho = BasePhotometry(starid)
			>>> for img in pho.images:
			>>> 	print(img)

		See Also:
			:py:meth:`images_cube`, :py:meth:`images_err`, :py:meth:`backgrounds`
		"""
		# Yield slices from the data-cube as an iterator:
		for k in range(self.Ntimes):
			yield self.images_cube[:, :, k]

	#----------------------------------------------------------------------------------------------
	@property
	def images_err(self):
		"""
		Iterator that will loop through the uncertainty image stamps.

		Returns:
			iterator: Iterator which can be used to loop through the uncertainty image stamps.

		Example:

			>>> pho = BasePhotometry(starid)
			>>> for imgerr in pho.images_err:
			>>> 	print(imgerr)

		See Also:
			:py:meth:`images_err_cube`, :py:meth:`images`, :py:meth:`images_cube`, :py:meth:`backgrounds`
		"""
		# Yield slices from the data-cube as an iterator:
		for k in range(self.Ntimes):
			yield self.images_err_cube[:, :, k]

	#----------------------------------------------------------------------------------------------
	@property
	def backgrounds(self):
		"""
		Iterator that will loop through the background-image stamps.

		Returns:
			iterator: Iterator which can be used to loop through the background-image stamps.

		Note:
			For each image, this function will actually load the necessary
			data from disk, so don't loop through it more than you absolutely
			have to to save I/O.

		Example:

			>>> pho = BasePhotometry(starid)
			>>> for img in pho.backgrounds:
			>>> 	print(img)

		See Also:
			:py:meth:`backgrounds_cube`, :py:meth:`images`
		"""
		# Yield slices from the data-cube as an iterator:
		for k in range(self.Ntimes):
			yield self.backgrounds_cube[:, :, k]

	#----------------------------------------------------------------------------------------------
	@property
	def sumimage(self):
		"""
		Average image.

		Calculated as the mean of all good images (quality=0) as a function of time.
		For FFIs this has been pre-calculated and for postage-stamps it is calculated
		on-the-fly when needed.

		Returns:
			numpy.array: Summed image across all valid timestamps.
		"""
		if self._sumimage is None:
			if self.datasource == 'ffi':
				ir1 = self._stamp[0] - self.pixel_offset_row
				ir2 = self._stamp[1] - self.pixel_offset_row
				ic1 = self._stamp[2] - self.pixel_offset_col
				ic2 = self._stamp[3] - self.pixel_offset_col
				self._sumimage = self._sumimage_full[ir1:ir2, ic1:ic2]
			else:
				self._sumimage = np.zeros((self._stamp[1]-self._stamp[0], self._stamp[3]-self._stamp[2]), dtype='float64')
				Nimg = np.zeros_like(self._sumimage, dtype='int32')
				for k, img in enumerate(self.images):
					if TESSQualityFlags.filter(self.lightcurve['quality'][k]):
						isgood = np.isfinite(img)
						img[~isgood] = 0
						Nimg += np.asarray(isgood, dtype='int32')
						self._sumimage += img

				isgood = (Nimg > 0)
				self._sumimage[isgood] /= Nimg[isgood]
				self._sumimage[~isgood] = np.NaN

			if self.plot:
				fig, ax = plt.subplots()
				plot_image(self._sumimage, ax=ax, offset_axes=(self._stamp[2]+1, self._stamp[0]+1),
					xlabel='Pixel Column Number', ylabel='Pixel Row Number', cbar='right')
				ax.plot(self.target_pos_column + 1, self.target_pos_row + 1, 'r+')
				save_figure(os.path.join(self.plot_folder, 'sumimage'), fig=fig)
				plt.close(fig)

		return self._sumimage

	#----------------------------------------------------------------------------------------------
	@property
	def aperture(self):
		"""
		Flags for each pixel, as defined by the TESS data product manual.

		Returns:
			numpy.array: 2D array of flags for each pixel.
		"""
		if self._aperture is None:
			if self.datasource == 'ffi':
				# Make aperture image:
				cols, rows = self.get_pixel_grid()
				self._aperture = np.asarray(np.isfinite(self.sumimage), dtype='int32')

				# Add mapping onto TESS output channels:
				self._aperture[(45 <= cols) & (cols <= 556)] |= 32 # CCD output A
				self._aperture[(557 <= cols) & (cols <= 1068)] |= 64 # CCD output B
				self._aperture[(1069 <= cols) & (cols <= 1580)] |= 128 # CCD output C
				self._aperture[(1581 <= cols) & (cols <= 2092)] |= 256 # CCD output D

				# Add information about which pixels were used for background calculation:
				if 'backgrounds_pixels_used' in self.hdf:
					# Coordinates in the FFI of image:
					ir1 = self._stamp[0] - self.pixel_offset_row
					ir2 = self._stamp[1] - self.pixel_offset_row
					ic1 = self._stamp[2] - self.pixel_offset_col
					ic2 = self._stamp[3] - self.pixel_offset_col
					# Extract the subimage of which pixels were used in background:
					bpu = self.hdf['backgrounds_pixels_used'][ir1:ir2, ic1:ic2]
					self._aperture[bpu] |= 4
			else:
				# Load the aperture from the TPF:
				ir1 = self._stamp[0] - self._max_stamp[0]
				ir2 = self._stamp[1] - self._max_stamp[0]
				ic1 = self._stamp[2] - self._max_stamp[2]
				ic2 = self._stamp[3] - self._max_stamp[2]
				self._aperture = np.asarray(self.tpf['APERTURE'].data[ir1:ir2, ic1:ic2], dtype='int32')

				# Remove the flags for SPOC mask and centroids:
				self._aperture[(self._aperture & 2) != 0] -= 2
				self._aperture[(self._aperture & 8) != 0] -= 8

		return self._aperture

	#----------------------------------------------------------------------------------------------
	@property
	def catalog(self):
		"""
		Catalog of stars in the current stamp.

		The table contains the following columns:
		* ``starid``: TIC identifier.
		* ``tmag``: TESS magnitude.
		* ``ra``: Right ascension in degrees at time of observation.
		* ``dec``: Declination in degrees at time of observation.
		* ``row``: Pixel row on CCD.
		* ``column``: Pixel column on CCD.
		* ``row_stamp``: Pixel row relative to the stamp.
		* ``column_stamp``: Pixel column relative to the stamp.

		Returns:
			``astropy.table.Table``: Table with all known stars falling within the current stamp.

		Example:
			If ``pho`` is an instance of :py:class:`BasePhotometry`:

			>>> pho.catalog['tmag']
			>>> pho.catalog[('starid', 'tmag', 'row', 'column')]

		See Also:
			:py:meth:`catalog_attime`
		"""

		if not self._catalog:
			# Pixel-positions of the corners of the current stamp:
			corners = np.array([
				[self._stamp[2]-0.5, self._stamp[0]-0.5],
				[self._stamp[2]-0.5, self._stamp[1]-0.5],
				[self._stamp[3]-0.5, self._stamp[0]-0.5],
				[self._stamp[3]-0.5, self._stamp[1]-0.5]
			], dtype='float64')
			# Because the TPF world coordinate solution is relative to the stamp,
			# add the pixel offset to these:
			if self.datasource.startswith('tpf'):
				corners[:, 0] -= self.pixel_offset_col
				corners[:, 1] -= self.pixel_offset_row

			corners_radec = self.wcs.all_pix2world(corners, 0, ra_dec_order=True)

			# Select only the stars within the current stamp:
			# TODO: Change to opening in read-only mode: sqlite3.connect("file:" + self.catalog_file + "?mode=ro", uri=True). Requires Python 3.4
			with contextlib.closing(sqlite3.connect(self.catalog_file)) as conn:
				cursor = conn.cursor()
				cat = catalog_sqlite_search_footprint(cursor, corners_radec, columns='starid,ra,decl,tmag', buffer_size=5)
				cursor.close()

			if not cat:
				# Nothing was found. Return an empty table with the correct format:
				self._catalog = Table(
					names=('starid', 'ra', 'dec', 'tmag', 'column', 'row', 'column_stamp', 'row_stamp'),
					dtype=('int64', 'float64', 'float64', 'float32', 'float32', 'float32', 'float32', 'float32')
				)
			else:
				# Convert data to astropy table for further use:
				self._catalog = Table(
					rows=cat,
					names=('starid', 'ra', 'dec', 'tmag'),
					dtype=('int64', 'float64', 'float64', 'float32')
				)

				# Use the WCS to find pixel coordinates of stars in mask:
				pixel_coords = self.wcs.all_world2pix(np.column_stack((self._catalog['ra'], self._catalog['dec'])), 0, ra_dec_order=True)

				# Because the TPF world coordinate solution is relative to the stamp,
				# add the pixel offset to these:
				if self.datasource.startswith('tpf'):
					pixel_coords[:,0] += self.pixel_offset_col
					pixel_coords[:,1] += self.pixel_offset_row

				# Create columns with pixel coordinates:
				col_x = Column(data=pixel_coords[:,0], name='column', dtype='float32')
				col_y = Column(data=pixel_coords[:,1], name='row', dtype='float32')

				# Subtract the positions of the edge of the current stamp:
				pixel_coords[:,0] -= self._stamp[2]
				pixel_coords[:,1] -= self._stamp[0]

				# Add the pixel positions to the catalog table:
				col_x_stamp = Column(data=pixel_coords[:,0], name='column_stamp', dtype='float32')
				col_y_stamp = Column(data=pixel_coords[:,1], name='row_stamp', dtype='float32')

				self._catalog.add_columns([col_x, col_y, col_x_stamp, col_y_stamp])

		return self._catalog

	#----------------------------------------------------------------------------------------------
	@property
	def MovementKernel(self):
		"""
		Movement Kernel which allows calculation of positions on the focal plane as a function of time.
		Instance of :py:class:`image_motion.ImageMovementKernel`.
		"""
		if self._MovementKernel is None:
			default_movement_kernel = 'wcs' # The default kernel to use - set to 'hdf5' if we should use the one from prepare instead
			if self.datasource == 'ffi' and default_movement_kernel == 'wcs' and isinstance(self.hdf['wcs'], h5py.Group):
				self._MovementKernel = ImageMovementKernel(warpmode='wcs', wcs_ref=self.wcs)
				self._MovementKernel.load_series(self.lightcurve['time'] - self.lightcurve['timecorr'], [self.hdf['wcs'][dset][0] for dset in self.hdf['wcs']])
			elif self.datasource == 'ffi' and 'movement_kernel' in self.hdf:
				self._MovementKernel = ImageMovementKernel(warpmode=self.hdf['movement_kernel'].attrs.get('warpmode'))
				self._MovementKernel.load_series(self.lightcurve['time'] - self.lightcurve['timecorr'], self.hdf['movement_kernel'])
			elif self.datasource.startswith('tpf'):
				# Create translation kernel from the positions provided in the
				# target pixel file.
				# Load kernels from FITS file:
				kernels = np.column_stack((self.tpf[1].data['POS_CORR1'], self.tpf[1].data['POS_CORR2']))
				indx = np.isfinite(self.lightcurve['time']) & np.all(np.isfinite(kernels), axis=1)
				times = self.lightcurve['time'][indx] - self.lightcurve['timecorr'][indx]
				kernels = kernels[indx]
				# Find the timestamp closest to the reference time:
				refindx = find_nearest(times, self._catalog_reference_time)
				# Rescale kernels to the reference point:
				kernels = np.column_stack((self.tpf[1].data['POS_CORR1'][indx], self.tpf[1].data['POS_CORR2'][indx]))
				kernels[:, 0] -= kernels[refindx, 0]
				kernels[:, 1] -= kernels[refindx, 1]
				# Create kernel object:
				self._MovementKernel = ImageMovementKernel(warpmode='translation')
				self._MovementKernel.load_series(times, kernels)
			else:
				# If we reached this point, we dont have enough information to
				# define the ImageMovementKernel, so we should just return the
				# unaltered catalog:
				self._MovementKernel = ImageMovementKernel(warpmode='unchanged')

		return self._MovementKernel

	#----------------------------------------------------------------------------------------------
	def catalog_attime(self, time):
		"""
		Catalog of stars, calculated at a given time-stamp, so CCD positions are
		modified according to the measured spacecraft jitter.

		Parameters:
			time (float): Time in MJD when to calculate catalog.

		Returns:
			`astropy.table.Table`: Table with the same columns as :py:meth:`catalog`,
				but with ``column``, ``row``, ``column_stamp`` and ``row_stamp`` calculated
				at the given timestamp.

		See Also:
			:py:meth:`catalog`
		"""

		# If we didn't have enough information, just return the unchanged catalog:
		if self.MovementKernel.warpmode == 'unchanged':
			return self.catalog

		# Get the reference catalog:
		xy = np.column_stack((self.catalog['column'], self.catalog['row']))

		# Lookup the position corrections in CCD coordinates:
		jitter = self.MovementKernel.interpolate(time, xy)

		# Modify the reference catalog:
		cat = deepcopy(self.catalog)
		cat['column'] += jitter[:, 0]
		cat['row'] += jitter[:, 1]
		cat['column_stamp'] += jitter[:, 0]
		cat['row_stamp'] += jitter[:, 1]

		return cat

	#----------------------------------------------------------------------------------------------
	def delete_plots(self):
		"""
		Delete all files in :py:attr:`plot_folder`.

		If plotting is not enabled, this method does nothing and will therefore
		leave any existing files in the plot folder, should it already exists.
		"""
		logger = logging.getLogger(__name__)
		if self.plot and self.plot_folder is not None:
			for f in glob.iglob(os.path.join(self.plot_folder, '*')):
				logger.debug("Deleting plot '%s'", f)
				os.unlink(f)

	#----------------------------------------------------------------------------------------------
	def report_details(self, error=None, skip_targets=None):
		"""
		Report details of the processing back to the overlying scheduler system.

		Parameters:
			error (string): Error message the be logged with the results.
			skip_targets (list): List of starids that can be safely skipped.
		"""

		if skip_targets is not None:
			self._details['skip_targets'] = skip_targets

		if error is not None:
			if 'errors' not in self._details: self._details['errors'] = []
			self._details['errors'].append(error)

	#----------------------------------------------------------------------------------------------
	def do_photometry(self):
		"""
		Run photometry algorithm.

		This should fill the :py:attr:`lightcurve` table with all relevant parameters.

		Returns:
			The status of the photometry.

		Raises:
			NotImplementedError
		"""
		raise NotImplementedError("You have to implement the actual lightcurve extraction yourself... Sorry!")

	#----------------------------------------------------------------------------------------------
	def photometry(self, *args, **kwargs):
		"""
		Run photometry.

		Will run the :py:meth:`do_photometry` method and
		check some of the output and calculate various
		performance metrics.

		See Also:
			:py:meth:`do_photometry`
		"""

		# Run the photometry:
		self._status = self.do_photometry(*args, **kwargs)

		# Check that the status has been changed:
		if self._status == STATUS.UNKNOWN:
			raise Exception("STATUS was not set by do_photometry")

		# Calculate performance metrics if status was not an error:
		if self._status in (STATUS.OK, STATUS.WARNING):
			# Simple check that entire lightcurve is not NaN:
			if allnan(self.lightcurve['flux']):
				raise Exception("Final lightcurve is all NaNs")

			# Pick out the part of the lightcurve that has a good quality
			# and only use this subset to calculate the diagnostic metrics:
			indx_good = TESSQualityFlags.filter(self.lightcurve['quality'])
			goodlc = self.lightcurve[indx_good]

			# Calculate the mean flux level:
			self._details['mean_flux'] = nanmedian(goodlc['flux'])

			# Convert to relative flux:
			flux = (goodlc['flux'] / self._details['mean_flux']) - 1
			flux_err = np.abs(1/self._details['mean_flux']) * goodlc['flux_err']

			# Calculate noise metrics of the relative flux:
			self._details['variance'] = nanvar(flux, ddof=1)
			self._details['rms_hour'] = rms_timescale(goodlc['time'], flux, timescale=3600/86400)
			self._details['ptp'] = nanmedian(np.abs(np.diff(flux)))

			# Calculate the median centroid position in pixel coordinates:
			self._details['pos_centroid'] = nanmedian(goodlc['pos_centroid'], axis=0)

			# Calculate variability used e.g. in CBV selection of stars:
			indx = np.isfinite(goodlc['time']) & np.isfinite(flux) & np.isfinite(flux_err)
			# Do a more robust fitting with a third-order polynomial,
			# where we are catching cases where the fitting goes bad.
			# This happens in the test-data because there are so few points.
			with warnings.catch_warnings():
				warnings.filterwarnings('error', category=np.RankWarning)
				try:
					p = np.polyfit(goodlc['time'][indx], flux[indx], 3, w=1/flux_err[indx])
				except np.RankWarning:
					p = [0]

			# Calculate the variability as the standard deviation of the
			# polynomial-subtracted lightcurve devided by the median error:
			self._details['variability'] = nanstd(flux - np.polyval(p, goodlc['time'])) / nanmedian(flux_err)

			if self.final_phot_mask is not None:
				# Calculate the total number of pixels in the mask:
				self._details['mask_size'] = int(np.sum(self.final_phot_mask))

				# Measure the total flux on the edge of the stamp,
				# if the mask is touching the edge of the stamp:
				# The np.sum here should return zero on an empty array.
				edge = np.zeros_like(self.sumimage, dtype='bool')
				edge[:, (0,-1)] = True
				edge[(0,-1), 1:-1] = True
				self._details['edge_flux'] = np.nansum(self.sumimage[self.final_phot_mask & edge])

			if self.additional_headers and 'AP_CONT' in self.additional_headers:
				self._details['contamination'] = self.additional_headers['AP_CONT'][0]

		# Unpack any errors or warnings that were sent to the logger during the photometry:
		if self.message_queue:
			if not self._details.get('errors'):
				self._details['errors'] = []
			self._details['errors'] += self.message_queue
			self.message_queue.clear()

	#----------------------------------------------------------------------------------------------
	def save_lightcurve(self, output_folder=None, version=None):
		"""
		Save generated lightcurve to file.

		Parameters:
			output_folder (string, optional): Path to directory where to save lightcurve.
				If ``None`` the directory specified in the attribute ``output_folder`` is used.
			version (integer, optional): Version number to add to the FITS header and file name.
				If not set, the :py:attr:`version` is used.

		Returns:
			string: Path to the generated file.
		"""

		# Check if another output folder was provided:
		if output_folder is None:
			output_folder = self.output_folder
		if version is None:
			if self.version is None:
				raise ValueError("VERSION has not been set")
			else:
				version = self.version

		# Make sure that the directory exists:
		os.makedirs(output_folder, exist_ok=True)

		# Create sumimage before changing the self.lightcurve object:
		SumImage = self.sumimage

		# Propergate the Background Shenanigans flags into the quality flags if
		# one was detected somewhere in the final stamp in the given timestamp:
		quality = np.zeros_like(self.lightcurve['time'], dtype='int32')
		for k, flg in enumerate(self.pixelflags):
			if np.any(flg & PixelQualityFlags.BackgroundShenanigans != 0):
				quality[k] |= CorrectorQualityFlags.BackgroundShenanigans

		# Remove timestamps that have no defined time:
		# This is a problem in the Sector 1 alert data.
		indx = np.isfinite(self.lightcurve['time'])
		self.lightcurve = self.lightcurve[indx]

		# Get the current date for the files:
		now = datetime.datetime.now()

		# Extract which photmetric method is being used by checking the
		# name of the class that is running:
		photmethod = {
			'BasePhotometry': 'base',
			'AperturePhotometry': 'aperture',
			'PSFPhotometry': 'psf',
			'LinPSFPhotometry': 'linpsf',
			'HaloPhotometry': 'halo'
		}.get(self.__class__.__name__, None)

		# Primary FITS header:
		hdu = fits.PrimaryHDU()
		hdu.header['NEXTEND'] = (3 + int(hasattr(self, 'halo_weightmap')), 'number of standard extensions')
		hdu.header['EXTNAME'] = ('PRIMARY', 'name of extension')
		hdu.header['ORIGIN'] = ('TASOC/Aarhus', 'institution responsible for creating this file')
		hdu.header['DATE'] = (now.strftime("%Y-%m-%d"), 'date the file was created')
		hdu.header['TELESCOP'] = ('TESS', 'telescope')
		hdu.header['INSTRUME'] = ('TESS Photometer', 'detector type')
		hdu.header['FILTER'] = ('TESS', 'Photometric bandpass filter')
		hdu.header['OBJECT'] = ("TIC {0:d}".format(self.starid), 'string version of TICID')
		hdu.header['TICID'] = (self.starid, 'unique TESS target identifier')
		hdu.header['CAMERA'] = (self.camera, 'Camera number')
		hdu.header['CCD'] = (self.ccd, 'CCD number')
		hdu.header['SECTOR'] = (self.sector, 'Observing sector')

		# Versions:
		hdu.header['PROCVER'] = (__version__, 'Version of photometry pipeline')
		hdu.header['FILEVER'] = ('1.4', 'File format version')
		hdu.header['DATA_REL'] = (self.data_rel, 'Data release number')
		hdu.header['VERSION'] = (version, 'Version of the processing')
		hdu.header['PHOTMET'] = (photmethod, 'Photometric method used')

		# Object properties:
		if self.target['pm_ra'] is None or self.target['pm_decl'] is None:
			pmtotal = fits.card.Undefined()
		else:
			pmtotal = np.sqrt(self.target['pm_ra']**2 + self.target['pm_decl']**2)

		hdu.header['RADESYS'] = ('ICRS', 'reference frame of celestial coordinates')
		hdu.header['EQUINOX'] = (2000.0, 'equinox of celestial coordinate system')
		hdu.header['RA_OBJ'] = (self.target_pos_ra_J2000, '[deg] Right ascension')
		hdu.header['DEC_OBJ'] = (self.target_pos_dec_J2000, '[deg] Declination')
		hdu.header['PMRA'] = (fits.card.Undefined() if not self.target['pm_ra'] else self.target['pm_ra'], '[mas/yr] RA proper motion')
		hdu.header['PMDEC'] = (fits.card.Undefined() if not self.target['pm_decl'] else self.target['pm_decl'], '[mas/yr] Dec proper motion')
		hdu.header['PMTOTAL'] = (pmtotal, '[mas/yr] total proper motion')
		hdu.header['TESSMAG'] = (self.target['tmag'], '[mag] TESS magnitude')
		hdu.header['TEFF'] = (fits.card.Undefined() if not self.target['teff'] else self.target['teff'], '[K] Effective temperature')
		hdu.header['TICVER'] = (self.ticver, 'TESS Input Catalog version')

		# Cosmic ray headers:
		hdu.header['CRMITEN'] = (self.header['CRMITEN'], 'spacecraft cosmic ray mitigation enabled')
		hdu.header['CRBLKSZ'] = (self.header['CRBLKSZ'], '[exposures] s/c cosmic ray mitigation block siz')
		hdu.header['CRSPOC'] = (self.header['CRSPOC'], 'SPOC cosmic ray cleaning enabled')

		# Add K2P2 Settings to the header of the file:
		if self.additional_headers:
			for key, value in self.additional_headers.items():
				hdu.header[key] = value

		# Make binary table:
		# Define table columns:
		c1 = fits.Column(name='TIME', format='D', disp='D14.7', unit='BJD - 2457000, days', array=self.lightcurve['time'])
		c2 = fits.Column(name='TIMECORR', format='E', disp='E13.6', unit='d', array=self.lightcurve['timecorr'])
		c3 = fits.Column(name='CADENCENO', format='J', disp='I10', array=self.lightcurve['cadenceno'])
		c4 = fits.Column(name='FLUX_RAW', format='D', disp='E26.17', unit='e-/s', array=self.lightcurve['flux'])
		c5 = fits.Column(name='FLUX_RAW_ERR', format='D', disp='E26.17', unit='e-/s', array=self.lightcurve['flux_err'])
		c6 = fits.Column(name='FLUX_BKG', format='D', disp='E26.17', unit='e-/s', array=self.lightcurve['flux_background'])
		c7 = fits.Column(name='FLUX_CORR', format='D', disp='E26.17', unit='ppm', array=np.full_like(self.lightcurve['time'], np.nan))
		c8 = fits.Column(name='FLUX_CORR_ERR', format='D', disp='E26.17', unit='ppm', array=np.full_like(self.lightcurve['time'], np.nan))
		c9 = fits.Column(name='QUALITY', format='J', disp='B16.16', array=quality)
		c10 = fits.Column(name='PIXEL_QUALITY', format='J', disp='B16.16', array=self.lightcurve['quality'])
		c11 = fits.Column(name='MOM_CENTR1', format='D', disp='F10.5', unit='pixels', array=self.lightcurve['pos_centroid'][:, 0]) # column
		c12 = fits.Column(name='MOM_CENTR2', format='D', disp='F10.5', unit='pixels', array=self.lightcurve['pos_centroid'][:, 1]) # row
		c13 = fits.Column(name='POS_CORR1', format='D', disp='F14.7', unit='pixels', array=self.lightcurve['pos_corr'][:, 0]) # column
		c14 = fits.Column(name='POS_CORR2', format='D', disp='F14.7', unit='pixels', array=self.lightcurve['pos_corr'][:, 1]) # row

		tbhdu = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14], name='LIGHTCURVE')

		# Add proper comments on all the table headers:
		tbhdu.header.comments['TTYPE1'] = 'column title: data time stamps'
		tbhdu.header.comments['TFORM1'] = 'column format: 64-bit floating point'
		tbhdu.header.comments['TUNIT1'] = 'column units: Barycenter corrected TESS Julian'
		tbhdu.header.comments['TDISP1'] = 'column display format'

		tbhdu.header.comments['TTYPE2'] = 'column title: barycenter - timeslice correction'
		tbhdu.header.comments['TFORM2'] = 'column format: 32-bit floating point'
		tbhdu.header.comments['TUNIT2'] = 'column units: day'
		tbhdu.header.comments['TDISP2'] = 'column display format'

		tbhdu.header.comments['TTYPE3'] = 'column title: unique cadence number'
		tbhdu.header.comments['TFORM3'] = 'column format: signed 32-bit integer'
		tbhdu.header.comments['TDISP3'] = 'column display format'

		tbhdu.header.comments['TTYPE4'] = 'column title: photometric flux'
		tbhdu.header.comments['TFORM4'] = 'column format: 64-bit floating point'
		tbhdu.header.comments['TUNIT4'] = 'column units: electrons per second'
		tbhdu.header.comments['TDISP4'] = 'column display format'

		tbhdu.header.comments['TTYPE5'] = 'column title: photometric flux error'
		tbhdu.header.comments['TFORM5'] = 'column format: 64-bit floating point'
		tbhdu.header.comments['TUNIT5'] = 'column units: electrons per second'
		tbhdu.header.comments['TDISP5'] = 'column display format'

		tbhdu.header.comments['TTYPE6'] = 'column title: photometric background flux'
		tbhdu.header.comments['TFORM6'] = 'column format: 64-bit floating point'
		tbhdu.header.comments['TUNIT6'] = 'column units: electrons per second'
		tbhdu.header.comments['TDISP6'] = 'column display format'

		tbhdu.header.comments['TTYPE7'] = 'column title: corrected photometric flux'
		tbhdu.header.comments['TFORM7'] = 'column format: 64-bit floating point'
		tbhdu.header.comments['TUNIT7'] = 'column units: rel. flux in parts-per-million'
		tbhdu.header.comments['TDISP7'] = 'column display format'

		tbhdu.header.comments['TTYPE8'] = 'column title: corrected photometric flux error'
		tbhdu.header.comments['TFORM8'] = 'column format: 64-bit floating point'
		tbhdu.header.comments['TUNIT8'] = 'column units: parts-per-million'
		tbhdu.header.comments['TDISP8'] = 'column display format'

		tbhdu.header.comments['TTYPE9'] = 'column title: photometry quality flags'
		tbhdu.header.comments['TFORM9'] = 'column format: signed 32-bit integer'
		tbhdu.header.comments['TDISP9'] = 'column display format'

		tbhdu.header.comments['TTYPE10'] = 'column title: pixel quality flags'
		tbhdu.header.comments['TFORM10'] = 'column format: signed 32-bit integer'
		tbhdu.header.comments['TDISP10'] = 'column display format'

		tbhdu.header.comments['TTYPE11'] = 'column title: moment-derived column centroid'
		tbhdu.header.comments['TFORM11'] = 'column format: 64-bit floating point'
		tbhdu.header.comments['TUNIT11'] = 'column units: pixels'
		tbhdu.header.comments['TDISP11'] = 'column display format'

		tbhdu.header.comments['TTYPE12'] = 'column title: moment-derived row centroid'
		tbhdu.header.comments['TFORM12'] = 'column format: 64-bit floating point'
		tbhdu.header.comments['TUNIT12'] = 'column units: pixels'
		tbhdu.header.comments['TDISP12'] = 'column display format'

		tbhdu.header.comments['TTYPE13'] = 'column title: column position correction'
		tbhdu.header.comments['TFORM13'] = 'column format: 64-bit floating point'
		tbhdu.header.comments['TUNIT13'] = 'column units: pixels'
		tbhdu.header.comments['TDISP13'] = 'column display format'

		tbhdu.header.comments['TTYPE14'] = 'column title: row position correction'
		tbhdu.header.comments['TFORM14'] = 'column format: 64-bit floating point'
		tbhdu.header.comments['TUNIT14'] = 'column units: pixels'
		tbhdu.header.comments['TDISP14'] = 'column display format'

		tbhdu.header.set('INHERIT', True, 'inherit the primary header', after='TFIELDS')

		# Timestamps of start and end of timeseries:
		cadence = 120 if self.datasource.startswith('tpf') else 1800
		tdel = cadence/86400
		tstart = self.lightcurve['time'][0] - tdel/2
		tstop = self.lightcurve['time'][-1] + tdel/2
		tstart_tm = Time(tstart, 2457000, format='jd', scale='tdb')
		tstop_tm = Time(tstop, 2457000, format='jd', scale='tdb')
		telapse = tstop - tstart

		frametime = 2.0
		int_time = 1.98
		readtime = 0.02
		if self.header['CRMITEN']:
			deadc = (int_time * 2/self.header['CRBLKSZ']) / frametime
		else:
			deadc = int_time / frametime

		# Headers related to time to be added to LIGHTCURVE extension:
		tbhdu.header['TIMEREF'] = ('SOLARSYSTEM', 'barycentric correction applied to times')
		tbhdu.header['TIMESYS'] = ('TDB', 'time system is Barycentric Dynamical Time (TDB)')
		tbhdu.header['BJDREFI'] = (2457000, 'integer part of BTJD reference date')
		tbhdu.header['BJDREFF'] = (0.0, 'fraction of the day in BTJD reference date')
		tbhdu.header['TIMEUNIT'] = ('d', 'time unit for TIME, TSTART and TSTOP')
		tbhdu.header['TSTART'] = (tstart, 'observation start time in BTJD')
		tbhdu.header['TSTOP'] = (tstop, 'observation stop time in BTJD')
		tbhdu.header['DATE-OBS'] = (tstart_tm.utc.isot, 'TSTART as UTC calendar date')
		tbhdu.header['DATE-END'] = (tstop_tm.utc.isot, 'TSTOP as UTC calendar date')
		tbhdu.header['MJD-BEG'] = (tstart_tm.mjd, 'observation start time in MJD')
		tbhdu.header['MJD-END'] = (tstop_tm.mjd, 'observation start time in MJD')
		tbhdu.header['TELAPSE'] = (telapse, '[d] TSTOP - TSTART')
		tbhdu.header['LIVETIME'] = (telapse*deadc, '[d] TELAPSE multiplied by DEADC')
		tbhdu.header['DEADC'] = (deadc, 'deadtime correction')
		tbhdu.header['EXPOSURE'] = (telapse*deadc, '[d] time on source')
		tbhdu.header['XPOSURE'] = (frametime*deadc*self.num_frm, '[s] Duration of exposure')
		tbhdu.header['TIMEPIXR'] = (0.5, 'bin time beginning=0 middle=0.5 end=1')
		tbhdu.header['TIMEDEL'] = (tdel, '[d] time resolution of data')
		tbhdu.header['INT_TIME'] = (int_time, '[s] photon accumulation time per frame')
		tbhdu.header['READTIME'] = (readtime, '[s] readout time per frame')
		tbhdu.header['FRAMETIM'] = (frametime, '[s] frame time (INT_TIME + READTIME)')
		tbhdu.header['NUM_FRM'] = (self.num_frm, 'number of frames per time stamp')
		tbhdu.header['NREADOUT'] = (self.n_readout, 'number of read per cadence')

		# Make aperture image:
		# TODO: Pixels used in background calculation (value=4)
		mask = self.aperture
		if self.final_phot_mask is not None:
			mask[self.final_phot_mask] |= 2
		if self.final_position_mask is not None:
			mask[self.final_position_mask] |= 8

		# Construct FITS header for image extensions:
		wcs = self.wcs[self._stamp[0]:self._stamp[1], self._stamp[2]:self._stamp[3]]
		header = wcs.to_header(relax=True)
		header.set('INHERIT', True, 'inherit the primary header', before=0) # Add inherit header

		# Create aperture image extension:
		img_aperture = fits.ImageHDU(data=mask, header=header, name='APERTURE')

		# Make sumimage image:
		img_sumimage = fits.ImageHDU(data=SumImage, header=header, name="SUMIMAGE")

		# List of the HDUs what will be put into the FITS file:
		hdus = [hdu, tbhdu, img_sumimage, img_aperture]

		# For Halo photometry, also add the weightmap to the FITS file:
		if hasattr(self, 'halo_weightmap'):
			# Create binary table to hold the list of weightmaps for halo photometry:
			c1 = fits.Column(name='CADENCENO1', format='J', array=self.halo_weightmap['initial_cadence'])
			c2 = fits.Column(name='CADENCENO2', format='J', array=self.halo_weightmap['final_cadence'])
			c3 = fits.Column(name='SAT_PIXELS', format='J', array=self.halo_weightmap['sat_pixels'])
			c4 = fits.Column(
				name='WEIGHTMAP',
				format='%dE' % np.prod(SumImage.shape),
				dim='(%d,%d)' % (SumImage.shape[1], SumImage.shape[0]),
				array=self.halo_weightmap['weightmap']
			)

			wm = fits.BinTableHDU.from_columns([c1, c2, c3, c4], header=header, name='WEIGHTMAP')

			wm.header['TTYPE1'] = ('CADENCENO1', 'column title: first cadence number')
			wm.header['TFORM1'] = ('J', 'column format: signed 32-bit integer')
			wm.header['TDISP1'] = ('I10', 'column display format')

			wm.header['TTYPE2'] = ('CADENCENO2', 'column title: last cadence number')
			wm.header['TFORM2'] = ('J', 'column format: signed 32-bit integer')
			wm.header['TDISP2'] = ('I10', 'column display format')

			wm.header['TTYPE3'] = ('SAT_PIXELS', 'column title: Saturated pixels')
			wm.header['TFORM3'] = ('J', 'column format: signed 32-bit integer')
			wm.header['TDISP3'] = ('I10', 'column display format')

			wm.header['TTYPE4'] = ('WEIGHTMAP', 'column title: Weightmap')
			wm.header.comments['TFORM4'] = 'column format: image of 32-bit floating point'
			wm.header['TDISP4'] = ('E14.7', 'column display format')
			wm.header.comments['TDIM4'] = 'column dimensions: pixel aperture array'

			# Add the new table to the list of HDUs:
			hdus.append(wm)

		# File name to save the lightcurve under:
		filename = 'tess{starid:011d}-s{sector:02d}-c{cadence:04d}-dr{datarel:02d}-v{version:02d}-tasoc_lc.fits.gz'.format(
			starid=self.starid,
			sector=self.sector,
			cadence=cadence,
			datarel=self.data_rel,
			version=version
		)

		# Write to file:
		filepath = os.path.join(output_folder, filename)
		with fits.HDUList(hdus) as hdulist:
			hdulist.writeto(filepath, checksum=True, overwrite=True)

		# Store the output file in the details object for future reference:
		if os.path.realpath(output_folder).startswith(os.path.realpath(self.input_folder)):
			self._details['filepath_lightcurve'] = os.path.relpath(filepath, os.path.abspath(self.input_folder)).replace('\\', '/')
		else:
			self._details['filepath_lightcurve'] = os.path.relpath(filepath, self.output_folder_base).replace('\\', '/')

		return filepath
