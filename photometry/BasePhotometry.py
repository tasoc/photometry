#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The basic photometry class for the TASOC Photometry pipeline.
All other specific photometric algorithms will inherit from BasePhotometry.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
import six
from six.moves import range
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module="h5py") # they are simply annoying!
from astropy.io import fits
from astropy.table import Table, Column
import h5py
import sqlite3
import logging
import datetime
import os.path
from copy import deepcopy
#from astropy import time, coordinates, units
from astropy.wcs import WCS
import enum
from bottleneck import replace, nanmedian, ss
from .image_motion import ImageMovementKernel
from .quality import TESSQualityFlags
from .utilities import find_tpf_files
from .plots import plot_image, plt, save_figure

__docformat__ = 'restructuredtext'

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

class BasePhotometry(object):
	"""
	The basic photometry class for the TASOC Photometry pipeline.
	All other specific photometric algorithms will inherit from this.

	Attributes:
		starid (integer): TIC number of star being processed.
		input_folder (string): Root directory where files are loaded from.
		output_folder (string): Root directory where output files are saved.
		plot (boolean): Indicates wheter plots should be created as part of the output.
		plot_folder (string): Directory where plots are saved to.

		sector (integer): TESS observing sector.
		camera (integer): TESS camera (1-4).
		ccd (integer): TESS CCD (1-4).
		n_readout (integer): Number of frames co-added in each timestamp.

		target_mag (float): TESS magnitude of the main target.
		target_pos_ra (float): Right ascension of the main target at time of observation.
		target_pos_dec (float): Declination of the main target at time of observation.
		target_pos_ra_J2000 (float): Right ascension of the main target at J2000.
		target_pos_dec_J2000 (float): Declination of the main target at J2000.
		target_pos_column (flat): Main target CCD column position.
		target_pos_row (float): Main target CCD row position.
		target_pos_column_stamp (float): Main target CCD column position in stamp.
		target_pos_row_stamp (float): Main target CCD row position in stamp.
		wcs (``astropy.wcs.WCS`` object): World Coordinate system solution.

		lightcurve (``astropy.table.Table`` object): Table to be filled with an extracted lightcurve.
		final_mask (numpy.ndarray): Mask indicating which pixels were used in extraction of lightcurve. ``True`` if used, ``False`` otherwise.
		additional_headers (dict): Additional headers to be included in FITS files.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	def __init__(self, starid, input_folder, output_folder, datasource='ffi',
		camera=None, ccd=None, plot=False):
		"""
		Initialize the photometry object.

		Parameters:
			starid (int): TIC number of star to be processed.
			input_folder (string): Root directory where files are loaded from.
			output_folder (string): Root directory where output files are saved.
			datasource (string, optional): Source of the data. Options are ``'ffi'`` or ``'tpf'``. Default is ``'ffi'``.
			plot (boolean, optional): Create plots as part of the output. Default is ``False``.
			camera (integer, optional): TESS camera (1-4) to load target from (Only used for FFIs).
			ccd (integer, optional): TESS CCD (1-4) to load target from (Only used for FFIs).

		Raises:
			IOError: If starid could not be found in catalog.
			ValueError: On invalid datasource.

		"""

		logger = logging.getLogger(__name__)
		
		# Store the input:
		self.starid = starid
		self.input_folder = input_folder
		self.output_folder = output_folder
		self.plot = plot
		self.datasource = datasource

		logger.debug('DATASOURCE = %s', self.datasource)

		self._status = STATUS.UNKNOWN
		self._details = {}
		self.tpf = None
		self.hdf = None
		self._MovementKernel = None

		# Set directory where diagnostics plots should be saved to:
		self.plot_folder = None
		if self.plot:
			self.plot_folder = os.path.join(self.output_folder, 'plots', '{0:011d}'.format(self.starid))
			if not os.path.exists(self.plot_folder):
				os.makedirs(self.plot_folder) # "exists_ok=True" doesn't work in Python 2.7

		# Init table that will be filled with lightcurve stuff:
		self.lightcurve = Table()

		if self.datasource == 'ffi':
			# FIXME: These should also come from the catalog somehow
			#        They will be needed to find the correct input files
			if camera is None or ccd is None:
				raise ValueError("CAMERA and CCD keywords must be provided for FFI targets.")

			self.camera = camera # TESS camera.
			self.ccd = ccd # TESS CCD.

			logger.debug('CAMERA = %s', self.camera)
			logger.debug('CCD = %s', self.ccd)

			# Load stuff from the common HDF5 file:
			filepath_hdf5 = os.path.join(input_folder, 'camera{0:d}_ccd{1:d}.hdf5'.format(self.camera, self.ccd))
			self.hdf = h5py.File(filepath_hdf5, 'r')

			self.lightcurve['time'] = Column(self.hdf['time'], description='Time', dtype='float64', unit='BJD')
			if 'timecorr' in self.hdf:
				self.lightcurve['timecorr'] = Column(self.hdf['timecorr'], description='Barycentric time correction', unit='days', dtype='float32')
			else:
				self.lightcurve['timecorr'] = Column(np.zeros(len(self.lightcurve['time']), dtype='float32'), description='Barycentric time correction', unit='days', dtype='float32')
			self.lightcurve['cadenceno'] = Column(self.hdf['cadenceno'], description='Cadence number', dtype='int32')
			self.lightcurve['quality'] = Column(self.hdf['quality'], description='Quality flags', dtype='int32')

			# World Coordinate System solution:
			if isinstance(self.hdf['wcs'], h5py.Group):
				refindx = self.hdf['wcs'].attrs['ref_frame']
				hdr_string = self.hdf['wcs']['%04d' % refindx][0]
			else:
				hdr_string = self.hdf['wcs'][0]
			if not isinstance(hdr_string, six.string_types): hdr_string = hdr_string.decode("utf-8") # For Python 3
			self.wcs = WCS(header=fits.Header().fromstring(hdr_string)) # World Coordinate system solution.

			# Correct timestamps for light-travel time:
			# http://docs.astropy.org/en/stable/time/#barycentric-and-heliocentric-light-travel-time-corrections
			#star_coord = coordinates.SkyCoord(self.target_pos_ra_J2000, self.target_pos_dec_J2000, unit=units.deg, frame='icrs')
			#tess = coordinates.EarthLocation.of_site('greenwich')
			#times = time.Time(self.lightcurve['time'], format='mjd', scale='utc', location=tess)
			#self.lightcurve['timecorr'] = times.light_travel_time(star_coord, ephemeris='jpl')
			#self.lightcurve['time'] = times.tdb + self.lightcurve['timecorr']

			# Get shape of sumimage from hdf5 file:
			self._max_stamp = (0, self.hdf['sumimage'].shape[0], 0, self.hdf['sumimage'].shape[1])
			self.pixel_offset_row = self.hdf['images'].attrs.get('PIXEL_OFFSET_ROW', 0)
			self.pixel_offset_col = self.hdf['images'].attrs.get('PIXEL_OFFSET_COLUMN', 44) # Default for TESS data

			# Get info for psf fit Gaussian statistic:
			self.readnoise = self.hdf['images'].attrs.get('READNOIS', 10)
			self.gain = self.hdf['images'].attrs.get('GAIN', 100)
			self.n_readout = self.hdf['images'].attrs.get('NUM_FRM', 900) # Number of frames co-added in each timestamp (Default=TESS).

		elif self.datasource == 'tpf':
			# Find the target pixel file for this star:
			fname = find_tpf_files(input_folder, self.starid)
			if len(fname) == 1:
				fname = fname[0]
			elif len(fname) == 0:
				raise IOError("Target Pixel File not found")
			elif len(fname) > 1:
				raise IOError("Multiple Target Pixel Files found matching pattern")

			# Open the FITS file:
			self.tpf = fits.open(fname, mode='readonly', memmap=True)

			# Load sector, camera and CCD from the FITS header:
			self.sector = self.tpf[0].header['SECTOR']
			self.camera = self.tpf[0].header['CAMERA']
			self.ccd = self.tpf[0].header['CCD']

			# Extract the relevant information from the FITS file:
			self.lightcurve['time'] = Column(self.tpf[1].data.field('TIME'), description='Time', dtype='float64')
			self.lightcurve['timecorr'] = Column(self.tpf[1].data.field('TIMECORR'), description='Barycentric time correction', unit='days', dtype='float32')
			self.lightcurve['cadenceno'] = Column(self.tpf[1].data.field('CADENCENO'), description='Cadence number', dtype='int32')
			self.lightcurve['quality'] = Column(self.tpf[1].data.field('QUALITY'), description='Quality flags', dtype='int32')

			# World Coordinate System solution:
			self.wcs = WCS(header=self.tpf[2].header)

			# Get the positions of the stamp from the FITS header:
			self._max_stamp = (
				self.tpf[2].header['CRVAL2P'] - 1,
				self.tpf[2].header['CRVAL2P'] - 1 + self.tpf[2].header['NAXIS1'],
				self.tpf[2].header['CRVAL1P'] - 1,
				self.tpf[2].header['CRVAL1P'] - 1 + self.tpf[2].header['NAXIS2']
			)
			self.pixel_offset_row = self.tpf[2].header['CRVAL2P'] - 1
			self.pixel_offset_col = self.tpf[2].header['CRVAL1P'] - 1

			logger.debug('Max stamp size: (%d, %d)',
				self._max_stamp[1] - self._max_stamp[0],
				self._max_stamp[2] - self._max_stamp[3]
			)

			# Get info for psf fit Gaussian statistic:
			self.readnoise = self.tpf[1].header.get('READNOIA', 10) # FIXME: This only loads readnoise from channel A!
			self.gain = self.tpf[1].header.get('GAINA', 100) # FIXME: This only loads gain from channel A!
			self.n_readout = self.tpf[1].header.get('NUM_FRM', 900) # Number of frames co-added in each timestamp.

		else:
			raise ValueError("Invalid datasource: '%s'" % self.datasource)

		# The file to load the star catalog from:
		self.catalog_file = os.path.join(input_folder, 'catalog_camera{0:d}_ccd{1:d}.sqlite'.format(self.camera, self.ccd))
		self._catalog = None
		logger.debug('Catalog file: %s', self.catalog_file)

		# Load information about main target:
		conn = sqlite3.connect(self.catalog_file)
		conn.row_factory = sqlite3.Row
		cursor = conn.cursor()
		cursor.execute("SELECT ra,decl,ra_J2000,decl_J2000,tmag FROM catalog WHERE starid={0:d};".format(self.starid))
		target = cursor.fetchone()
		if target is None:
			raise IOError("Star could not be found in catalog: {0:d}".format(self.starid))
		self.target_tmag = target['tmag'] # TESS magnitude of the main target.
		self.target_pos_ra = target['ra'] # Right ascension of the main target at time of observation.
		self.target_pos_dec = target['decl'] # Declination of the main target at time of observation.
		self.target_pos_ra_J2000 = target['ra_J2000'] # Right ascension of the main target at J2000.
		self.target_pos_dec_J2000 = target['decl_J2000'] # Declination of the main target at J2000.
		cursor.execute("SELECT sector,reference_time FROM settings LIMIT 1;")
		target = cursor.fetchone()
		if target is not None:
			self._catalog_reference_time = target['reference_time']
			self.sector = target['sector']
		cursor.close()
		conn.close()

		# Define the columns that have to be filled by the do_photometry method:
		N = len(self.lightcurve['time'])
		self.lightcurve['flux'] = Column(length=N, description='Flux', dtype='float64')
		self.lightcurve['flux_background'] = Column(length=N, description='Background flux', dtype='float64')
		self.lightcurve['pos_centroid'] = Column(length=N, shape=(2,), description='Centroid position', unit='pixels', dtype='float64')

		# Init arrays that will be filled with lightcurve stuff:
		self.final_mask = None # Mask indicating which pixels were used in extraction of lightcurve.
		self.additional_headers = {} # Additional headers to be included in FITS files.

		# Project target position onto the pixel plane:
		self.target_pos_column, self.target_pos_row = self.wcs.all_world2pix(self.target_pos_ra, self.target_pos_dec, 0, ra_dec_order=True)
		if self.datasource == 'tpf':
			self.target_pos_column += self.pixel_offset_col
			self.target_pos_row += self.pixel_offset_row
		logger.info("Target column: %f", self.target_pos_column)
		logger.info("Target row: %f", self.target_pos_row)

		# Init the stamp:
		self._stamp = None
		self.target_pos_column_stamp = None # Main target CCD column position in stamp.
		self.target_pos_row_stamp = None # Main target CCD row position in stamp.
		self._set_stamp()
		self._sumimage = None

	def __enter__(self):
		return self

	def __exit__(self, *args):
		self.close()

	def close(self):
		"""Close photometry object and close all associated open file handles."""
		if self.hdf:
			self.hdf.close()
		if self.tpf:
			self.tpf.close()

	@property
	def status(self):
		"""The status of the photometry. From :py:class:`STATUS`."""
		return self._status

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
		Npixels = np.interp(self.target_tmag, np.array([8.0, 9.0, 10.0, 12.0, 14.0, 16.0]), np.array([350.0, 200.0, 125.0, 100.0, 50.0, 40.0]))
		Nrows = np.maximum(np.ceil(np.sqrt(Npixels)), 10)
		Ncolumns = np.maximum(np.ceil(np.sqrt(Npixels)), 10)
		return Nrows, Ncolumns

	def resize_stamp(self, down=None, up=None, left=None, right=None):
		"""
		Resize the stamp in a given direction.

		Parameters:
			down (int, optional): Number of pixels to extend downwards.
			up (int, optional): Number of pixels to extend upwards.
			left (int, optional): Number of pixels to extend left.
			right (int, optional): Number of pixels to extend right.

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
		self._stamp = tuple(self._stamp)

		# Count the number of times that we are resizing the stamp:
		self._details['stamp_resizes'] = self._details.get('stamp_resizes', 0) + 1

		# Return if the stamp actually changed:
		return self._set_stamp(old_stamp)

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
					int(self.target_pos_row) - Nrows//2,
					int(self.target_pos_row) + Nrows//2 + 1,
					int(self.target_pos_column) - Ncolumns//2,
					int(self.target_pos_column) + Ncolumns//2 + 1
				)
			else:
				Nrows = self._max_stamp[1] - self._max_stamp[0]
				Ncolumns = self._max_stamp[3] - self._max_stamp[2]
				logger.info("Setting default stamp with sizes (%d,%d)", Nrows, Ncolumns)
				self._stamp = self._max_stamp

		# Limit the stamp to not go outside the limits of the images:
		self._stamp = list(self._stamp)
		self._stamp[0] = int(np.maximum(self._stamp[0], self._max_stamp[0]))
		self._stamp[1] = int(np.minimum(self._stamp[1], self._max_stamp[1]))
		self._stamp[2] = int(np.maximum(self._stamp[2], self._max_stamp[2]))
		self._stamp[3] = int(np.minimum(self._stamp[3], self._max_stamp[3]))
		self._stamp = tuple(self._stamp)

		# Sanity checks:
		if self._stamp[0] > self._stamp[1] or self._stamp[2] > self._stamp[3]:
			raise ValueError("Invalid stamp selected")

		# Check if the stamp actually changed:
		if self._stamp == compare_stamp:
			return False

		# Calculate main target position in stamp:
		self.target_pos_row_stamp = self.target_pos_row - self._stamp[0]
		self.target_pos_column_stamp = self.target_pos_column - self._stamp[2]

		# Force sum-image and catalog to be recalculated next time:
		self._sumimage = None
		self._catalog = None
		return True

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

	@property
	def stamp(self):
		"""
		Tuple indicating the stamps position within the larger image.

		Returns:
			tuple: Tuple of (row_min, row_max, col_min, col_max).
		"""
		return self._stamp

	@property
	def images(self):
		"""
		Iterator that will loop through the image stamps.

		Returns:
			iterator: Iterator which can be used to loop through the image stamps.

		Note:
			The images has had the large-scale background subtracted. If needed
			the backgrounds can be added again from :py:func:`backgrounds`.

		Note:
			For each image, this function will actually load the necessary
			data from disk, so don't loop through it more than you absolutely
			have to to save I/O.

		Example:

			>>> pho = BasePhotometry(starid)
			>>> for img in pho.images:
			>>> 	print(img)

		See Also:
			:py:func:`backgrounds`
		"""
		if self.datasource == 'ffi':
			ir1 = self._stamp[0] - self.pixel_offset_row
			ir2 = self._stamp[1] - self.pixel_offset_row
			ic1 = self._stamp[2] - self.pixel_offset_col
			ic2 = self._stamp[3] - self.pixel_offset_col
			for k in range(len(self.hdf['images'])):
				yield self.hdf['images/%04d' % k][ir1:ir2, ic1:ic2]
		else:
			ir1 = self._stamp[0] - self._max_stamp[0]
			ir2 = self._stamp[1] - self._max_stamp[0]
			ic1 = self._stamp[2] - self._max_stamp[2]
			ic2 = self._stamp[3] - self._max_stamp[2]
			for k in range(self.tpf[1].header['NAXIS2']):
				yield self.tpf[1].data['FLUX'][k][ir1:ir2, ic1:ic2]

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
			:py:func:`images`
		"""
		if self.datasource == 'ffi':
			ir1 = self._stamp[0] - self.pixel_offset_row
			ir2 = self._stamp[1] - self.pixel_offset_row
			ic1 = self._stamp[2] - self.pixel_offset_col
			ic2 = self._stamp[3] - self.pixel_offset_col
			for k in range(len(self.hdf['backgrounds'])):
				yield self.hdf['backgrounds/%04d' % k][ir1:ir2, ic1:ic2]
		else:
			ir1 = self._stamp[0] - self._max_stamp[0]
			ir2 = self._stamp[1] - self._max_stamp[0]
			ic1 = self._stamp[2] - self._max_stamp[2]
			ic2 = self._stamp[3] - self._max_stamp[2]
			for k in range(self.tpf[1].header['NAXIS2']):
				yield self.tpf[1].data['FLUX_BKG'][k][ir1:ir2, ic1:ic2]

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
				self._sumimage = self.hdf['sumimage'][ir1:ir2, ic1:ic2]
			else:
				self._sumimage = np.zeros((self._stamp[1]-self._stamp[0], self._stamp[3]-self._stamp[2]), dtype='float64')
				Nimg = np.zeros_like(self._sumimage, dtype='int32')
				for k, img in enumerate(self.images):
					if TESSQualityFlags.filter(self.lightcurve['quality'][k]):
						Nimg += np.isfinite(img)
						replace(img, np.nan, 0)
						self._sumimage += img
				self._sumimage /= Nimg

			if self.plot:
				fig = plt.figure()
				ax = fig.add_subplot(111)
				plot_image(self._sumimage, ax=ax, offset_axes=(self._stamp[2], self._stamp[0]))
				ax.plot(self.target_pos_column, self.target_pos_row, 'r+')
				save_figure(os.path.join(self.plot_folder, 'sumimage'), fig=fig)
				plt.close(fig)

		return self._sumimage

	@property
	def catalog(self):
		"""
		Catalog of stars in the current stamp.

		The table contains the following columns:
		 * ``starid``:       TIC identifier.
		 * ``tmag``:         TESS magnitude.
		 * ``ra``:           Right ascension in degrees at time of observation.
		 * ``dec``:          Declination in degrees at time of observation.
		 * ``row``:          Pixel row on CCD.
		 * ``column``:       Pixel column on CCD.
		 * ``row_stamp``:    Pixel row relative to the stamp.
		 * ``column_stamp``: Pixel column relative to the stamp.

		Returns:
			``astropy.table.Table``: Table with all known stars falling within the current stamp.

		Example:
			If ``pho`` is an instance of :py:class:`BasePhotometry`:

			>>> pho.catalog['tmag']
			>>> pho.catalog[('starid', 'tmag', 'row', 'column')]

		See Also:
			:py:func:`catalog_attime`
		"""

		if not self._catalog:
			# Pixel-positions of the corners of the current stamp:
			corners = np.array([
				[self._stamp[2], self._stamp[0]],
				[self._stamp[2], self._stamp[1]],
				[self._stamp[3], self._stamp[0]],
				[self._stamp[3], self._stamp[1]]
			], dtype='int32')
			# Because the TPF world coordinate solution is relative to the stamp,
			# add the pixel offset to these:
			if self.datasource == 'tpf':
				corners[:, 0] -= self.pixel_offset_col
				corners[:, 1] -= self.pixel_offset_row

			# Convert the corners into (ra, dec) coordinates and find the max and min values:
			pixel_scale = 21.0 # Size of single pixel in arcsecs
			buffer_size = 3 # Buffer to add around stamp in pixels
			corners_radec = self.wcs.all_pix2world(corners, 0, ra_dec_order=True)
			radec_min = np.min(corners_radec, axis=0) - buffer_size*pixel_scale/3600.0
			radec_max = np.max(corners_radec, axis=0) + buffer_size*pixel_scale/3600.0

			# Upper and lower bounds on ra and dec:
			ra_min = radec_min[0]
			ra_max = radec_max[0]
			dec_min = radec_min[1]
			dec_max = radec_max[1]

			# Select only the stars within the current stamp:
			conn = sqlite3.connect(self.catalog_file)
			cursor = conn.cursor()
			query = "SELECT starid,ra,decl,tmag FROM catalog WHERE ra BETWEEN :ra_min AND :ra_max AND decl BETWEEN :dec_min AND :dec_max;"
			if dec_min < -90:
				# We are very close to the southern pole
				# Ignore everything about RA
				cursor.execute(query, {
					'ra_min': 0,
					'ra_max': 360,
					'dec_min': -90,
					'dec_max': dec_max
				})
			elif dec_max > 90:
				# We are very close to the northern pole
				# Ignore everything about RA
				cursor.execute(query, {
					'ra_min': 0,
					'ra_max': 360,
					'dec_min': dec_min,
					'dec_max': 90
				})
			elif ra_min < 0:
				cursor.execute("""SELECT starid,ra,decl,tmag FROM catalog WHERE ra <= :ra_max AND decl BETWEEN :dec_min AND :dec_max
				UNION
				SELECT starid,ra,decl,tmag FROM catalog WHERE ra BETWEEN :ra_min AND 360 AND decl BETWEEN :dec_min AND :dec_max;""", {
					'ra_min': 360 - abs(ra_min),
					'ra_max': ra_max,
					'dec_min': dec_min,
					'dec_max': dec_max
				})
			elif ra_max > 360:
				cursor.execute("""SELECT starid,ra,decl,tmag FROM catalog WHERE ra >= :ra_min AND decl BETWEEN :dec_min AND :dec_max
				UNION
				SELECT starid,ra,decl,tmag FROM catalog WHERE ra BETWEEN 0 AND :ra_max AND decl BETWEEN :dec_min AND :dec_max;""", {
					'ra_min': ra_min,
					'ra_max': ra_max - 360,
					'dec_min': dec_min,
					'dec_max': dec_max
				})
			else:
				cursor.execute(query, {
					'ra_min': ra_min,
					'ra_max': ra_max,
					'dec_min': dec_min,
					'dec_max': dec_max
				})

			cat = cursor.fetchall()
			cursor.close()
			conn.close()

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
				if self.datasource == 'tpf':
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

	def catalog_attime(self, time):
		"""
		Catalog of stars, calculated at a given time-stamp, so CCD positions are
		modified according to the measured spacecraft jitter.

		Parameters:
			time (float): Time in MJD when to calculate catalog.

		Returns:
			`astropy.table.Table`: Table with the same columns as :py:func:`catalog`, but with ``column``, ``row``, ``column_stamp`` and ``row_stamp`` calculated at the given timestamp.

		See Also:
			:py:func:`catalog`
		"""

		if self._MovementKernel is None:
			if self.datasource == 'ffi' and 'movement_kernel' in self.hdf:
				self._MovementKernel = ImageMovementKernel(warpmode=self.hdf['movement_kernel'].attrs.get('warpmode'))
				self._MovementKernel.load_series(self.lightcurve['time'], self.hdf['movement_kernel'])
			elif self.datasource == 'tpf':
				# Create translation kernel from the positions provided in the
				# target pixel file.
				# Find the timestamp closest to the reference time:
				refindx = np.searchsorted(self.lightcurve['time'], self._catalog_reference_time, side='left')
				if refindx > 0 and (refindx == len(self.lightcurve['time']) or abs(self._catalog_reference_time - self.lightcurve['time'][refindx-1]) < abs(self._catalog_reference_time - self.lightcurve['time'][refindx])):
					refindx -= 1
				# Load kernels from FITS file, and rescale the reference point:
				kernels = np.column_stack((self.tpf[1].data['POS_CORR1'], self.tpf[1].data['POS_CORR2']))
				kernels[:, 0] -= kernels[refindx, 0]
				kernels[:, 1] -= kernels[refindx, 1]
				# Create kernel:
				self._MovementKernel = ImageMovementKernel(warpmode='translation')
				self._MovementKernel.load_series(self.lightcurve['time'], kernels)
			else:
				# If we reached this point, we dont have enough information to
				# define the ImageMovementKernel, so we should just return the
				# unaltered catalog:
				self._MovementKernel = 'unchanged'

		# If we didn't have enough information, just return the unchanged catalog:
		if self._MovementKernel == 'unchanged':
			return self.catalog

		# Get the reference catalog:
		xy = np.column_stack((self.catalog['column'], self.catalog['row']))

		# Lookup the position corrections in CCD coordinates:
		jitter = self._MovementKernel.interpolate(time, xy)

		# Modify the reference catalog:
		cat = deepcopy(self.catalog)
		cat['column'] += jitter[:, 0]
		cat['row'] += jitter[:, 1]
		cat['column_stamp'] += jitter[:, 0]
		cat['row_stamp'] += jitter[:, 1]

		return cat

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

	def do_photometry(self):
		"""
		Run photometry algorithm.

		This should fill the following
		* self.lightcurve

		Returns:
			The status of the photometry.

		Raises:
			NotImplemented
		"""
		raise NotImplemented("You have to implement the actual lightcurve extraction yourself... Sorry!")


	def photometry(self, *args, **kwargs):
		"""
		Run photometry.

		Will run the :py:func:`do_photometry` method and
		check some of the output and calculate various
		performance metrics.

		See Also:
			:py:func:`do_photometry`
		"""

		# Run the photometry:
		self._status = self.do_photometry(*args, **kwargs)

		# Check that the status has been changed:
		if self._status == STATUS.UNKNOWN:
			raise Exception("STATUS was not set by do_photometry")

		# TODO: Calculate performance metrics
		if self._status in (STATUS.OK, STATUS.WARNING):
			self._details['mean_flux'] = nanmedian(self.lightcurve['flux'])
			self._details['variance'] = ss(self.lightcurve['flux'] - self._details['mean_flux']) / (len(self.lightcurve['flux'])-1)
			self._details['pos_centroid'] = nanmedian(self.lightcurve['pos_centroid'], axis=0)
			if self.final_mask is not None:
				self._details['mask_size'] = int(np.sum(self.final_mask))
			if self.additional_headers and 'AP_CONT' in self.additional_headers:
				self._details['contamination'] = self.additional_headers['AP_CONT'][0]

	def save_lightcurve(self, output_folder=None):
		"""
		Save generated lightcurve to file.

		Parameters:
			output_folder (string, optional): Path to directory where to save lightcurve. If ``None`` the directory specified in the attribute ``output_folder`` is used.
		"""

		# Check if another output folder was provided:
		if output_folder is None:
			output_folder = self.output_folder

		# Get the current date for the files:
		now = datetime.datetime.now()

		# Primary FITS header:
		hdu = fits.PrimaryHDU()
		hdu.header['NEXTEND'] = (3, 'number of standard extensions')
		hdu.header['EXTNAME'] = ('PRIMARY', 'name of extension')
		hdu.header['ORIGIN'] = ('TASOC/Aarhus', 'institution responsible for creating this file')
		hdu.header['DATE'] = (now.strftime("%Y-%m-%d"), 'date the file was created')
		hdu.header['TELESCOP'] = ('TESS', 'telescope')
		hdu.header['INSTRUME'] = ('TESS Photometer', 'detector type')
		hdu.header['OBJECT'] = ("TIC {0:d}".format(self.starid), 'string version of TICID')
		hdu.header['TICID'] = (self.starid, 'unique TESS target identifier')
		hdu.header['CAMERA'] = (self.camera, 'Camera number')
		hdu.header['CCD'] = (self.ccd, 'CCD number')
		hdu.header['SECTOR'] = (self.sector, 'Observing sector')
		#hdu.header['PHOTMET'] = ('aperture', 'Photometric method used')

		# Versions:
		#hdu.header['VERPIXEL'] = (__version__, 'version of K2P2 pipeline')
		#hdu.header['DATA_REL'] = (__version__, 'version of K2P2 pipeline')

		# Object properties:
		hdu.header['RADESYS'] = ('ICRS', 'reference frame of celestial coordinates')
		hdu.header['EQUINOX'] = (2000.0, 'equinox of celestial coordinate system')
		hdu.header['RA_OBJ'] = (self.target_pos_ra_J2000, '[deg] Right ascension')
		hdu.header['DEC_OBJ'] = (self.target_pos_dec_J2000, '[deg] Declination')
		hdu.header['TESSMAG'] = (self.target_tmag, '[mag] TESS magnitude')

		# Add K2P2 Settings to the header of the file:
		if self.additional_headers:
			for key, value in self.additional_headers.items():
				hdu.header[key] = value

		# Make binary table:
		# Define table columns:
		c1 = fits.Column(name='TIME', format='D', array=self.lightcurve['time'])
		c2 = fits.Column(name='TIMECORR', format='E', array=self.lightcurve['timecorr'])
		c3 = fits.Column(name='CADENCENO', format='J', array=self.lightcurve['cadenceno'])
		c4 = fits.Column(name='FLUX', format='D', unit='e-/s', array=self.lightcurve['flux'])
		c5 = fits.Column(name='FLUX_BKG', format='D', unit='e-/s', array=self.lightcurve['flux_background'])
		c6 = fits.Column(name='QUALITY', format='J', array=self.lightcurve['quality'])
		c7 = fits.Column(name='MOM_CENTR1', format='D', unit='pixels', array=self.lightcurve['pos_centroid'][:, 0]) # column
		c8 = fits.Column(name='MOM_CENTR2', format='D', unit='pixels', array=self.lightcurve['pos_centroid'][:, 1]) # row
		#c10 = fits.Column(name='POS_CORR1', format='E', unit='pixels', array=poscorr1) # column
		#c11 = fits.Column(name='POS_CORR2', format='E', unit='pixels', array=poscorr2) # row

		tbhdu = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8], name='LIGHTCURVE')

		tbhdu.header['TTYPE1'] = ('TIME', 'column title: data time stamps')
		tbhdu.header['TFORM1'] = ('D', 'column format: 64-bit floating point')
		tbhdu.header['TUNIT1'] = ('BJD', 'column unit: barycenter corrected JD')
		tbhdu.header['TDISP1'] = ('D14.7', 'column display format')

		tbhdu.header['TTYPE2'] = ('TIMECORR', 'column title: barycenter - timeslice correction')
		tbhdu.header['TFORM2'] = ('E', 'column format: 32-bit floating point')
		tbhdu.header['TUNIT2'] = ('d', 'column units: day')
		tbhdu.header['TDISP2'] = ('E13.6', 'column display format')

		tbhdu.header['TTYPE3'] = ('CADENCENO', 'column title: unique cadence number')
		tbhdu.header['TFORM3'] = ('J', 'column format: signed 32-bit integer')
		tbhdu.header['TDISP3'] = ('I10', 'column display format')

		tbhdu.header['TTYPE4'] = ('FLUX', 'column title: photometric flux')
		tbhdu.header['TFORM4'] = ('D', 'column format: 64-bit floating point')
		tbhdu.header['TUNIT4'] = ('e-/s', 'column units: electrons per second')
		tbhdu.header['TDISP4'] = ('E26.17', 'column display format')

		tbhdu.header['TTYPE5'] = ('FLUX_BKG', 'column title: photometric background flux')
		tbhdu.header['TFORM5'] = ('D', 'column format: 64-bit floating point')
		tbhdu.header['TUNIT5'] = ('e-/s', 'column units: electrons per second')
		tbhdu.header['TDISP5'] = ('E26.17', 'column display format')

		tbhdu.header['TTYPE6'] = ('QUALITY', 'column title: photometry quality flag')
		tbhdu.header['TFORM6'] = ('J', 'column format: signed 32-bit integer')
		tbhdu.header['TDISP6'] = ('B16.16', 'column display format')

		tbhdu.header['TTYPE7'] = ('MOM_CENTR1', 'column title: moment-derived column centroid')
		tbhdu.header['TFORM7'] = ('D', 'column format: 64-bit floating point')
		tbhdu.header['TUNIT7'] = ('pixel', 'column units: pixels')
		tbhdu.header['TDISP7'] = ('F10.5', 'column display format')

		tbhdu.header['TTYPE8'] = ('MOM_CENTR2', 'column title: moment-derived row centroid')
		tbhdu.header['TFORM8'] = ('D', 'column format: 64-bit floating point')
		tbhdu.header['TUNIT8'] = ('pixel', 'column units: pixels')
		tbhdu.header['TDISP8'] = ('F10.5', 'column display format')

		#tbhdu.header['TTYPE10'] = ('POS_CORR1', 'column title: column position correction')
		#tbhdu.header['TFORM10'] = ('E', 'column format: 32-bit floating point')
		#tbhdu.header['TUNIT10'] = ('pixel', 'column units: pixels')
		#tbhdu.header['TDISP10'] = ('F14.7', 'column display format')

		#tbhdu.header['TTYPE11'] = ('POS_CORR2', 'column title: row position correction')
		#tbhdu.header['TFORM11'] = ('E', 'column format: 32-bit floating point')
		#tbhdu.header['TUNIT11'] = ('pixel', 'column units: pixels')
		#tbhdu.header['TDISP11'] = ('F14.7', 'column display format')

		tbhdu.header.set('INHERIT', True, 'inherit the primary header', after='TFIELDS')

		# Make aperture image:
		# TODO: Pixels used in background calculation (value=4)
		cols, rows = self.get_pixel_grid()
		mask = np.asarray(np.isfinite(self.sumimage), dtype='int32')
		if self.final_mask is not None:
			mask[self.final_mask] += 10 # 2 + 8
		# Add mapping onto TESS output channels:
		mask[(45 <= cols) & (cols <= 556)] += 32 # CCD output A
		mask[(557 <= cols) & (cols <= 1068)] += 64 # CCD output B
		mask[(1069 <= cols) & (cols <= 1580)] += 128 # CCD output C
		mask[(1581 <= cols) & (cols <= 2092)] += 256 # CCD output D

		# Construct FITS header for image extensions:
		header = self.wcs.to_header(relax=True)
		header.set('INHERIT', True, 'inherit the primary header', before=0) # Add

		# Create aperture image extension:
		img_aperture = fits.ImageHDU(data=mask, header=header, name='APERTURE')

		# Make sumimage image:
		img_sumimage = fits.ImageHDU(data=self.sumimage, header=header, name="SUMIMAGE")

		# File name to save the lightcurve under:
		filename = 'tess{starid:011d}-s{sector:02d}-{cadence:s}-v{version:02d}-tasoc_lc.fits'.format(
			starid=self.starid,
			sector=self.sector,
			cadence={'ffi': 'ffi', 'tpf': '120'}[self.datasource],
			version=0 # FIXME: This needs to be set
		)

		# Write to file:
		with fits.HDUList([hdu, tbhdu, img_sumimage, img_aperture]) as hdulist:
			hdulist.writeto(os.path.join(output_folder, filename), checksum=True, overwrite=True)
