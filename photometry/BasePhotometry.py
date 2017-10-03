#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 18:12:33 2017

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
from six.moves import range, zip
import numpy as np
import astropy.io.fits as pf
from astropy.table import Table, Column
import h5py
import logging
import datetime
import os.path
#from astropy import time, coordinates, units
from astropy.wcs import WCS

global_catalog = None

class BasePhotometry(object):
	"""Base photometry class.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	"""Status to be returned by do_photometry on success."""
	STATUS_OK = 0

	"""Status to be returned by do_photometry on error."""
	STATUS_ERROR = 1

	"""Status to be returned by do_photometry on warning."""
	STATUS_WARNING = 2

	def __init__(self, starid, mode='ffi'):
		"""Initialize the photometry object.

		:param starid: TIC number of star to be processed
		"""

		self.starid = starid
		self.mode = mode

		self._stamp = None
		self._catalog = None
		input_folder = r'C:\Users\au195407\Documents\GitHub\photometry\input'

		global global_catalog
		if global_catalog is None:
			cat = np.genfromtxt(os.path.join(input_folder, 'catalog.txt.gz'), skip_header=1, usecols=(0,1,2,3,4,5,6), dtype='float64')
			cat = np.column_stack((np.arange(1, cat.shape[0]+1, dtype='int64'), cat))

			global_catalog = Table(cat,
				names=('starid', 'ra', 'dec', 'pm_ra', 'pm_dec', 'x', 'y', 'tmag'),
				dtype=('int64', 'float64','float64','float32','float32','float64','float64','float32')
			)
			global_catalog.add_index('starid', unique=True)
			global_catalog.add_index('ra')
			global_catalog.add_index('dec')

		# Load information about main target:
		# TODO: HOW?
		target = global_catalog.loc['starid', starid]
		self.target_tmag = target['tmag']
		self.target_pos_ra = target['ra']
		self.target_pos_dec = target['dec']

		# TODO: These should also come from the catalog somehow
		#       They will be needed to find the correct input files
		self.camera = 1
		self.ccd = 1

		# Init table that will be filled with lightcurve stuff:
		self.lightcurve = Table()

		if self.mode == 'ffi':
			# Load stuff from the common HDF5 file:
			filepath_hdf5 = os.path.join(input_folder, 'test2.hdf5')
			self.hdf = h5py.File(filepath_hdf5, 'r')

			self.lightcurve['time'] = Column(self.hdf['time'], description='Time', dtype='float64', unit='BJD')
			self.lightcurve['timecorr'] = Column(np.zeros(len(self.hdf['time']), dtype='float32'), description='Barycentric time correction', unit='days', dtype='float32')
			self.lightcurve['cadenceno'] = Column(self.hdf['cadenceno'], description='Cadence number', dtype='int32')

			hdr = pf.Header().fromstring(self.hdf['wcs'][0])
			self.wcs = WCS(header=hdr)

			# Correct timestamps for light-travel time:
			# http://docs.astropy.org/en/stable/time/#barycentric-and-heliocentric-light-travel-time-corrections
			#ip_peg = coordinates.SkyCoord(self.target_pos_ra, self.target_pos_dec, unit=units.deg, frame='icrs')
			#greenwich = coordinates.EarthLocation.of_site('greenwich')
			#times = time.Time(self.lightcurve['time'], format='mjd', scale='utc', location=greenwich)
			#self.lightcurve['timecorr'] = times.light_travel_time(ip_peg, ephemeris='jpl')
			#self.lightcurve['time'] = times.tdb + self.lightcurve['timecorr']

		elif self.mode == 'stamp':
			with pf.open(mode, mode='readonly', memmap=True) as hdu:

				self.lightcurve['time'] = Column(hdu[1].data.field('TIME'), description='Time', dtype='float64')
				self.lightcurve['timecorr'] = Column(hdu[1].data.field('TIMECORR'), description='Barycentric time correction', unit='days', dtype='float32')
				self.lightcurve['cadenceno'] = Column(hdu[1].data.field('CADENCENO'), description='Cadence number', dtype='int32')

				self.wcs = WCS(header=hdu[2].header)

		# Define the columns that have to be filled by the do_photometry method:
		N = len(self.lightcurve['time'])
		self.lightcurve['flux'] = Column(length=N, description='Flux', dtype='float64')
		self.lightcurve['flux_background'] = Column(length=N, description='Background flux', dtype='float64')
		self.lightcurve['quality'] = Column(length=N, description='Quality flags', dtype='int32')
		self.lightcurve['pos_centroid'] = Column(length=N, shape=(2,), description='Centroid position', unit='pixels', dtype='float64')

		# Init arrays that will be filled with lightcurve stuff:
		self.final_mask = None

		self.target_pos_column, self.target_pos_row = self.wcs.all_world2pix(self.target_pos_ra, self.target_pos_dec, 0, ra_dec_order=True)
		#print(self.target_pos_row, self.target_pos_column)

		# Init the stamp:
		self._stamp = None
		self.set_stamp()
		self._sumimage = None

	def __enter__(self):
		return self

	def __exit__(self, *args):
		self.close()

	def close(self):
		if self.hdf:
			self.hdf.close()

	def get_sumimage(self):
		if self._sumimage is None:
			if self.mode == 'ffi':
				self._sumimage = self.hdf['sumimage'][self._stamp[0]:(self._stamp[1]+1), self._stamp[2]:(self._stamp[3]+1)]
			else:
				self._sumimage = np.zeros((self._stamp[1]-self._stamp[0], self._stamp[3]-self._stamp[2]), dtype='float64')
				for k, img in enumerate(self.images):
					self._sumimage += img
				#self._sumimage /= N

		return self._sumimage

	def default_stamp(self):
		# Decide how many pixels to use based on lookup tables as a function of Tmag:
		Npixels = np.interp(self.target_tmag, np.array([8.0, 9.0, 10.0, 12.0, 14.0, 16.0]), np.array([350.0, 200.0, 125.0, 100.0, 50.0, 40.0]))
		Nrows = np.maximum(np.ceil(np.sqrt(Npixels)), 10)
		Ncolumns = np.maximum(np.ceil(np.sqrt(Npixels)), 10)
		return Nrows, Ncolumns

	def resize_stamp(self, down=None, up=None, left=None, right=None):
		"""
		Resize the stamp in a given direction.
		
		Parameters
		----------
		:param down: Number of pixels to extend the stamp down.
		:param up: Number of pixels to extend the stamp up.
		:param left: Number of pixels to extend the stamp left.
		:param right: Number of pixels to extend the stamp right.
		"""
		
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
		self.set_stamp()

	def set_stamp(self):
		"""

		NB: Stamp is zero-based counted from the TOP of the image
		"""

		logger = logging.getLogger(__name__)

		if not self._stamp:
			Nrows, Ncolumns = self.default_stamp()
			logger.info("Setting default stamp with sizes (%d,%d)", Nrows, Ncolumns)
			self._stamp = (
				int(self.target_pos_row) - Nrows//2,
				int(self.target_pos_row) + Nrows//2 + 1,
				int(self.target_pos_column) - Ncolumns//2,
				int(self.target_pos_column) + Ncolumns//2 + 1
			)

		self._stamp = list(self._stamp)
		self._stamp[0] = int(np.maximum(self._stamp[0], 0))
		self._stamp[1] = int(np.minimum(self._stamp[1], 2048))
		self._stamp[2] = int(np.maximum(self._stamp[2], 0))
		self._stamp[3] = int(np.minimum(self._stamp[3], 2048))
		self._stamp = tuple(self._stamp)

		# Sanity checks:
		if self._stamp[0] > self._stamp[1] or self._stamp[2] > self._stamp[3]:
			raise ValueError("Invalid stamp selected")

		# Force sum-image and catalog to be recalculated next time:
		self._sumimage = None
		self._catalog = None

	def get_pixel_grid(self):
		"""Returns mesh-grid of the pixels (1-based) in the stamp."""
		return np.meshgrid(
			np.arange(self._stamp[2]+1, self._stamp[3]+1, 1, dtype='int32'),
			np.arange(self._stamp[0]+1, self._stamp[1]+1, 1, dtype='int32')
		)

	@property
	def stamp(self):
		return self._stamp

	@property
	def images(self):
		"""Iterator that will loop through the image stamps."""
		for img in self.hdf['images']:
			filename = os.path.basename(img).rstrip('.gz')
			with pf.open(os.path.join('input/images', filename), memmap=True, mode='readonly') as hdu:
				data = hdu[0].data[self._stamp[0]:self._stamp[1], self._stamp[2]:self._stamp[3]]

			yield data

	@property
	def backgrounds(self):
		"""Iterator that will loop through the background-image stamps."""
		for k in range(self.hdf['backgrounds'].shape[2]):
			yield self.hdf['backgrounds'][self._stamp[0]:self._stamp[1], self._stamp[2]:self._stamp[3], k]

	@property
	def sumimage(self):
		if self._sumimage is None:
			if self.mode == 'ffi':
				self._sumimage = self.hdf['sumimage'][self._stamp[0]:self._stamp[1], self._stamp[2]:self._stamp[3]]
			else:
				self._sumimage = np.zeros((self._stamp[1]-self._stamp[0], self._stamp[3]-self._stamp[2]), dtype='float64')
				for img, bck in zip(self.images, self.backgrounds):
					self._sumimage += (img - bck)
				#self._sumimage /= N

		return self._sumimage

	@property
	def catalog(self):
		"""Catalog of stars in the current stamp.

		Returns an astropy.table.Table object"""

		if not self._catalog:
			# Pixel-positions of the corners of the current stamp:
			corners = np.array([
				[self._stamp[2], self._stamp[0]],
				[self._stamp[2], self._stamp[1]],
				[self._stamp[3], self._stamp[0]],
				[self._stamp[3], self._stamp[1]]
			], dtype='int32')

			# Convert the corners into (ra, dec) coordinates and find the max and min values:
			corners_radec = self.wcs.all_pix2world(corners, 0, ra_dec_order=True)
			radec_min = np.min(corners_radec, axis=0)
			radec_max = np.max(corners_radec, axis=0)

			# Select only the stars within the current stamp:
			# TODO: This could be improved with an index!
			# TODO: Include proper-motion movement to "now" => Modify (ra, dec).
			indx = (global_catalog['ra'] >= radec_min[0]) & (global_catalog['ra'] <= radec_max[0]) & (global_catalog['dec'] >= radec_min[1]) & (global_catalog['dec'] <= radec_max[1])
			# global_catalog.loc['ra', radec_min[0]:radec_max[0]]
			# global_catalog.loc['dec', radec_min[1]:radec_max[1]]

			self._catalog = global_catalog[indx]

			# Use the WCS to find pixel coordinates of stars in mask:
			pixel_coords = self.wcs.all_world2pix(np.column_stack((self._catalog['ra'], self._catalog['dec'])), 0, ra_dec_order=True)

			# Subtract the positions of the edge of the current stamp:
			pixel_coords[:,0] -= self._stamp[2]
			pixel_coords[:,1] -= self._stamp[0]

			# Add the pixel positions to the catalog table:
			col_x = Column(data=pixel_coords[:,0], name='row', dtype='float32')
			col_y = Column(data=pixel_coords[:,1], name='column', dtype='float32')
			self._catalog.add_columns([col_x, col_y])

			# Sort the catalog after brightness:
			#self._catalog.sort('tmag')

		return self._catalog

	def do_photometry(self):
		"""Run photometry algorithm.

		This should fill the following
		* self.flux
		* self.flux_background
		* self.quality
		* self.pos_centroid

		Returns the status of the photometry.
		"""
		raise NotImplemented()

	def save_lightcurve(self):
		"""Save generated lightcurve to file."""

		# Get the current date for the files:
		now = datetime.datetime.now()

		# Primary FITS header:
		hdu = pf.PrimaryHDU()
		hdu.header['NEXTEND'] = (3, 'number of standard extensions')
		hdu.header['EXTNAME'] = ('PRIMARY', 'name of extension')
		hdu.header['ORIGIN'] = ('TASOC/Aarhus', 'institution responsible for creating this file')
		hdu.header['DATE'] = (now.strftime("%Y-%m-%d"), 'date the file was created')
		#hdu.header['KEPLERID'] = (star, 'Kepler identifier')
		#hdu.header['OBJECT'] = (kasocutil.star_identifier(star), 'string version of KEPLERID')
		#hdu.header['TELESCOP'] = ('Kepler', 'telescope')
		#hdu.header['INSTRUME'] = ('Kepler Photometer', 'detector type')
		#hdu.header['OBSMODE'] = (cadence, 'observing mode')
		#hdu.header['TRACEID'] = (traceid, 'unique id in KASOC of processed file')
		#hdu.header['CAMPAIGN'] = (campaign, 'Kepler quarters used in this file')
		#hdu.header['VERPIXEL'] = (__version__, 'version of K2P2 pipeline')
		#hdu.header['SUBTARG'] = (subtarget, 'sub-target number from K2P2')

		# Add K2P2 Settings to the header of the file:
		#if custom_mask:
		#	hdu.header['KP_MODE']	= 'Custom Mask'
		#else:
		#	hdu.header['KP_MODE'] = 'Normal'
		#hdu.header['KP_SUBKG'] = (bool(subtract_background), 'K2P2 subtract background?')
		#hdu.header['KP_THRES'] = (sumimage_threshold, 'K2P2 sum-image threshold')
		#hdu.header['KP_MIPIX'] = (min_no_pixels_in_mask, 'K2P2 min pixels in mask')
		#hdu.header['KP_MICLS'] = (min_for_cluster, 'K2P2 min pix. for cluster')
		#hdu.header['KP_CLSRA'] = (cluster_radius, 'K2P2 cluster radius')
		#hdu.header['KP_WS'] = (bool(ws), 'K2P2 watershed segmentation')
		#hdu.header['KP_WSALG'] = (ws_alg, 'K2P2 watershed weighting')
		#hdu.header['KP_WSBLR'] = (ws_blur, 'K2P2 watershed blur')
		#hdu.header['KP_WSTHR'] = (ws_threshold, 'K2P2 watershed threshold')
		#hdu.header['KP_WSFOT'] = (ws_footprint, 'K2P2 watershed footprint')
		#hdu.header['KP_EX'] = (bool(extend_overflow), 'K2P2 extend overflow')

		# Make binary table:
		# Define table columns:
		c1 = pf.Column(name='TIME', format='D', array=self.lightcurve['time'])
		c2 = pf.Column(name='TIMECORR', format='E', array=self.lightcurve['timecorr'])
		c3 = pf.Column(name='CADENCENO', format='J', array=self.lightcurve['cadenceno'])
		c4 = pf.Column(name='FLUX', format='D', unit='e-/s', array=self.lightcurve['flux'])
		c5 = pf.Column(name='QUALITY', format='J', array=self.lightcurve['quality'])
		c6 = pf.Column(name='FLUX_BKG', format='D', unit='e-/s', array=self.lightcurve['flux_background'])
		c7 = pf.Column(name='MOM_CENTR1', format='D', unit='pixels', array=self.lightcurve['pos_centroid'][:, 0]) # column
		c8 = pf.Column(name='MOM_CENTR2', format='D', unit='pixels', array=self.lightcurve['pos_centroid'][:, 1]) # row
		#c10 = pf.Column(name='POS_CORR1', format='E', unit='pixels', array=poscorr1) # column
		#c11 = pf.Column(name='POS_CORR2', format='E', unit='pixels', array=poscorr2) # row

		tbhdu = pf.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7, c8])
		tbhdu.header['EXTNAME'] = ('LIGHTCURVE', 'name of extension')
		tbhdu.header['TTYPE1'] = ('TIME', 'column title: data time stamps')
		tbhdu.header['TFORM1'] = ('D', 'column format: 64-bit floating point')
		#tbhdu.header['TUNIT1'] = (ori_table_header['TUNIT1'], ori_table_header.comments['TUNIT1'])
		#tbhdu.header['TDISP1'] = (ori_table_header['TDISP1'], 'column display format')
		tbhdu.header['TTYPE2'] = ('TIMECORR', 'column title: barycenter - timeslice correction')
		tbhdu.header['TFORM2'] = ('E', 'column format: 32-bit floating point')
		#tbhdu.header['TUNIT2'] = (ori_table_header['TUNIT2'], ori_table_header.comments['TUNIT2'])
		#tbhdu.header['TDISP2'] = (ori_table_header['TDISP2'], 'column display format')
		tbhdu.header['TTYPE3'] = ('CADENCENO', 'column title: data time stamps')
		tbhdu.header['TFORM3'] = ('J', 'column format: signed 32-bit integer')
		#tbhdu.header['TDISP3'] = (ori_table_header['TDISP3'], 'column display format')
		tbhdu.header['TTYPE4'] = ('FLUX', 'column title: Relative photometric flux')
		tbhdu.header['TFORM4'] = ('D', 'column format: 64-bit floating point')
		tbhdu.header['TUNIT4'] = ('e-/s', 'column units: electrons per second')
		tbhdu.header['TDISP4'] = ('E26.17', 'column display format')
		tbhdu.header['TTYPE5'] = ('QUALITY', 'column title: photometry quality flag')
		tbhdu.header['TFORM5'] = ('J', 'column format: signed 32-bit integer')
		tbhdu.header['TDISP5'] = ('B16.16', 'column display format')
		tbhdu.header['TTYPE6'] = ('FLUX_BKG', 'column title: photometric background flux')
		tbhdu.header['TFORM6'] = ('D', 'column format: 64-bit floating point')
		tbhdu.header['TUNIT6'] = ('e-/s', 'column units: electrons per second')
		tbhdu.header['TDISP6'] = ('E26.17', 'column display format')
		tbhdu.header['TTYPE7'] = ('MOM_CENTR1', 'column title: moment-derived column centroid')
		tbhdu.header['TFORM7'] = ('D', 'column format: 64-bit floating point')
		tbhdu.header['TUNIT7'] = ('pixel', 'column units: pixels')
		tbhdu.header['TDISP7'] = ('F10.5', 'column display format')
		tbhdu.header['TTYPE8'] = ('MOM_CENTR2', 'column title: moment-derived row centroid')
		tbhdu.header['TFORM8'] = ('D', 'column format: 64-bit floating point')
		tbhdu.header['TUNIT8'] = ('pixel', 'column units: pixels')
		tbhdu.header['TDISP8'] = ('F10.5', 'column display format')
		tbhdu.header['TTYPE9'] = ('FLUX_BKG_SUM', 'column title: pho. background flux in sumimage')
		tbhdu.header['TFORM9'] = ('D', 'column format: 64-bit floating point')
		tbhdu.header['TUNIT9'] = ('e-/s', 'column units: electrons per second')
		tbhdu.header['TDISP9'] = ('E26.17', 'column display format')
		tbhdu.header['TTYPE10'] = ('POS_CORR1', 'column title: column position correction')
		tbhdu.header['TFORM10'] = ('E', 'column format: 32-bit floating point')
		tbhdu.header['TUNIT10'] = ('pixel', 'column units: pixels')
		tbhdu.header['TDISP10'] = ('F14.7', 'column display format')
		tbhdu.header['TTYPE11'] = ('POS_CORR2', 'column title: row position correction')
		tbhdu.header['TFORM11'] = ('E', 'column format: 32-bit floating point')
		tbhdu.header['TUNIT11'] = ('pixel', 'column units: pixels')
		tbhdu.header['TDISP11'] = ('F14.7', 'column display format')

		# Make aperture image:
		img_aperture = []
		if self.final_mask is not None:
			mask = np.ones_like(self.final_mask, dtype='int32')
			mask[self.final_mask] = 3
			img_aperture = pf.ImageHDU(data=mask, header=self.wcs.to_header(), name='APERTURE') # header=ori_mask_header,

		# Make sumimage image:
		img_sumimage = pf.ImageHDU(data=self.get_sumimage(), header=self.wcs.to_header(), name="SUMIMAGE") # header=ori_mask_header,

		# Write to file:
		with pf.HDUList([hdu, tbhdu, img_sumimage, img_aperture]) as hdulist:
			output_folder = 'output'
			hdulist.writeto(os.path.join(output_folder, 'tess{0:09d}.fits'.format(self.starid)), checksum=True, overwrite=True)
