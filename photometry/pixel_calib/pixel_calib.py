#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pixel-level calibrations.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import os
import numpy as np
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
import astropy.units as u
import xml.etree.ElementTree as ET
import logging
from bottleneck import replace, nanmedian
from .CalibImage import CalibImage
from .polynomial import polynomial
from ..utilities import download_file
from ..plots import plt, plot_image

#--------------------------------------------------------------------------------------------------
class PixelCalibrator(object):

	def __init__(self, camera=None, ccd=None):

		assert camera in (1,2,3,4), "Invalid camera given"
		assert ccd in (1,2,3,4), "Invalid camera given"

		self.logger = logging.getLogger(__name__)
		self.logger.info("Starting calibration module")

		self.camera = camera
		self.ccd = ccd

		self.datadir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

		# General settings for TESS data:
		self.exposure_time = 1.98 * u.second
		self.frametransfer_time = 0.02 * u.second
		self.readout_time = 0.5 * u.second

		# Storage of calibration data:
		self._twodblack = None
		self._flatfield = None
		self._gain = None
		self._linearity = None
		self._undershoot = None

	def __enter__(self, *args, **kwargs):
		return self

	def __exit__(self, *args, **kwargs):
		pass

	def __repr__(self):
		return "<PixelCalibrator(camera={0:d}, ccd={1:d})>".format(self.camera, self.ccd)

	#----------------------------------------------------------------------------------------------
	@property
	def twodblack(self):
		"""
		2D black image for current stamp.

		Returns:
			CalibImage: 2D array with image of 2D black for current stamp.
		"""
		if self._twodblack is None:
			# FITS file containing flat-field model:
			blackfile = os.path.join(self.datadir, 'twodblack',
			   'tess2018324-{camera:d}-{ccd:d}-2dblack.fits.gz'.format(
					camera=self.camera,
					ccd=self.ccd
				))

			if not os.path.isfile(blackfile):
				url = 'https://tasoc.dk/pipeline/twodblack/' + os.path.basename(blackfile)
				os.makedirs(os.path.dirname(blackfile), exist_ok=True)
				download_file(url, blackfile)

			with fits.open(blackfile, mode='readonly', memmap=True) as hdu:
				self._twodblack = CalibImage(
					data=np.asarray(hdu[1].data),
					uncertainty=StdDevUncertainty(hdu[2].data),
					unit=u.electron # u.ct?
				)

		return self._twodblack

	#----------------------------------------------------------------------------------------------
	@property
	def flatfield(self):
		"""
		Flatfield for current stamp.

		Returns:
			CalibImage: 2D array with image of flatfield for current stamp.
		"""
		if self._flatfield is None:
			# FITS file containing flat-field model:
			flatfile = os.path.join(self.datadir, 'flatfield',
			   'tess2018323-{camera:d}-{ccd:d}-flat.fits.gz'.format(
					camera=self.camera,
					ccd=self.ccd
				))

			if not os.path.isfile(flatfile):
				url = 'https://tasoc.dk/pipeline/flatfield/' + os.path.basename(flatfile)
				os.makedirs(os.path.dirname(flatfile), exist_ok=True)
				download_file(url, flatfile)

			with fits.open(flatfile, mode='readonly', memmap=True) as hdu:
				self._flatfield = CalibImage(
					data=np.asarray(hdu[1].data),
					uncertainty=StdDevUncertainty(hdu[2].data),
					unit=u.dimensionless_unscaled
				)

		return self._flatfield

	#----------------------------------------------------------------------------------------------
	def plot_flatfield(self):

		fig = plt.figure()
		ax = fig.add_subplot(111)
		plot_image(self.flatfield.data, ax=ax, scale='linear', cmap='seismic', vmin=0.95, vmax=1.05, make_cbar=True)

	#----------------------------------------------------------------------------------------------
	def plot_twodblack(self):

		fig = plt.figure()
		ax = fig.add_subplot(111)
		plot_image(self.twodblack.data, ax=ax, scale='linear', make_cbar=True)

	#----------------------------------------------------------------------------------------------
	@property
	def virtual_rows(self):
		"""
		Flatfield for current stamp.

		Returns:
			numpy.array: 2D array with image of flatfield for current stamp.
		"""
		if self._virtual_rows is None:
			if self.datasource == 'ffi':
				pass
				#self.hdf['virtual_rows/%04d' % k][:, ic1:ic2]

			else:
				outputs = self.aperture & (32+64+128+256)

				vrowsfile = os.path.join(self.datadir, 'flatfield',
				   'tess2018234235059-s{sector:04d}-vrow-{camera:d}-{ccd:d}-{output:s}-0121-s_col.fits'.format(
						sector=self.sector,
						camera=self.camera,
						ccd=self.ccd,
						output='a'
					))

				ic1 = self._stamp[2]
				ic2 = self._stamp[3]

				with fits.open(vrowsfile, mode='readonly', memmap=True) as hdu:
					self._virtual_rows = np.asarray(hdu[1].data['VROW_CAL'][:, ic1:ic2])

		return self._virtual_rows

	#----------------------------------------------------------------------------------------------
	@property
	def gain(self):
		"""CCD gain model."""

		if self._gain is None:
			self.logger.info("Loading gain model...")
			gain_model = ET.parse(os.path.join(self.datadir, 'tess2018143203310-41006_100-gain.xml')).getroot()

			self._gain = {}
			for output in ('A', 'B', 'C', 'D'):
				gain = gain_model.find("./channel[@cameraNumber='{camera:d}'][@ccdNumber='{ccd:d}'][@ccdOutput='{output}']".format(
					camera=self.camera,
					ccd=self.ccd,
					output=output
				)).get('gainElectronsPerDN')

				self._gain[output] = float(gain) * u.photon/u.electron

		return self._gain

	#----------------------------------------------------------------------------------------------
	@property
	def linearity(self):
		"""CCD Linearity model."""

		if self._linearity is None:
			# Load the linearity models from XML files:
			self.logger.info("Loading linearity model...")
			linearity_model = ET.parse(os.path.join(self.datadir, 'tess2018143203310-41006_100-linearity.xml')).getroot()

			self._linearity = {}
			for output in ('A', 'B', 'C', 'D'):
				linpoly = linearity_model.find("./channel[@cameraNumber='{camera:d}'][@ccdNumber='{ccd:d}'][@ccdOutput='{output}']/linearityPoly".format(
					camera=self.camera,
					ccd=self.ccd,
					output=output
				))
				self._linearity[output] = polynomial(linpoly)

		return self._linearity

	#----------------------------------------------------------------------------------------------
	@property
	def undershoot(self):
		"""CCD undershoot model."""

		if self._undershoot is None:
			# Load the linearity models from XML files:
			self.logger.info("Loading undershoot model...")
			undershoot_model = ET.parse(os.path.join(self.datadir, 'undershoot.xml')).getroot()

			self._undershoot = {}
			for output in ('A', 'B', 'C', 'D'):
				ushoot = undershoot_model.find("./channel[@cameraNumber='{camera:d}'][@ccdNumber='{ccd:d}'][@ccdOutput='{output}']".format(
					camera=self.camera,
					ccd=self.ccd,
					output=output
				)).get('undershoot')
				self._undershoot[output] = float(ushoot)

		return self._undershoot

	#----------------------------------------------------------------------------------------------
	def apply_linearity_gain(self, img):
		"""
		CCD Linearity/gain correction.
		"""

		self.logger.info("Doing gain/linearity correction...")

		outputs = img.meta['aperture'][0,:] & (32 + 64 + 128 + 256)
		coadds = img.meta['coadds']

		#
		for output in np.unique(outputs):
			output_name = {32: 'A', 64: 'B', 128: 'C', 256: 'D'}[output]
			gain = self.gain[output_name]
			linpoly = self.linearity[output_name]

			#
			indx = (outputs == output)
			subimg = img[:, indx]

			# Evaluate the polynomial and multiply the image values by it:
			DN0 = subimg / coadds
			subimg = DN0 * linpoly(DN0.data)

			# Multiply with the gain:
			subimg *= gain * coadds

			img[:, indx] = subimg

		return img

	#----------------------------------------------------------------------------------------------
	def prepare_smear(self, img):
		"""
		TODO:
			 - Should we weight everything with the number of rows used in masked vs virtual regions?
			 - Should we take self.frametransfer_time into account?
			 - Cosmic ray rejection requires images before and after in time?
		"""

		#Remove cosmic rays in collateral data:
		# TODO: Can cosmic rays also show up in virtual pixels? If so, also include img.virtual_smear
		#index_collateral_cosmicrays = cosmic_rays(img.masked_smear)
		index_collateral_cosmicrays = np.zeros_like(img.masked_smear, dtype='bool')
		img.masked_smear[index_collateral_cosmicrays] = np.nan

		# Average the masked and virtual smear across their rows:
		masked_smear = nanmedian(img.masked_smear, axis=0)
		virtual_smear = nanmedian(img.virtual_smear, axis=0)

		# Estimate dark current:
		# TODO: Should this be self.frametransfer_time?
		fdark = nanmedian( masked_smear - virtual_smear * (self.exposure_time + self.readout_time) / self.exposure_time )
		img.dark = fdark # Save for later use
		self.logger.info('Dark current: %f', img.dark)
		if np.isnan(fdark):
			fdark = 0

		# Correct the smear regions for the dark current:
		masked_smear -= fdark
		virtual_smear -= fdark * (self.exposure_time + self.readout_time) / self.exposure_time

		# Weights from number of pixels in different regions:
		Nms = np.sum(~np.isnan(img.masked_smear), axis=0)
		Nvs = np.sum(~np.isnan(img.virtual_smear), axis=0)
		c_ms = Nms/np.maximum(Nms + Nvs, 1)
		c_vs = Nvs/np.maximum(Nms + Nvs, 1)

		# Weights as in Kepler where you only have one row in each sector:
		#g_ms = ~np.isnan(masked_smear)
		#g_vs = ~np.isnan(virtual_smear)
		#c_ms = g_ms/np.maximum(g_ms + g_vs, 1)
		#c_vs = g_vs/np.maximum(g_ms + g_vs, 1)

		# Estimate the smear for all columns, taking into account
		# that some columns could be missing:
		replace(masked_smear, np.nan, 0)
		replace(virtual_smear, np.nan, 0)
		smear = c_ms*masked_smear + c_vs*virtual_smear

		return fdark, smear, index_collateral_cosmicrays

	#--------------------------------------------------------------------------
	def apply_smear(self, img):
		"""CCD dark current and smear correction.
		"""
		self.logger.info("Doing smear correction...")

		# Load collateral data from collateral library:
		with h5py.File(self.collateral_library, 'r') as hdf:
			grp = hdf['cadence-%09d' % img.cadence_no]
			fdark = np.asarray(grp['dark'])
			fsmear = np.asarray(grp['smear'])
			collateral_columns = np.asarray(grp['collateral_columns'])

		# Correct the science pixels for dark current and smear:
		img -= fdark
		for k, col in enumerate(collateral_columns):
			img[img.columns == col] -= fsmear[k]

		return img

	#---------------------------------------------------------------------------------------------
	def apply_twodblack(self, img):
		"""2D black-level correction.

		TODO:
			 - Correct collateral pixels as well.
		"""
		self.logger.info("Doing 2D black correction...")
		return img - img.meta['coadds'] * self.twodblack[img.meta['index_rows'], img.meta['index_columns']]

	#---------------------------------------------------------------------------------------------
	def apply_undershoot(self, img):

		self.logger.info("Doing undershoot correction...")
		outputs = img.meta['aperture'][0,:] & (32 + 64 + 128 + 256)

		for output in np.unique(outputs):
			output_name = {32: 'A', 64: 'B', 128: 'C', 256: 'D'}[output]
			ushoot = self.undershoot[output_name]

			#
			indx = (outputs == output)
			subimg = img[:, indx]

			if output in ('B', 'D'):
				subimg[:, 1:] = subimg[:, 1:] + ushoot * subimg[:, :-1]
			else:
				subimg[:, :-1] = subimg[:, :-1] + ushoot * subimg[:, 1:]

		return img

	#--------------------------------------------------------------------------
	def apply_flatfield(self, img):
		"""CCD flat-field correction."""
		self.logger.info("Doing flatfield correction...")
		return img / self.flatfield[img.meta['index_rows'], img.meta['index_columns']]

	#--------------------------------------------------------------------------
	def to_counts_per_second(self, img):
		self.logger.info("Converting to counts per second...")
		return img / (self.exposure_time * img.meta['coadds'])

	#--------------------------------------------------------------------------
	def calibrate(self, img):
		"""Perform all calibration steps in sequence."""

		img = self.apply_twodblack(img)
		img = self.apply_undershoot(img)
		img = self.apply_linearity_gain(img)
		#img = self.apply_smear(img)
		#img = self.apply_flatfield(img)
		img = self.to_counts_per_second(img)

		return img

	#--------------------------------------------------------------------------
	def calibrate_tpf(self, tpf):

		meta = {}
		meta['coadds'] = int(tpf['PIXELS'].header['NREADOUT'])
		meta['index_rows'] = slice(tpf[2].header['CRVAL2P'] - 1, tpf[2].header['CRVAL2P'] + tpf[2].header['NAXIS2'] - 1, 1)
		meta['index_columns'] = slice(tpf[2].header['CRVAL1P'] - 1, tpf[2].header['CRVAL1P'] + tpf[2].header['NAXIS1'] - 1, 1)
		meta['aperture'] = np.asarray(tpf['APERTURE'].data, dtype='int32')

		# Mask of pixels not to include in any kind of calculations:
		mask = (meta['aperture'] & 1 == 0)

		#outputs = img.meta['aperture'][0, :] & (32 + 64 + 128 + 256)

		for k in range(len(tpf['PIXELS'].data['RAW_CNTS'])):
			# Construct CCDData image object:
			img = CalibImage(
				data=tpf['PIXELS'].data['RAW_CNTS'][k],
				uncertainty=StdDevUncertainty(np.sqrt(tpf['PIXELS'].data['RAW_CNTS'][k])),
				unit=u.electron, # u.ct? u.adu?
				mask=mask,
				meta=meta
			)

			# Calibrate the image:
			img_cal = self.calibrate(img.copy())
			print(img_cal.unit)

			plt.figure(figsize=(20,6))
			plt.subplot(131)
			plot_image(img.data, xlabel=None, ylabel=None, make_cbar=True, clabel='Raw counts (e/cadence)')
			plt.subplot(132)
			plot_image(tpf['PIXELS'].data['FLUX'][k] + tpf['PIXELS'].data['FLUX_BKG'][k], xlabel=None, ylabel=None, make_cbar=True)
			plt.subplot(133)
			plot_image(img_cal.data, xlabel=None, ylabel=None, make_cbar=True)

			# Store the resulting calibrated image in the FITS hdu:
			tpf['PIXELS'].data['FLUX'][k][:] = np.asarray(img_cal.data)
			tpf['PIXELS'].data['FLUX_ERR'][k][:] = img_cal.uncertainty.array

			break

		return tpf
