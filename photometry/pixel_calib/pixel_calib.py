#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pixel-level calibrations.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
.. codeauthor:: Timothy R. White
"""

import os
import numpy as np
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
import astropy.units as u
import xml.etree.ElementTree as ET
import logging
from bottleneck import replace, nanmedian, nanmean
from .CalibImage import CalibImage
from .polynomial import polynomial
from ..utilities import download_file
from ..plots import plt, plot_image

#--------------------------------------------------------------------------------------------------
def get_tpf_col(img, coltype=None):
	"""
	Load collateral data matching the given Target Pixel File.
	"""

	if coltype not in ('vrow', 'smrow', 'lvcol', 'tvcol'):
		raise ValueError()

	logger = logging.getLogger(__name__)

	sector = img.meta['sector']
	camera = img.meta['camera']
	ccd = img.meta['ccd']
	cadence = img.meta['cadence']
	suffix = {120: 'col', 20: 'fast-col'}[cadence]

	# Find the outputs from the aperture extension:
	outputs = np.unique(img.aperture & (32 + 64 + 128 + 256))

	collateral = [] # np.zeros((len(outputs),flux.shape[0],2078,11), dtype='int32')
	for output in outputs:

		colfile = os.path.join('.',
			f's{sector:04d}',
			f'camera{camera:d}-ccd{ccd:d}',
			f'tess2018349182459-s{sector:04d}-{coltype:s}-{camera:d}-{ccd:d}-{output:s}-0126-s_{suffix:s}.fits')

		if not os.path.isfile(colfile):
			url = f'https://archive.stsci.edu/missions/tess/ffi/s{sector:04d}/2018/349/{camera:d}-{ccd:d}/' + os.path.basename(colfile)
			os.makedirs(os.path.dirname(colfile), exist_ok=True)
			download_file(url, colfile)

		logger.info("... loading trailing virtual column file %s", colfile)
		with fits.open(colfile, mode='readonly') as hdu:
			raw = np.asarray(hdu[1].data[coltype.upper() + '_RAW'], dtype='int32')
			collateral = np.column_stack((collateral, raw))

	print(collateral)
	return collateral

#--------------------------------------------------------------------------------------------------
class PixelCalibrator(object):

	def __init__(self, camera=None, ccd=None):
		"""
		"""
		# Basic checks of input:
		if camera not in (1,2,3,4):
			raise ValueError("Invalid camera given")
		if ccd not in (1,2,3,4):
			raise ValueError("Invalid ccd given")

		self.logger = logging.getLogger(__name__)
		self.logger.info("Starting calibration module")

		self.camera = camera
		self.ccd = ccd

		# Directory where calibration files will be stored:
		self.datadir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

		self.cache_dir = '.'

		# General settings for TESS data:
		self.exposure_time = 1.98 * u.second
		self.frametransfer_time = 0.02 * u.second
		self.readout_time = 0.5 * u.second

		# Storage of calibration data:
		self._readnoise = None
		self._twodblack = None
		self._flatfield = None
		self._gain = None
		self._linearity = None
		self._undershoot = None

	#----------------------------------------------------------------------------------------------
	def __enter__(self, *args, **kwargs):
		return self

	#----------------------------------------------------------------------------------------------
	def __exit__(self, *args, **kwargs):
		self.close()

	#----------------------------------------------------------------------------------------------
	def __repr__(self):
		return f"<PixelCalibrator(camera={self.camera:d}, ccd={self.ccd:d})>"

	#----------------------------------------------------------------------------------------------
	def close(self):
		pass

	#----------------------------------------------------------------------------------------------
	@property
	def twodblack(self):
		"""
		2D black image for current stamp.

		Returns:
			:class:`CalibImage`: 2D array with image of 2D black for current stamp.
		"""
		if self._twodblack is None:
			# FITS file containing flat-field model:
			blackfile = os.path.join(self.datadir, 'twodblack',
			   f'tess2018324-{self.camera:d}-{self.ccd:d}-2dblack.fits.gz')

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
			:class:`CalibImage`: 2D array with image of flatfield for current stamp.
		"""
		if self._flatfield is None:
			# FITS file containing flat-field model:
			flatfile = os.path.join(self.datadir, 'flatfield',
			   f'tess2018323-{self.camera:d}-{self.ccd:d}-flat.fits.gz')

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
	@property
	def readnoise(self):
		"""CCD read-noise model."""
		if self._readnoise is None:
			self.logger.info("Loading read-noise model...")
			readnoise_model = ET.parse(os.path.join(self.datadir, 'tess2018143203310-41006_100-read-noise.xml')).getroot()

			self._readnoise = {}
			for output in ('A', 'B', 'C', 'D'):
				readnoise = readnoise_model.find(f"./channel[@cameraNumber='{self.camera:d}'][@ccdNumber='{self.ccd:d}'][@ccdOutput='{output:s}']").get('readNoiseDNPerRead')
				self._readnoise[output] = float(readnoise) * u.electron

		return self._readnoise

	#----------------------------------------------------------------------------------------------
	@property
	def gain(self):
		"""CCD gain model."""
		if self._gain is None:
			self.logger.info("Loading gain model...")
			gain_model = ET.parse(os.path.join(self.datadir, 'tess2018143203310-41006_100-gain.xml')).getroot()

			self._gain = {}
			for output in ('A', 'B', 'C', 'D'):
				gain = gain_model.find(f"./channel[@cameraNumber='{self.camera:d}'][@ccdNumber='{self.ccd:d}'][@ccdOutput='{output:s}']").get('gainElectronsPerDN')
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
				linpoly = linearity_model.find(f"./channel[@cameraNumber='{self.camera:d}'][@ccdNumber='{self.ccd:d}'][@ccdOutput='{output:s}']/linearityPoly")
				self._linearity[output] = polynomial(linpoly)

		return self._linearity

	#----------------------------------------------------------------------------------------------
	@property
	def undershoot(self):
		"""CCD undershoot model."""
		if self._undershoot is None:
			# Load the linearity models from XML files:
			self.logger.info("Loading undershoot model...")
			undershoot_model = ET.parse(os.path.join(self.datadir, 'tess-undershoot.xml')).getroot()

			self._undershoot = {}
			for output in ('A', 'B', 'C', 'D'):
				ushoot = undershoot_model.find(f"./channel[@cameraNumber='{self.camera:d}'][@ccdNumber='{self.ccd:d}'][@ccdOutput='{output:s}']").get('undershoot')
				self._undershoot[output] = float(ushoot)

		return self._undershoot

	#----------------------------------------------------------------------------------------------
	def apply_linearity_gain(self, img):
		"""
		Apply CCD Linearity/gain correction.
		"""
		self.logger.info("Doing gain/linearity correction...")

		outputs = img.aperture & (32 + 64 + 128 + 256)
		coadds = img.meta['coadds']

		#
		for output in np.unique(outputs):
			output_name = {32: 'A', 64: 'B', 128: 'C', 256: 'D'}[output]
			gain = self.gain[output_name]
			linpoly = self.linearity[output_name]

			#
			indx = (outputs == output)
			subimg = img[indx]

			# Evaluate the polynomial and multiply the image values by it:
			DN0 = subimg / coadds
			subimg = DN0 * linpoly(DN0.data)

			# Multiply with the gain:
			subimg *= gain * coadds

			img[indx] = subimg

		return img

	#----------------------------------------------------------------------------------------------
	def apply_dark_smear(self, img):
		"""
		TODO:
			 - Should we weight everything with the number of rows used in masked vs virtual regions?
			 - Should we take self.frametransfer_time into account?
			 - Cosmic ray rejection requires images before and after in time?
		"""

		# Short-hand for the meta-data:
		img_masked_smear = np.asarray(img.meta['masked_smear'], dtype='float64')
		img_virtual_smear = np.asarray(img.meta['virtual_smear'], dtype='float64')

		# Remove cosmic rays in collateral data:
		# TODO: Can cosmic rays also show up in virtual pixels? If so, also include img.virtual_smear
		#index_collateral_cosmicrays = cosmic_rays(img_masked_smear)
		index_collateral_cosmicrays = np.zeros_like(img_masked_smear, dtype='bool')
		img_masked_smear[index_collateral_cosmicrays] = np.nan

		# Average the masked and virtual smear across their rows:
		masked_smear = nanmedian(img_masked_smear, axis=0)
		virtual_smear = nanmedian(img_virtual_smear, axis=0)

		# Estimate dark current:
		masked_virtual_ratio = float((self.exposure_time + self.frametransfer_time) / self.exposure_time)
		fdark = nanmedian( masked_smear - virtual_smear * masked_virtual_ratio )
		img.meta['dark'] = fdark * u.electron # Save for later use
		self.logger.info('Dark current: %s', img.meta['dark'])
		if np.isnan(fdark):
			fdark = 0

		# Correct the smear regions for the dark current:
		masked_smear -= fdark
		virtual_smear -= fdark * masked_virtual_ratio

		# Weights from number of pixels in different regions:
		Nms = np.sum(~np.isnan(img_masked_smear), axis=0)
		Nvs = np.sum(~np.isnan(img_virtual_smear), axis=0)
		c_ms = Nms/np.maximum(Nms + Nvs, 1)
		c_vs = Nvs/np.maximum(Nms + Nvs, 1)

		# Estimate the smear for all columns, taking into account
		# that some columns could be missing:
		replace(masked_smear, np.nan, 0)
		replace(virtual_smear, np.nan, 0)
		img.meta['smear'] = c_ms*masked_smear + c_vs*virtual_smear

		self.logger.info("Doing smear correction...")

		# Correct the science pixels for dark current and smear:
		img -= img.meta['dark']
		#for k in range(44, 44+len(img.meta['smear'])):
		#	img[:2048, k] -= img.meta['smear'][k] * u.electron

		return img

	#----------------------------------------------------------------------------------------------
	def apply_smear_tim(self, img, smrow, smear='alternate'):
		"""
		Parameters:
			smear (str): 'standard', or 'alternate'
				If the string 'standard' is passed, will calculate the smear correction directly from the collateral files.
				If the string 'alternate' is passed, smear correction will be estimated from the target pixel file.
		"""

		# In TPFs the masked smear rows and virtual rows have to be fetched separately:
		if img.meta['cadence'] == 'ffi':
			smrow = img.mask['smear_rows']
		else:
			smrow = get_tpf_col(img, coltype='smrow', output=output_name)

		# Add mean black level per outout:
		outputs = img.aperture & (32 + 64 + 128 + 256)
		for output in np.unique(outputs):
			output_name = {32: 'A', 64: 'B', 128: 'C', 256: 'D'}[output]
			msk = (outputs == output)

			# Origin and dimensions of the target pixel file relative to the smear data
			if output_name == 'A':
				sx1 = 44
			if output_name == 'B':
				sx1 = 556
			if output_name == 'C':
				sx1 = 1068
			if output_name == 'D':
				sx1 = 1580

			x1 = self.hdu[2].header['CRVAL1P']-1-sx1
			dx = self.hdu[2].header['NAXIS1']
			dy = self.hdu[2].header['NAXIS2']

			if smrow.shape[0] == 2: # TPF goes over two outputs
				newsmrow = np.zeros((smrow.shape[1],10,1024))
				newsmrow[:,:,:512] = smrow[0]
				newsmrow[:,:,512] = smrow[1]
				smrow = newsmrow
			else:
				smrow = smrow[0]

			# Calculate the 'regular' smear correction
			smcor = nanmedian(smrow, axis=1)

			if smear == 'alternate':
				# Find which rows are below the bleed column
				medimg = nanmedian(img, axis=0)

				maxrows = []
				for col in np.arange(dx):
					maxrows.append(np.argmax(medimg[:,col][medimg[:,col] < 1e5]))

				# 10 row buffer to (hopefully) ensure we're always below the bleed column
				maxrow = np.nanmin(maxrows) - 10

				# Build a mask from the pixels with the lowest flux in each column
				smmask = np.zeros((dy,dx), dtype='bool')
				for col in np.arange(dx):
					idx = medimg[:maxrow,col] < np.percentile(medimg[:maxrow,col],20)
					smmask[:maxrow,col][idx] = True

				# From the background pixels calculate a smear correction + background for each column as a function of time
				smbkgd = nanmean(img.T[smmask.T].reshape(dx,int(np.sum(smmask)/dx),img.shape[0]),axis=1).T

				# Estimate the background and remove it from the smear correction.
				# Here we can use the median level for the regular smear correction
				msk = (np.arange(smrow.shape[2]) < x1) | (np.arange(smrow.shape[2]) > x1 + dx)
				bkgd = np.nanmin(smbkgd,axis=1) - nanmedian(smcor[:,msk], axis=1)
				sm = smbkgd - np.tile(bkgd[:,np.newaxis], dx)

			elif smear == 'standard':
				sm = smcor[:, x1:x1+dx]

			# Apply the new correction
			img[msk] -= np.expand_dims(sm, axis=1)

		return img

	#----------------------------------------------------------------------------------------------
	def apply_twodblack(self, img):
		"""
		Apply 2D black-level correction.
		"""
		self.logger.info("Doing 2D black correction...")

		print(img.aperture)

		# Subtract fixed offset:
		fxdoff = img.meta['header'].get('FXDOFF', None)
		if fxdoff is not None:
			img -= int(fxdoff) * u.electron

		print(img.aperture)

		# Add mean black level per outout:
		outputs = img.aperture & (32 + 64 + 128 + 256)
		for output in np.unique(outputs):
			output_name = {32: 'A', 64: 'B', 128: 'C', 256: 'D'}[output]
			indx = (outputs == output)
			img[indx] += float(img.meta['header']['MEANBLC' + output_name]) * u.electron

		# Subtract 2D-black model:
		return img - (img.meta['coadds'] * self.twodblack[img.rows, img.cols])

	#----------------------------------------------------------------------------------------------
	def onedblack(self, img, output):

		#
		sector = img.meta['sector']
		cadence = img.meta['cadence']

		# Cache file:
		cache_file = os.path.join(self.cache_dir,
			f's{sector:04d}',
			f'camera{self.camera:d}-ccd{self.ccd:d}',
			f'tess-s{sector:04d}-c{cadence:04d}-{self.camera:d}-{self.ccd:d}-{output:s}-1dblack.npy')

		# If a cached file exists, load it and return that:
		if os.path.isfile(cache_file):
			return np.load(cache_file)

		if cadence == 'ffi':
			tvcol = img.meta['trailing_virtual_columns']
		else:
			tvcol = get_tpf_col(img, coltype='tvcol')

		# First, apply the normal 2D Black correction to the collateral data:
		tvcol = self.apply_twodblack(tvcol)

		# For each frame, fit a polynomial to the TV col values, choosing the order using the AIC
		nrows = tvcol.shape[1]
		x = np.arange(nrows)/nrows-0.5
		polyorders = np.arange(20, 50)

		aics = np.empty(len(polyorders), dtype='float64')
		_onedblack = np.empty([tvcol.shape[0], len(x)], dtype='float64')
		with warnings.catch_warnings():
			warnings.simplefilter('ignore', np.RankWarning)

			for frame in range(tvcol.shape[0]):
				# Average over all but the last column, which seems to be off
				y = nanmean(tvcol[frame,:,:-1], axis=1)

			self.logger.info("... loading trailing virtual column file "+tvcolfile)

			with fits.open(tvcolfile, mode='readonly') as hdu:
				tv_cal[idx] = hdu[1].data['TVCOL_RAW']

				_onedblack[frame, :] = p(x)

		# Save this for later reuse:
		np.save(cache_file, _onedblack)
		return _onedblack

		sm_cal = np.zeros((len(set(outputs)), flux.shape[0], 10, 512))
		for idx, output in enumerate(set(outputs)):

			smrowfile = path.split('/')[-1].split('-')[0]+'-'+path.split('/')[-1].split('-')[1]+'-smrow-'+str(self.camera)+'-'+str(self.ccd)+'-'+output.lower()+'-'+path.split('/')[-1].split('-')[3]+'-s_col.fits'
			self.logger.info("... loading smear row file "+smrowfile)

			with fits.open(smrowfile, mode='readonly') as hdu:
				sm_cal[idx] = hdu[1].data['SMROW_RAW']

		return sm_cal

	#----------------------------------------------------------------------------------------------
	def apply_onedblack(self, img):
		"""
		Apply time-dependent 1D black correction.
		"""
		self.logger.info("Doing 1D black correction...")

		cadenceno = img.meta['cadenceno']

		# Subtract 1D black level per output:
		outputs = img.aperture & (32 + 64 + 128 + 256)
		for output in np.unique(outputs):
			indx = (outputs == output)
			output_name = {32: 'a', 64: 'b', 128: 'c', 256: 'd'}[output]

			# This will contain an array with time down the rows and the rows along the columns
			# - Yes, that is very confusing!
			onedblack = self.onedblack(img, output_name)

			img[indx] -= np.expand_dims(onedblack[cadenceno, img.rows], axis=1)

		return img

	#----------------------------------------------------------------------------------------------
	def apply_undershoot(self, img):

		self.logger.info("Doing undershoot correction...")
		outputs = img.aperture[0,:] & (32 + 64 + 128 + 256)

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

	#----------------------------------------------------------------------------------------------
	def apply_flatfield(self, img):
		"""CCD flat-field correction."""
		self.logger.info("Doing flatfield correction...")
		return img / self.flatfield[img.rows, img.cols]

	#----------------------------------------------------------------------------------------------
	def to_electrons_per_second(self, img):
		"""Convert image to electrons per seconds."""
		self.logger.info("Converting to counts per second...")
		return img / (self.exposure_time * img.meta['coadds'])

	#----------------------------------------------------------------------------------------------
	def calibrate(self, img, smear='alternate'):
		"""
		Perform all calibration steps for a single image.

		Parameters:
			img (:class:`CalibImage`): Image to be calibrated.
			smear (str): 'standard', or 'alternate'
				If the string 'standard' is passed, will calculate the smear correction directly from the collateral files.
				If the string 'alternate' is passed, smear correction will be estimated from the target pixel file.

		Returns:
			:class:`CalibImage`: Calibrated image.
		"""

		#sm_cal = self.smear_rows

		# Apply 2D black to flux images and smear images:
		print(img.aperture)
		img = self.apply_twodblack(img)
		#sm_cal = self.apply_twodblack(sm_cal)

		# 1D black:
		img = self.apply_onedblack(img)
		#sm_cal = self.apply_onedblack(sm_cal)

		# Linearity-Gain correction:
		img = self.apply_linearity_gain(img)
		#sm_cal = self.apply_linearity_gain(sm_cal)

		# Undershoot correction:
		img = self.apply_undershoot(img)
		#sm_cal = self.apply_undershoot(sm_cal)

		# Smear corrections:
		#img = self.apply_dark_smear(img)
		#img = self.apply_smear_tim(img, sm_cal, smear=smear)

		# Flatfield correction:
		img = self.apply_flatfield(img)

		# Convert to electrons per second:
		img = self.to_electrons_per_second(img)

		return img

	#----------------------------------------------------------------------------------------------
	def calibrate_tpf(self, tpf):
		"""
		Calibrate Target Pixel File.
		"""

		if tpf[0].header['CAMERA'] != self.camera:
			raise ValueError("")
		if tpf[0].header['CCD'] != self.ccd:
			raise ValueError("")

		meta = {}
		meta['coadds'] = int(tpf['PIXELS'].header['NREADOUT'])
		meta['index_rows'] = slice(tpf[2].header['CRVAL2P'] - 1, tpf[2].header['CRVAL2P'] + tpf[2].header['NAXIS2'] - 1, 1)
		meta['index_columns'] = slice(tpf[2].header['CRVAL1P'] - 1, tpf[2].header['CRVAL1P'] + tpf[2].header['NAXIS1'] - 1, 1)
		aperture = np.asarray(tpf['APERTURE'].data, dtype='int32')
		meta['sector'] = int(tpf[0].header['SECTOR'])
		meta['cadence'] = 120
		meta['header'] = tpf['PIXELS'].header
		meta['aperture'] = aperture

		# Mask of pixels not to include in any kind of calculations:
		mask = (aperture & 1 == 0)

		# Build read-noise image:
		outputs = aperture & (32 + 64 + 128 + 256)
		read_noise = np.zeros_like(aperture, dtype='float64')
		for output in np.unique(outputs):
			indx = (outputs == output)
			output_name = {32: 'A', 64: 'B', 128: 'C', 256: 'D'}[output]
			read_noise[indx] = meta['coadds'] * self.readnoise[output_name].value

		#
		#smrow = get_tpf_col(img, coltype='smrow')

		# Loop through all target pixel images, calibrating them one at a time:
		tab = tpf['PIXELS'].data
		for k in tqdm.trange(len(tab)):

			# Create uncertainty image:
			shot_noise = np.sqrt(tab['RAW_CNTS'][k])
			total_noise = np.sqrt( read_noise**2 + shot_noise**2 )

			# Construct CalibImage image object:
			meta['cadenceno'] = tab['CADENCENO'][k]
			img = CalibImage(
				data=tab['RAW_CNTS'][k],
				uncertainty=StdDevUncertainty(total_noise),
				unit=u.electron, # u.ct? u.adu?
				mask=mask,
				meta=meta
			)

			# Calibrate the image:
			img_cal = self.calibrate(img.copy())

			plt.figure(figsize=(6,10))
			plt.subplot(321)
			plot_image(img.data, xlabel=None, ylabel=None, make_cbar=True, clabel='Raw counts (e/cadence)', title='RAW_CNTS')
			plt.subplot(323)
			plot_image(tpf['PIXELS'].data['FLUX'][k] + tpf['PIXELS'].data['FLUX_BKG'][k], xlabel=None, ylabel=None, make_cbar=True, title='SPOC Flux')
			plt.subplot(324)
			plot_image(img_cal.data, xlabel=None, ylabel=None, make_cbar=True, title='TASOC Flux')

			#plt.subplot(324)
			#plot_image(img.uncertainty.array, xlabel=None, ylabel=None, make_cbar=True, clabel='Raw counts (e/cadence)')
			plt.subplot(325)
			plot_image(tpf['PIXELS'].data['FLUX_ERR'][k], xlabel=None, ylabel=None, make_cbar=True, title='SPOC Error')
			plt.subplot(326)
			plot_image(img_cal.uncertainty.array, xlabel=None, ylabel=None, make_cbar=True, title='TASOC Error')

			# Store the resulting calibrated image in the FITS hdu:
			tpf['PIXELS'].data['FLUX'][k][:] = np.asarray(img_cal.data)
			tpf['PIXELS'].data['FLUX_ERR'][k][:] = img_cal.uncertainty.array

			break

		return tpf

	#----------------------------------------------------------------------------------------------
	def calibrate_ffi(self, ffi):
		"""
		Calibrate Full Frame Image.
		"""
		if isinstance(ffi, str):
			ffi = fits.open(ffi, mode='readonly')
		if ffi[0].header['CAMERA'] != self.camera:
			raise ValueError("")
		if ffi[0].header['CCD'] != self.ccd:
			raise ValueError("")

		meta = {}
		meta['coadds'] = int(ffi[1].header['NREADOUT'])
		meta['header'] = ffi[1].header

		meta['index_rows'] = slice(None, None, 1)
		meta['index_columns'] = slice(None, None, 1)
		meta['sector'] = int(ffi[0].header['SECTOR'])
		meta['cadence'] = 'ffi'

		# Create a large and aperture image:
		aperture = np.ones_like(ffi[1].data, dtype='int32')
		aperture[:, 44:556] |= 32 # CCD output A
		aperture[:, 556:1068] |= 64 # CCD output B
		aperture[:, 1068:1580] |= 128 # CCD output C
		aperture[:, 1580:2092] |= 256 # CCD output D

		# Leading columns (underclocks):
		aperture[:, 0:11] |= 32 # CCD output A
		aperture[:, 11:22] |= 64 # CCD output B
		aperture[:, 22:33] |= 128 # CCD output C
		aperture[:, 33:44] |= 256 # CCD output C

		# Trailing columns (overclocks):
		aperture[:, 2092:2103] |= 32 # CCD output A
		aperture[:, 2103:2114] |= 64 # CCD output B
		aperture[:, 2114:2125] |= 128 # CCD output C
		aperture[:, 2125:2136] |= 256 # CCD output C

		# Mask of pixels not to include in any kind of calculations:
		mask = (aperture & 1 == 0)

		# Build read-noise image:
		outputs = aperture & (32 + 64 + 128 + 256)
		read_noise = np.zeros_like(aperture, dtype='float64')
		for output in np.unique(outputs):
			indx = (outputs == output)
			output_name = {32: 'A', 64: 'B', 128: 'C', 256: 'D'}[output]
			read_noise[indx] = meta['coadds'] * self.readnoise[output_name].value

		shot_noise = np.sqrt(ffi[1].data)
		total_noise = np.sqrt( read_noise**2 + shot_noise**2 )

		# Construct CCDData image object:
		img = CalibImage(
			data=ffi[1].data,
			uncertainty=StdDevUncertainty(total_noise),
			unit=u.electron, # u.ct? u.adu?
			mask=mask,
			meta=meta,
			flags=aperture
		)

		img = img[:2048, 44:2092]

		# In FFIs the masked smear rows and virtual rows are part of the image itself:
		smear_rows = img[2058:2068, 44:2092]
		virtual_rows = img[2068:, 44:2092]
		trailing_virtual_cols = img[:2048, 2092:]

		plt.figure()
		plot_image(aperture)

		# Calibrate the image:
		img_cal = self.calibrate(img.copy())

		plt.figure()
		plt.plot(img_cal.meta['smear'])

		plt.figure(figsize=(15,8))

		ax = plt.subplot(121)
		plot_image(img.data, xlabel=None, ylabel=None, cbar=True, clabel='Raw counts (e/cadence)', title='RAW_CNTS')
		plt.subplot(122, sharex=ax, sharey=ax)
		plot_image(img_cal.data, xlabel=None, ylabel=None, cbar=True, clabel='Flux (e/s)', title='TASOC Flux')

		# Store the resulting calibrated image in the FITS hdu:
		ffi[1].data = np.asarray(img_cal.data)
		ffi[2].data = img_cal.uncertainty.array
		return ffi

	#----------------------------------------------------------------------------------------------
	def plot_flatfield(self):
		fig, ax = plt.subplots()
		plot_image(self.flatfield.data, ax=ax, scale='linear', cmap='seismic', vmin=0.95, vmax=1.05, cbar='right')

	#----------------------------------------------------------------------------------------------
	def plot_twodblack(self):
		fig, ax = plt.subplots()
		plot_image(self.twodblack.data, ax=ax, scale='linear', cbar='right')
