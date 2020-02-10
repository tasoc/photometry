#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import os.path
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
import astropy.units as u
import xml.etree.ElementTree as ET
import logging
from bottleneck import replace, nanmedian
from .polynomial import polynomial
from ..utilities import download_file
#from ..plots import plt

#------------------------------------------------------------------------------
class PixelCalibrator(object):

	def __init__(self, camera=None, ccd=None):

		self.logger = logging.getLogger(__name__)
		self.logger.info("Starting calibration module")

		# General settings for TESS data:
		self.exposure_time = 1.96 * u.second # seconds
		self.frametransfer_time = 0.04 * u.second # seconds
		self.readout_time = 0.5 * u.second # seconds

		self._twodblack = None
		self._flatfield = None


	def __enter__(self, *args, **kwargs):
		return self

	def __exit__(self, *args, **kwargs):
		pass

	#--------------------------------------------------------------------------
	@property
	def twodblack(self):
		"""
		2D black image for current stamp.

		Returns:
			numpy.array: 2D array with image of 2D black for current stamp.
		"""
		if self._twodblack is None:
			# FITS file containing flat-field model:
			blackfile = os.path.join(os.path.dirname(__file__), '..', 'data', 'twodblack',
			   'tess2018324-{camera:d}-{ccd:d}-2dblack.fits.gz'.format(
					camera=self.camera,
					ccd=self.ccd
				))

			if not os.path.exists(blackfile):
				url = 'https://tasoc.dk/pipeline/twodblack/' + os.path.basename(blackfile)
				download_file(url, blackfile)

			with fits.open(blackfile, mode='readonly', memmap=True) as hdu:
				self._twodblack = CCDData(
					data=np.asarray(hdu[1].data,
					uncertainty=np.asarray(hdu[2].data),
					unit=u.adu # u.ct?
				)

		return self._twodblack

	#--------------------------------------------------------------------------
	@property
	def flatfield(self):
		"""
		Flatfield for current stamp.

		Returns:
			numpy.array: 2D array with image of flatfield for current stamp.
		"""
		if self._flatfield is None:
			# FITS file containing flat-field model:
			flatfile = os.path.join(os.path.dirname(__file__), '..', 'data', 'flatfield',
			   'tess2018323-{camera:d}-{ccd:d}-flat.fits.gz'.format(
					camera=self.camera,
					ccd=self.ccd
				))

			if not os.path.exists(flatfile):
				url = 'https://tasoc.dk/pipeline/flatfield/' + os.path.basename(flatfile)
				download_file(url, flatfile)

			with fits.open(flatfile, mode='readonly', memmap=True) as hdu:
				self._flatfield = CCDData(
					data=np.asarray(hdu[1].data),
					uncertainty=np.asarray(hdu[2].data),
					unit=u.dimensionless_unscaled
				)

		return self._flatfield

	#--------------------------------------------------------------------------
	@property
	def virtual_rows(self):
		"""
		Flatfield for current stamp.

		Returns:
			numpy.array: 2D array with image of flatfield for current stamp.
		"""
		if self._virtual_rows is None:
			if self.datasource == 'ffi':

				self.hdf['virtual_rows/%04d' % k][:, ic1:ic2]

			else:
				outputs = self.aperture & (32+64+128+256)

				vrowsfile = os.path.join('data', 'flatfield',
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

	#--------------------------------------------------------------------------
	def linearity_gain(self, img):
		"""CCD Linearity/gain correction."""

		self.logger.info("Doing gain/linearity correction...")

		# Load the gain- and linearity models from XML files:
		self.logger.info("Loading gain and linearity models...")
		gain_model = ET.parse('data/gain.xml').getroot()
		linearity_model = ET.parse('data/linearity.xml').getroot()

		#
		for output in np.unique(img.outputs):

			gain = gain_model.find("./channel[@cameraNumber='{camera:d}'][@ccdNumber='{ccd:d}'][@ccdOutput='{output}']".format(
				camera=self.camera,
				ccd=self.ccd,
				output=output
			)).get('gainElectronsPerDN')
			gain = float(gain) # * u.photon/u.adu

			linpoly = linearity_model.find("./channel[@cameraNumber='{camera:d}'][@ccdNumber='{ccd:d}'][@ccdOutput='{output}']/linearityPoly".format(
				camera=self.camera,
				ccd=self.ccd,
				output=output
			))
			linpoly = polynomial(linpoly)

			# Evaluate the polynomial and multiply the image values by it:
			DN0 = img[img.outputs == output] / img.coadds
			img[img.outputs == output] = DN0 * linpoly(DN0)
			img[img.outputs == output] *= gain * img.coadds

		return img

	#--------------------------------------------------------------------------
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
	def smear(self, img):
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

	#--------------------------------------------------------------------------
	def apply_twodblack(self, img):
		"""2D black-level correction.

		TODO:
			 - Correct collateral pixels as well.
		"""
		self.logger.info("Doing 2D black correction...")
		img -= self.twodblack[img.index_rows, img.index_columns]
		return img

	#--------------------------------------------------------------------------
	def apply_flatfield(self, img):
		"""CCD flat-field correction."""
		self.logger.info("Doing flatfield correction...")
		img /= self.flatfield[img.index_rows, img.index_columns]
		return img

	#--------------------------------------------------------------------------
	def to_counts_per_second(self, img):
		img /= self.exposure_time * img.coadds
		return img

	#--------------------------------------------------------------------------
	def calibrate(self, img):
		"""Perform all calibration steps in sequence."""
		
		img = self.apply_twodblack(img)
		img = self.linearity_gain(img)
		img = self.smear(img)
		img = self.apply_flatfield(img)
		img = self.to_counts_per_second(img)
		
		return img

	#--------------------------------------------------------------------------
	def calibrate_tpf(self, tpf):
	
		for k in range(len(tpf['PIXELS'].data['RAW_CNTS'])):
			img = CCDData(
				data=tpf['PIXELS'].data['RAW_CNTS'][k]
				uncertainty=np.sqrt(tpf['PIXELS'].data['RAW_CNTS'][k]),
				unit=u.adu # u.ct?
			)
			
			img_cal = self.calibrate(img)
		
			tpf['PIXELS']['FLUX'][k] = img_cal.data
			tpf['PIXELS']['FLUX_ERR'][k] = img_cal.uncertainty
	
