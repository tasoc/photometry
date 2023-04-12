#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import logging
import configparser
import json
import os.path
import glob
import re
import itertools
import warnings
from functools import lru_cache
from collections import defaultdict
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.wcs import WCS, FITSFixedWarning
from .utilities import to_tuple

#--------------------------------------------------------------------------------------------------
class FFIImage(CCDData):
	def __init__(self, path):
		self.is_tess = False
		self.smear = None
		self.vsmear = None

		uncert = None
		w = None
		hdr = {}
		if isinstance(path, np.ndarray):
			data = path
		elif isinstance(path, str):
			with fits.open(path, mode='readonly') as hdu:
				hdr = hdu[0].header

				with warnings.catch_warnings():
					warnings.filterwarnings('ignore', category=FITSFixedWarning)
					w = WCS(header=hdu[1].header, relax=True)

				# Check if this is real TESS data:
				# Could proberly be done more elegant, but if it works, it works...
				if hdr.get('TELESCOP') == 'TESS' and hdu[1].header.get('NAXIS1') == 2136 and hdu[1].header.get('NAXIS2') == 2078:
					data = np.asarray(hdu[1].data[0:2048, 44:2092], dtype='float32')
					uncert = np.asarray(hdu[2].data[0:2048, 44:2092], dtype='float32')
					self.is_tess = True

					hdr = dict(hdu[0].header)
					hdr.update(dict(hdu[1].header))

					# This header is not added before sector 6, so in that case
					# we are doing a simple scaling of the timestamps.
					if 'FFIINDEX' not in hdr and hdr['EXPOSURE']*86400 > 1000:
						# The following numbers comes from unofficial communication
						# with Doug Caldwell and Roland Vanderspek:
						# The timestamp in TJD and the corresponding cadenceno:
						time = 0.5*(hdr['TSTART'] + hdr['TSTOP'])
						timecorr = hdr.get('BARYCORR', 0)
						first_time = 0.5*(1325.317007851970 + 1325.337841177751) - 3.9072474e-03
						first_cadenceno = 4697
						timedelt = 1800/86400
						# Extracpolate the cadenceno as a simple linear relation:
						offset = first_cadenceno - first_time/timedelt
						hdr['FFIINDEX'] = np.round((time - timecorr)/timedelt + offset)

					self.smear = CCDData(
						data=np.asarray(hdu[1].data[2058:2068, 44:2092], dtype='float32'),
						uncertainty=StdDevUncertainty(np.asarray(hdu[2].data[2058:2068, 44:2092], dtype='float32'), copy=False),
						unit='electron/s'
					)
					self.vsmear = CCDData(
						data=np.asarray(hdu[1].data[2068:, 44:2092], dtype='float32'),
						uncertainty=StdDevUncertainty(np.asarray(hdu[2].data[2068:, 44:2092], dtype='float32'), copy=False),
						unit='electron/s'
					)

				else:
					data = np.asarray(hdu[0].data, dtype='float32')
					uncert = np.asarray(hdu[1].data, dtype='float32')
		else:
			raise ValueError("Input image must be either 2D ndarray or path to file.")

		super().__init__(
			data=data,
			uncertainty=StdDevUncertainty(uncert, copy=False),
			mask=~np.isfinite(data),
			wcs=w,
			meta=hdr,
			unit='electron/s'
		)

#--------------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_settings():
	"""
	Load settings.

	Returns:
		:class:`configparser.ConfigParser`:
	"""

	settings = configparser.ConfigParser()
	settings.read(os.path.join(os.path.dirname(__file__), 'data', 'settings.ini'))
	return settings

#--------------------------------------------------------------------------------------------------
@lru_cache(maxsize=10)
def load_sector_settings(sector=None):

	with open(os.path.join(os.path.dirname(__file__), 'data', 'sectors.json'), 'r') as fid:
		settings = json.load(fid)

	if sector is not None:
		return settings['sectors'][str(sector)]

	return settings

#--------------------------------------------------------------------------------------------------
@lru_cache(maxsize=32)
def find_ffi_files(rootdir, sector=None, camera=None, ccd=None):
	"""
	Search directory recursively for TESS FFI images in FITS format.

	The function is cached, meaning the first time it is run on a particular ``rootdir``
	the list of files in that directory will be read and cached to memory and used in
	subsequent calls to the function. This means that any changes to files on disk after
	the first call of the function will not be picked up in subsequent calls to this function.

	Parameters:
		rootdir (str): Directory to search recursively for TESS FFI images.
		sector (int or None, optional): Only return files from the given sector.
			If ``None``, files from all sectors are returned.
		camera (int or None, optional): Only return files from the given camera number (1-4).
			If ``None``, files from all cameras are returned.
		ccd (int or None, optional): Only return files from the given CCD number (1-4).
			If ``None``, files from all CCDs are returned.

	Returns:
		list: List of full paths to FFI FITS files found in directory. The list will
			be sorted accoridng to the filename of the files, i.e. primarily by time.
	"""

	logger = logging.getLogger(__name__)

	# Create the filename pattern to search for:
	sector_str = r'\d{4}' if sector is None else f'{sector:04d}'
	camera = r'\d' if camera is None else str(camera)
	ccd = r'\d' if ccd is None else str(ccd)
	filename_pattern = r'^tess\d+-s(?P<sector>' + sector_str + ')-(?P<camera>' + camera + r')-(?P<ccd>' + ccd + r')-\d{4}-[xsab]_ffic\.fits(\.gz)?$'
	logger.debug("Searching for FFIs in '%s' using pattern '%s'", rootdir, filename_pattern)
	regexp = re.compile(filename_pattern)

	# Do a recursive search in the directory, finding all files that match the pattern:
	matches = []
	for root, dirnames, filenames in os.walk(rootdir, followlinks=True):
		for filename in filenames:
			if regexp.match(filename):
				matches.append(os.path.join(root, filename))

	# Sort the list of files by thir filename:
	matches.sort(key=lambda x: os.path.basename(x))

	return matches

#--------------------------------------------------------------------------------------------------
@lru_cache(maxsize=10)
def _find_tpf_files(rootdir, sector=None, cadence=None):

	logger = logging.getLogger(__name__)

	# Create the filename pattern to search for:
	sector_str = r'\d{4}' if sector is None else f'{sector:04d}'
	suffix = {None: '(fast-)?tp', 120: 'tp', 20: 'fast-tp'}[cadence]
	re_pattern = r'^tess\d+-s(?P<sector>' + sector_str + r')-(?P<starid>\d+)-\d{4}-[xsab]_' + suffix + r'\.fits(\.gz)?$'
	regexps = [re.compile(re_pattern)]
	logger.debug("Searching for TPFs in '%s' using pattern '%s'", rootdir, re_pattern)

	# Pattern used for TESS Alert data:
	if cadence is None or cadence == 120:
		sector_str = r'\d{2}' if sector is None else f'{sector:02d}'
		re_pattern2 = r'^hlsp_tess-data-alerts_tess_phot_(?P<starid>\d+)-s(?P<sector>' + sector_str + r')_tess_v\d+_tp\.fits(\.gz)?$'
		regexps.append(re.compile(re_pattern2))
		logger.debug("Searching for TPFs in '%s' using pattern '%s'", rootdir, re_pattern2)

	# Do a recursive search in the directory, finding all files that match the pattern:
	filedict = defaultdict(list)
	for root, dirnames, filenames in os.walk(rootdir, followlinks=True):
		for filename in filenames:
			for regex in regexps:
				m = regex.match(filename)
				if m:
					starid = int(m.group('starid'))
					filedict[starid].append(os.path.join(root, filename))
					break

	# Ensure that each list is sorted by itself. We do this once here
	# so we don't have to do it each time a specific starid is requested:
	for key in filedict.keys():
		filedict[key].sort(key=lambda x: os.path.basename(x))

	return filedict

#--------------------------------------------------------------------------------------------------
def find_tpf_files(rootdir, starid=None, sector=None, camera=None, ccd=None, cadence=None,
	findmax=None):
	"""
	Search directory recursively for TESS Target Pixel Files.

	The function is cached, meaning the first time it is run on a particular ``rootdir``
	the list of files in that directory will be read and cached to memory and used in
	subsequent calls to the function. This means that any changes to files on disk after
	the first call of the function will not be picked up in subsequent calls to this function.

	Parameters:
		rootdir (str): Directory to search recursively for TESS TPF files.
		starid (int, optional): Only return files from the given TIC number.
			If ``None``, files from all TIC numbers are returned.
		sector (int, optional): Only return files from the given sector.
			If ``None``, files from all sectors are returned.
		camera (int or None, optional): Only return files from the given camera number (1-4).
			If ``None``, files from all cameras are returned.
		ccd (int, optional): Only return files from the given CCD number (1-4).
			If ``None``, files from all CCDs are returned.
		cadence (int, optional): Only return files from the given cadence (20 or 120).
			If ``None``, files from all cadences are returned.
		findmax (int, optional): Maximum number of files to return.
			If ``None``, return all files.

	Note:
		Filtering on camera and/or ccd will cause the program to read the headers
		of the files in order to determine the camera and ccd from which they came.
		This can significantly slow down the query.

	Returns:
		list: List of full paths to TPF FITS files found in directory. The list will
			be sorted according to the filename of the files, i.e. primarily by time.
	"""

	if cadence is not None and cadence not in (120, 20):
		raise ValueError("Invalid cadence. Must be either 20 or 120.")

	# Call cached function which searches for files on disk:
	filedict = _find_tpf_files(rootdir, sector=sector, cadence=cadence)

	if starid is not None:
		files = filedict.get(starid, [])
	else:
		# If we are not searching for a particilar starid,
		# simply flatten the dict to a list of all found files
		# and sort the list of files by thir filename:
		files = list(itertools.chain(*filedict.values()))
		files.sort(key=lambda x: os.path.basename(x))

	# Expensive check which involve opening the files and reading headers:
	# We are only removing elements, and preserving the ordering, so there
	# is no need for re-sorting the list afterwards.
	if camera is not None or ccd is not None:
		matches = []
		for fpath in files:
			if camera is not None and fits.getval(fpath, 'CAMERA', ext=0) != camera:
				continue
			if ccd is not None and fits.getval(fpath, 'CCD', ext=0) != ccd:
				continue

			# Add the file to the list, but stop if we have already
			# reached the number of files we need to find:
			matches.append(fpath)
			if findmax is not None and len(matches) >= findmax:
				break

		files = matches

	# Just to ensure that we are not returning more than we should:
	if findmax is not None and len(files) > findmax:
		files = files[:findmax]

	return files

#--------------------------------------------------------------------------------------------------
@lru_cache(maxsize=32)
def find_hdf5_files(rootdir, sector=None, camera=None, ccd=None):
	"""
	Search the input directory for HDF5 files matching constraints.

	Parameters:
		rootdir (str): Directory to search for HDF5 files.
		sector (int, list or None, optional): Only return files from the given sectors.
			If ``None``, files from all TIC numbers are returned.
		camera (int, list or None, optional): Only return files from the given camera.
			If ``None``, files from all cameras are returned.
		ccd (int, list or None, optional): Only return files from the given ccd.
			If ``None``, files from all ccds are returned.

	Returns:
		list: List of paths to HDF5 files matching constraints.
	"""

	sector = to_tuple(sector, (None,))
	camera = to_tuple(camera, (1,2,3,4))
	ccd = to_tuple(ccd, (1,2,3,4))

	filelst = []
	for sector, camera, ccd in itertools.product(sector, camera, ccd):
		sector_str = '???' if sector is None else f'{sector:03d}'
		filelst += glob.glob(os.path.join(rootdir, f'sector{sector_str:s}_camera{camera:d}_ccd{ccd:d}.hdf5'))

	return filelst

#--------------------------------------------------------------------------------------------------
@lru_cache(maxsize=32)
def find_catalog_files(rootdir, sector=None, camera=None, ccd=None):
	"""
	Search the input directory for CATALOG (sqlite) files matching constraints.

	Parameters:
		rootdir (str): Directory to search for CATALOG files.
		sector (int, list or None, optional): Only return files from the given sectors.
			If ``None``, files from all TIC numbers are returned.
		camera (int, list or None, optional): Only return files from the given camera.
			If ``None``, files from all cameras are returned.
		ccd (int, list or None, optional): Only return files from the given ccd.
			If ``None``, files from all ccds are returned.

	Returns:
		list: List of paths to CATALOG files matching constraints.
	"""

	sector = to_tuple(sector, (None,))
	camera = to_tuple(camera, (1,2,3,4))
	ccd = to_tuple(ccd, (1,2,3,4))

	filelst = []
	for sector, camera, ccd in itertools.product(sector, camera, ccd):
		sector_str = '???' if sector is None else f'{sector:03d}'
		filelst += glob.glob(os.path.join(rootdir, f'catalog_sector{sector_str:s}_camera{camera:d}_ccd{ccd:d}.sqlite'))

	return filelst
