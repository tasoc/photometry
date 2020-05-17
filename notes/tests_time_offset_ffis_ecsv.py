#!/bin/env python
# -*- coding: utf-8 -*-
"""
Create input data for tests of time offset of FFIs.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import sys
import os.path
import numpy as np
from astropy.io import fits
from astropy.table import Table
import itertools
import gzip
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from photometry import utilities

#--------------------------------------------------------------------------------------------------
def s20_lc():

	with fits.open('/aadc/tasoc/archive/S20_DR27/spoc_lightcurves/tess2019357164649-s0020-0000000004164018-0165-s_lc.fits.gz') as hdu:
		hdr1 = hdu[0].header
		time1 = np.atleast_2d(hdu[1].data['TIME']).T

	with fits.open('/aadc/tasoc/archive/S20_DR27v2/spoc_lightcurves/tess2019357164649-s0020-0000000004164018-0165-s_lc.fits.gz') as hdu:
		time2 = np.atleast_2d(hdu[1].data['TIME']).T

	A = np.concatenate((time1, time2), axis=1)
	tab = Table(data=A, names=('time1', 'time2'), dtype=('float64', 'float64'))
	tab.meta = dict(hdr1)

	tab.write('time_offset_s20.ecsv', delimiter=',', format='ascii.ecsv')
	print(tab)

	with open('time_offset_s20.ecsv', 'rb') as src:
		with gzip.open('time_offset_s20.ecsv.gz', 'wb') as dst:
			dst.writelines(src)
	os.remove('time_offset_s20.ecsv')

#--------------------------------------------------------------------------------------------------
def s21_lc():

	with fits.open('/aadc/tasoc/archive/S21_DR29/spoc_lightcurves/tess2020020091053-s0021-0000000001096672-0167-s_lc.fits.gz') as hdu:
		hdr1 = hdu[0].header
		time1 = np.atleast_2d(hdu[1].data['TIME']).T

	with fits.open('/aadc/tasoc/archive/S21_DR29v2/spoc_lightcurves/tess2020020091053-s0021-0000000001096672-0167-s_lc.fits.gz') as hdu:
		time2 = np.atleast_2d(hdu[1].data['TIME']).T

	A = np.concatenate((time1, time2), axis=1)
	tab = Table(data=A, names=('time1', 'time2'), dtype=('float64', 'float64'))
	tab.meta = dict(hdr1)

	tab.write('time_offset_s21.ecsv', delimiter=',', format='ascii.ecsv')
	print(tab)

	with open('time_offset_s21.ecsv', 'rb') as src:
		with gzip.open('time_offset_s21.ecsv.gz', 'wb') as dst:
			dst.writelines(src)
	os.remove('time_offset_s21.ecsv')

#--------------------------------------------------------------------------------------------------
def ffis():

	np.random.seed(42)

	folders = [
		(20, '/aadc/tasoc/archive/S20_DR27/ffi/', '/aadc/tasoc/archive/S20_DR27v2/ffi/'),
		(21, '/aadc/tasoc/archive/S21_DR29/ffi/', '/aadc/tasoc/archive/S21_DR29v2/ffi/'),
	]
	print(folders)

	ffis = []
	for (sector, dpath, dpath_corr), camera, ccd in itertools.product(folders, (1,2,3,4), (1,2,3,4)):
		# Find all the files from this CCD:
		files = utilities.find_ffi_files(dpath, sector=sector, camera=camera, ccd=ccd)

		# Pick 2 random files from this CCD:
		pick = np.random.choice(files, 2, replace=False)

		for fpath in pick:
			fpath_corrected = os.path.join(dpath_corr, os.path.relpath(fpath, dpath))

			with fits.open(fpath_corrected, mode='readonly', memmap=True) as hdu:
				time2_start = hdu[1].header['TSTART'] - hdu[1].header['BARYCORR']
				time2_stop = hdu[1].header['TSTOP'] - hdu[1].header['BARYCORR']

			with fits.open(fpath, mode='readonly', memmap=True) as hdu:
				time1_start = hdu[1].header['TSTART'] - hdu[1].header['BARYCORR']
				time1_stop = hdu[1].header['TSTOP'] - hdu[1].header['BARYCORR']

				ffis.append(dict(
					sector=sector,
					data_rel=hdu[0].header['DATA_REL'],
					procver=hdu[0].header['PROCVER'],
					ffiindex=hdu[0].header['FFIINDEX'],
					camera=hdu[1].header['CAMERA'],
					ccd=hdu[1].header['CCD'],
					time_start=time1_start,
					time_stop=time1_stop,
					time_start_corrected=time2_start,
					time_mid_corrected=0.5*(time2_start + time2_stop),
					time_stop_corrected=time2_stop
				))

	# Create Astropy Table and save it as ECSV file:
	tab = Table(rows=ffis)
	tab.write('ffis.ecsv', format='ascii.ecsv', delimiter=',', overwrite=True)

	print(tab)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	s20_lc()
	s21_lc()
	ffis()
