#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, with_statement, absolute_import
import sys
import os
import numpy as np
import astropy.coordinates as coord
import astropy.units as u
from astropy.time import Time
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from astropy.io import fits
import h5py
from mpl_toolkits.mplot3d import Axes3D
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from photometry.spice import TESS_SPICE
from photometry.utilities import find_tpf_files, find_hdf5_files

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')

def test_spice():

	# Initialize our home-made TESS Kernel object:
	with TESS_SPICE() as knl:
		for starid in (260795451, 267211065):

			tpf_file = find_tpf_files(INPUT_DIR, starid=starid)[0]
			with fits.open(tpf_file, mode='readonly', memmap=True) as hdu:
				time_tpf = hdu[1].data['TIME']
				timecorr_tpf = hdu[1].data['TIMECORR']
				camera = hdu[0].header['CAMERA']
				ccd = hdu[0].header['CCD']

				# Coordinates of the target as astropy SkyCoord object:
				star_coord = coord.SkyCoord(
					ra=hdu[0].header['RA_OBJ'],
					dec=hdu[0].header['DEC_OBJ'],
					unit=u.deg,
					frame='icrs',
					obstime=Time('J2000'),
					pm_ra_cosdec=hdu[0].header['PMRA']*u.mas/u.yr,
					pm_dec=hdu[0].header['PMDEC']*u.mas/u.yr
				)

			print(star_coord)
			print(camera, ccd)

			# Load the original timestamps from FFIs:
			hdf_file = find_hdf5_files(INPUT_DIR, camera=camera, ccd=ccd)[0]
			with h5py.File(hdf_file, 'r') as hdf:
				ffi_time = np.asarray(hdf['time'])
				ffi_timecorr = np.asarray(hdf['timecorr'])

			f = interp1d(time_tpf-timecorr_tpf, timecorr_tpf, kind='linear')

			# Change the timestamps bach to JD:
			time_nocorr = ffi_time - ffi_timecorr

			# Get the location of TESS as a function of time relative to Earth in kilometers:
			pos = knl.position(time_nocorr + 2457000)
			assert pos.shape == (len(time_nocorr), 3)

			# Plot TESS orbits in 3D:
			time_inter = np.linspace(time_nocorr[0] - 3, time_nocorr[-1] + 3, 2000)
			pos_inter = knl.position(time_inter + 2457000)

			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')
			ax.plot(pos_inter[:,0], pos_inter[:,1], pos_inter[:,2], 'r-')
			ax.scatter(pos[:,0], pos[:,1], pos[:,2], alpha=0.5)

			fig = plt.figure()
			ax = fig.add_subplot(111)
			ax.plot(time_inter, np.linalg.norm(pos_inter, axis=1), 'r-')
			ax.scatter(time_nocorr, np.linalg.norm(pos, axis=1), alpha=0.5)

			# Get the location of TESS as a function of time relative to Earth in kilometers:
			tess_position = knl.EarthLocation(time_nocorr + 2457000)

			# Calculate the light time travel correction for the stars coordinates:
			times = Time(time_nocorr, 2457000, format='jd', scale='utc', location=tess_position)
			timecorr = times.light_travel_time(star_coord, kind='barycentric', ephemeris=knl.planetary_ephemeris).value
			#time = time_nocorr + timecorr

			# Use Greenwich as location instead of TESS (similar to what is done in Elenor):
			greenwich = coord.EarthLocation.of_site('greenwich')
			times_greenwich = Time(time_nocorr, 2457000, format='jd', scale='utc', location=greenwich)
			timecorr_greenwich = times_greenwich.light_travel_time(star_coord, kind='barycentric', ephemeris=knl.planetary_ephemeris).value
			#time_greenwich = time_nocorr + timecorr_greenwich

			# Plot the new barycentric time correction and the old one:
			plt.figure()
			plt.scatter(time_nocorr, ffi_timecorr*86400, alpha=0.3, s=4, label='FFI timecorr')
			plt.scatter(time_tpf - timecorr_tpf, timecorr_tpf*86400, alpha=0.3, s=4, label='TPF timecorr')
			plt.scatter(time_nocorr, timecorr_greenwich*86400, alpha=0.3, s=4, label='Greenwich timecorr')
			plt.scatter(time_nocorr, timecorr*86400, alpha=0.3, s=4, label='New timecorr')
			plt.xlabel('Uncorrected Time (JD - 2457000)')
			plt.ylabel('Barycentric Time Correction (seconds)')
			plt.title('TIC %d' % starid)
			plt.legend()

			# Plot the new barycentric time correction and the old one:
			fig = plt.figure(figsize=(8,10))
			ax1 = plt.subplot(311)
			ax1.axhline(0, color='k', ls=':', lw=0.5)
			ax1.plot(time_nocorr, (ffi_timecorr - f(time_nocorr))*86400, '.')
			ax1.set_ylabel('FFI - TPF (seconds)')
			ax1.set_title('TIC %d' % starid)
			ax2 = plt.subplot(312)
			ax2.axhline(0, color='k', ls=':', lw=0.5)
			ax2.plot(time_nocorr, (timecorr_greenwich - f(time_nocorr))*86400, '.')
			ax2.set_ylabel('Greenwich - TPF (seconds)')
			ax3 = plt.subplot(313)
			ax3.axhline(0, color='k', ls=':', lw=0.5)
			ax3.plot(time_nocorr, (timecorr - f(time_nocorr))*86400, '.')
			ax3.set_ylabel('New - TPF (seconds)')
			ax3.set_xlabel('Uncorrected Time (TJD)')
			ax1.set_xticks([])
			ax2.set_xticks([])
			plt.tight_layout()

			assert np.all((timecorr - f(time_nocorr))*86400 < 0.5)

	plt.show()

if __name__ == '__main__':
	test_spice()
