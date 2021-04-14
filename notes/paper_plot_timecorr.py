#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to generate plot for Photometry Paper.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
from astropy.time import Time
from astropy.io import fits
import astropy.coordinates as coord
import astropy.units as u
from astropy.table import Table
from scipy.interpolate import interp1d
import h5py
import sys
import os.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from photometry.plots import plt, plot_image, matplotlib
from photometry.spice import TESS_SPICE
from photometry.utilities import find_tpf_files, find_hdf5_files, add_proper_motion
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = '14'
matplotlib.rcParams['axes.titlesize'] = '18'
matplotlib.rcParams['axes.labelsize'] = '16'
plt.rc('text', usetex=True)

if __name__ == '__main__':
	plt.switch_backend('Qt5Agg')
	plt.close('all')

	INPUT_DIR = r'F:\tess_data\S06_DR08'

	fig4 = plt.figure()
	ax = fig4.add_subplot(111)
	ax.axhline(0, color='k', ls='--')

	for starid in (59545062, 25155664, 14003429):

		# Initialize our home-made TESS Kernel object:
		with TESS_SPICE() as knl:
			print("="*72)
			print("TIC %d" % starid)

			tpf_file = find_tpf_files(INPUT_DIR, starid=starid)[0]
			with fits.open(tpf_file, mode='readonly', memmap=True) as hdu:
				time_tpf = hdu[1].data['TIME']
				timecorr_tpf = hdu[1].data['TIMECORR']
				camera = hdu[0].header['CAMERA']
				ccd = hdu[0].header['CCD']
				print(camera, ccd)

				# Coordinates of the target as astropy SkyCoord object:
				star_coord = coord.SkyCoord(
					ra=hdu[0].header['RA_OBJ'],
					dec=hdu[0].header['DEC_OBJ'],
					unit=u.deg,
					frame='icrs',
					obstime=Time('J2000'),
					pm_ra_cosdec=hdu[0].header['PMRA']*u.mas/u.yr,
					pm_dec=hdu[0].header['PMDEC']*u.mas/u.yr,
					radial_velocity=0*u.km/u.s
				)

			# Load the original timestamps from FFIs:
			hdf_file = find_hdf5_files(INPUT_DIR, camera=camera, ccd=ccd)[0]
			with h5py.File(hdf_file, 'r') as hdf:
				ffi_time = np.asarray(hdf['time'])
				ffi_timecorr = np.asarray(hdf['timecorr'])

			f = interp1d(time_tpf-timecorr_tpf, timecorr_tpf, kind='linear')
			f2 = interp1d(time_tpf-timecorr_tpf, time_tpf, kind='linear')

			# Change the timestamps bach to JD:
			time_nocorr = ffi_time - ffi_timecorr

			times = Time(time_nocorr, 2457000, format='jd', scale='tdb')
			ras, decs = add_proper_motion(
				star_coord.ra.value,
				star_coord.dec.value,
				star_coord.pm_ra_cosdec.value,
				star_coord.pm_dec.value,
				times.jd
			)
			star_coord = coord.SkyCoord(
				ra=ras[0],
				dec=decs[0],
				unit=u.deg,
				frame='icrs'
			)

			#print(star_coord)

			# Use Greenwich as location instead of TESS (similar to what is done in Elenor):
			greenwich = coord.EarthLocation.of_site('greenwich')
			times_greenwich = Time(time_nocorr+2457000, format='jd', scale='utc', location=greenwich)
			timecorr_greenwich = times_greenwich.light_travel_time(star_coord, kind='barycentric', ephemeris='builtin').value
			time_greenwich = time_nocorr + timecorr_greenwich
			#print(time_greenwich)

			# Calculate barycentric correction using our method:
			time_astropy, timecorr_astropy = knl.barycorr(time_nocorr + 2457000, star_coord)
			time_astropy -= 2457000
			#print(time_astropy)

		# Plot the new barycentric time correction and the old one:
		ax.scatter(time_nocorr, 86400*(ffi_time - f2(time_nocorr)), marker='.', label='FFI centre', alpha=0.5)
		ax.scatter(time_nocorr, 86400*(time_greenwich - f2(time_nocorr)), marker='.', label='Elenor', alpha=0.5)
		ax.scatter(time_nocorr, 86400*(time_astropy - f2(time_nocorr)), marker='.', c='r', label='Ours', alpha=0.5)

		tab = Table([time_nocorr, f2(time_nocorr), ffi_time, time_greenwich, time_astropy],
			names=['time_nocorr', 'time_spoc', 'time_fficentre', 'time_greenwich', 'time_ours'])
		print(tab)
		tab.write('time_TIC%d.ecsv' % starid, format='ascii.ecsv', delimiter=',')


	ax.set_xlabel('Uncorrected Time [TJD]')
	ax.set_ylabel('Time - SPOC time [s]')
	#ax.set_title('TIC %d' % starid)
	#ax.set_yscale('log')
	plt.legend()


	print("="*72)
	plt.show(block=True)
