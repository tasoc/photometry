#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of SPICE Kernel module.
"""

import pytest
import os
import numpy as np
from scipy.interpolate import interp1d
import astropy.coordinates as coord
import astropy.units as u
from astropy.time import Time
from astropy.io import fits
import h5py
from tempfile import TemporaryDirectory
import conftest # noqa: F401
from photometry import AperturePhotometry
from photometry.spice import TESS_SPICE, InadequateSpiceError
from photometry.utilities import find_tpf_files, find_hdf5_files, add_proper_motion
from photometry.plots import plt, plots_interactive
from mpl_toolkits.mplot3d import Axes3D # noqa: F401

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('starid', [260795451, 267211065])
def test_timestamps(SHARED_INPUT_DIR, starid):
	with TemporaryDirectory() as OUTPUT_DIR:
		print("="*72)
		print("TIC %d" % starid)

		tpf_file = find_tpf_files(SHARED_INPUT_DIR, starid=starid)[0]
		with fits.open(tpf_file, mode='readonly', memmap=True) as hdu:
			time_tpf = np.asarray(hdu[1].data['TIME'])
			timecorr_tpf = np.asarray(hdu[1].data['TIMECORR']) * 86400 * 1000

		indx = np.isfinite(time_tpf) & np.isfinite(timecorr_tpf)
		intp_timecorr = interp1d(time_tpf[indx], timecorr_tpf[indx], kind='linear')

		with AperturePhotometry(starid, SHARED_INPUT_DIR, OUTPUT_DIR,
			plot=False, datasource='ffi', sector=1, camera=3, ccd=2) as pho:
			#pho.photometry()
			time_ffi = np.asarray(pho.lightcurve['time'])
			timecorr_ffi = np.asarray(pho.lightcurve['timecorr']) * 86400 * 1000

		# Print difference between
		print("Timestamp difference (milliseconds):")
		print( intp_timecorr(time_ffi) - timecorr_ffi )

		# Should be within 1 millisecond of the TPF files:
		np.testing.assert_allclose(timecorr_ffi, intp_timecorr(time_ffi), rtol=1e-3)

	print("="*72)

#--------------------------------------------------------------------------------------------------
def test_position_velocity():

	time_nocorr = np.array([1325.32351727, 1325.34435059, 1325.36518392, 1325.38601724])

	intv = Time([time_nocorr[0], time_nocorr[-1]], 2457000, format='jd', scale='utc')
	with TESS_SPICE(intv=intv, download=False) as knl:

		# We should be able to load and close (not unload!) without
		# affecting the results of the following:
		with TESS_SPICE(intv=intv):
			pass

		# We should fail with a timestamp that are outside the TESS SPICE coverage:
		# The timestamp used here is well before TESS was launched
		time_nocoverage = 1000 + 2457000
		with pytest.raises(InadequateSpiceError) as e:
			knl.position(time_nocoverage)
		assert str(e.value) == 'Inadequate SPICE kernels available'

		with pytest.raises(InadequateSpiceError) as e:
			knl.velocity(time_nocoverage)
		assert str(e.value) == 'Inadequate SPICE kernels available'

		with pytest.raises(InadequateSpiceError) as e:
			knl.position_velocity(time_nocoverage)
		assert str(e.value) == 'Inadequate SPICE kernels available'

		# Get the location of TESS as a function of time relative to Earth in kilometers:
		pos = knl.position(time_nocorr + 2457000)
		assert pos.shape == (len(time_nocorr), 3)

		# Get the location of TESS as a function of time relative to Earth in kilometers:
		vel = knl.velocity(time_nocorr + 2457000)
		assert vel.shape == (len(time_nocorr), 3)

		# Plot TESS orbits in 3D:
		time_inter = np.linspace(time_nocorr[0] - 3, time_nocorr[-1] + 3, 2000)
		pos_inter = knl.position(time_inter + 2457000)
		vel_inter = knl.velocity(time_inter + 2457000)

		fig1 = plt.figure()
		ax = fig1.add_subplot(111, projection='3d')
		ax.plot(pos_inter[:,0], pos_inter[:,1], pos_inter[:,2], 'r-')
		ax.scatter(pos[:,0], pos[:,1], pos[:,2], alpha=0.5)
		ax.scatter(0, 0, 0, alpha=0.5, c='b')

		fig2 = plt.figure()
		ax1 = fig2.add_subplot(211)
		ax1.plot(time_inter, np.linalg.norm(pos_inter, axis=1), 'r-')
		ax1.scatter(time_nocorr, np.linalg.norm(pos, axis=1), alpha=0.5)
		ax1.set_ylabel('Distance (km)')
		ax1.set_xticks([])

		ax2 = fig2.add_subplot(212)
		ax2.plot(time_inter, np.linalg.norm(vel_inter, axis=1), 'r-')
		ax2.scatter(time_nocorr, np.linalg.norm(vel, axis=1), alpha=0.5)
		ax2.set_ylabel('Velocity (km/s)')
		ax2.set_xlabel('Time')

		plt.tight_layout()

#--------------------------------------------------------------------------------------------------
def test_sclk2jd():

	print("="*72)

	star_coord = coord.SkyCoord(
		ra='04:52:6.92',
		dec='-70:43:52.4',
		unit=(u.hourangle, u.deg),
		frame='icrs',
		pm_ra_cosdec=1.13036*u.mas/u.yr,
		pm_dec=-11.1042*u.mas/u.yr,
		obstime=Time("J2000")
	)

	desired = 1468.416666534158

	intv = Time([desired-0.1, desired+0.1], 2457000, format='jd', scale='utc')
	with TESS_SPICE(intv=intv) as knl:

		jdtdb = knl.sclk2jd('1228946341.75')
		time = jdtdb - 2457000
		diff = (time - desired)*86400

		print("Converted time: %.16f" % time)
		print("Desired:        %.16f" % desired)
		print("Difference:     %.6f s" % diff )

		#np.testing.assert_allclose(diff, 0, atol=0.1)

		time, timecorr = knl.barycorr(jdtdb, star_coord)
		time -= 2457000
		diff = (time - desired)*86400

		print("Barycorr:       %.6f" % (timecorr*86400))
		print("Converted time: %.16f" % time)
		print("Difference:     %.6f s" % diff )

	print("="*72)

#--------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('starid', [260795451, 267211065])
def test_spice(SHARED_INPUT_DIR, starid):

	intv = Time([1325.30104564163, 1326.68855796131], 2457000, format='jd', scale='tdb')

	# Initialize our home-made TESS Kernel object:
	with TESS_SPICE(intv=intv, download=False) as knl:
		print("="*72)
		print("TIC %d" % starid)

		tpf_file = find_tpf_files(SHARED_INPUT_DIR, starid=starid)[0]
		with fits.open(tpf_file, mode='readonly', memmap=True) as hdu:
			time_tpf = np.asarray(hdu[1].data['TIME'])
			timecorr_tpf = np.asarray(hdu[1].data['TIMECORR'])
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
				pm_dec=hdu[0].header['PMDEC']*u.mas/u.yr,
				radial_velocity=0*u.km/u.s
			)

		# Remove bad timestamps from TPF file:
		indx = np.isfinite(time_tpf) & np.isfinite(timecorr_tpf)
		time_tpf = time_tpf[indx]
		timecorr_tpf = timecorr_tpf[indx]

		# Load the original timestamps from FFIs:
		hdf_file = find_hdf5_files(SHARED_INPUT_DIR, camera=camera, ccd=ccd)[0]
		with h5py.File(hdf_file, 'r') as hdf:
			ffi_time = np.asarray(hdf['time'])
			ffi_timecorr = np.asarray(hdf['timecorr'])

		f = interp1d(time_tpf-timecorr_tpf, timecorr_tpf, kind='linear')

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

		print(star_coord)

		# Use Greenwich as location instead of TESS (similar to what is done in Elenor):
		greenwich = coord.EarthLocation.of_site('greenwich')
		times_greenwich = Time(time_nocorr+2457000, format='jd', scale='utc', location=greenwich)
		timecorr_greenwich = times_greenwich.light_travel_time(star_coord, kind='barycentric', ephemeris='builtin').value
		#time_greenwich = time_nocorr + timecorr_greenwich

		# Calculate barycentric correction using our method:
		time_astropy, timecorr_astropy = knl.barycorr(time_nocorr + 2457000, star_coord)

		# Calculate barycentric correction using second method:
		timecorr_knl = knl.barycorr2(time_nocorr + 2457000, star_coord)
		print(timecorr_knl)

		# Plot the new barycentric time correction and the old one:
		fig1 = plt.figure()
		ax = fig1.add_subplot(111)
		ax.plot(time_tpf - timecorr_tpf, timecorr_tpf*86400, 'g.-', label='TPF timecorr')
		ax.plot(time_nocorr, ffi_timecorr*86400, 'b.-', label='FFI centre')
		ax.scatter(time_nocorr, timecorr_greenwich*86400, marker='x', label='Elenor timecorr')
		ax.scatter(time_nocorr, timecorr_astropy*86400, marker='x', c='r', label='Our timecorr')
		ax.set_xlabel('Uncorrected Time [TJD]')
		ax.set_ylabel('Barycentric Time Correction [s]')
		ax.set_title('TIC %d' % starid)
		ax.set_yscale('log')
		plt.legend()

		# Plot the new barycentric time correction and the old one:
		fig2 = plt.figure(figsize=(8,10))
		ax3 = fig2.add_subplot(111)
		ax3.axhline(1.0, color='k', ls='--', lw=1)
		ax3.plot(time_nocorr, np.abs(ffi_timecorr - f(time_nocorr))*86400, '.-', label='FFI centre')
		ax3.plot(time_nocorr, np.abs(timecorr_greenwich - f(time_nocorr))*86400*1000, '.-', label='Elenor')
		ax3.plot(time_nocorr, np.abs(timecorr_astropy - f(time_nocorr))*86400*1000, '.-', label='Astropy')
		ax3.plot(time_nocorr, np.abs(timecorr_knl - f(time_nocorr))*86400*1000, '.-', label='Kernel')
		ax3.set_ylabel('abs(TimeCorr - TPF TimeCorr) [ms]')
		ax3.set_xlabel('Uncorrected Time [TJD]')
		ax3.set_yscale('log')
		ax3.set_ylim(bottom=0.1)
		ax3.legend()
		ax3.grid(axis='y', which='both', color='0.7', ls=':', lw=0.5)
		ax3.set_title('TIC %d' % starid)
		plt.tight_layout()

		assert np.all(86400000*np.abs(timecorr_astropy - f(time_nocorr)) < 1), "AstroPy method: More than 1 ms away"
		assert np.all(86400000*np.abs(timecorr_knl - f(time_nocorr)) < 1), "HomeMade method: More than 1 ms away"

	print("="*72)

#--------------------------------------------------------------------------------------------------
@pytest.mark.skipif(os.environ.get('GITHUB_ACTIONS') == 'true' and os.environ.get('OS','').startswith('macos'),
	reason='Requires full list of SPICE kernels')
@pytest.mark.web
@pytest.mark.parametrize('starid', [260795451, 267211065])
def test_spice_with_interval(SHARED_INPUT_DIR, starid):

	tpf_file = find_tpf_files(SHARED_INPUT_DIR, starid=starid)[0]

	with fits.open(tpf_file, mode='readonly', memmap=True) as hdu:
		time_tpf = np.asarray(hdu[1].data['TIME'])
		timecorr_tpf = np.asarray(hdu[1].data['TIMECORR'])

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

	time = time_tpf - timecorr_tpf + 2457000

	with TESS_SPICE(download=True) as knl:
		num_kernels_full = len(knl.kernel_files)
		t1 = knl.barycorr(time, star_coord)
		p1 = knl.position_velocity(time)

	print(time)
	print([time[0], time[-1]])
	intv = Time([time[0], time[-1]], format='jd', scale='tdb')
	with TESS_SPICE(intv=intv, download=False) as knl:
		num_kernels_intv = len(knl.kernel_files)
		t2 = knl.barycorr(time, star_coord)
		p2 = knl.position_velocity(time)

	# There should be fewer kernels loaded when using an interval:
	assert num_kernels_full > num_kernels_intv

	# Using the interval should give exactly the same results:
	np.testing.assert_allclose(t2, t1)
	np.testing.assert_allclose(p2, p1)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	plots_interactive()
	pytest.main([__file__])
	plt.show()
