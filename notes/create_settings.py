#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script used for defining sector reference times in data/sectors.json

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import os.path
import numpy as np
from scipy.signal import find_peaks
from astropy.time import Time
import sys
if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path:
	sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from photometry.spice import TESS_SPICE, load_kernel_files_table
from photometry.plots import plt, plots_interactive

if __name__ == '__main__':
	plots_interactive()

	# Anchor-point which defines the "zeropoint" of the reference times:
	zp_sector = 7
	zp_reftime = 2458497.374306

	# The maximum time where we have reliable SPICE kernels:
	spicetab = load_kernel_files_table()
	indx_def = np.asarray([f.startswith('TESS_EPH_DEF_') for f in spicetab['fname']], dtype='bool')
	time_end = max(spicetab[indx_def]['tmax'])
	print(f"Max time: {time_end}")

	# Create time axis:
	launch = Time('2018-04-18T18:51:00', format='isot', scale='utc')
	dt = 300/86400
	time = np.arange(launch.utc.jd + 20, time_end.utc.jd, dt)

	# Use SPICE kernels to get the distance between TESS and the Earth:
	radius_earth = 6378.1370 # km
	with TESS_SPICE(download=True) as knl:
		pos = knl.position(time, of='TESS', relative_to='EARTH')
		dist = np.linalg.norm(pos, axis=1) / radius_earth
		knl.unload()

	# Find peaks, meaning positions where TESS is the furthest from Earth, separated by
	# at least 10 days (orbit ~13.4 days) and at a hight of at least 40 Earth radii:
	peaks, _ = find_peaks(dist, height=40, distance=10/dt)

	# Find the peak which is the closest to the given "zeropoint" reference time:
	indx = np.argmin(np.abs(time[peaks] - zp_reftime))

	# Create figure of TESS orbit with reference times marked:
	fig, ax = plt.subplots()
	ax.plot(time - 2457000, dist)
	ax.scatter(time[peaks] - 2457000, dist[peaks])
	ax.axvline(zp_reftime - 2457000, c='r')
	ax.set_xlabel('Time (JD - 2457000)')
	ax.set_ylabel('Distance (Earth radii)')
	#ax.set_title('TESS orbit height')
	#ax.set_xlim(2000, 2760)

	# Go two orbit at a time forward and mark those peaks as the reference times for that sector:
	sector = zp_sector
	for i in range(indx, len(peaks), 2):
		if sector < 27:
			ffi_cadence = 1800
			reference_time = time[peaks][i]
		elif sector < 55:
			ffi_cadence = 600
			reference_time = time[peaks][i]
		else:
			# Now TESS is doing downlinks at apogee as well as perigee,
			# therefore we are setting the reference time 3 days before
			# the firt apogee.
			ffi_cadence = 200
			reference_time = time[peaks][i] - 3

		ax.axvline(reference_time - 2457000, ls='--', c='g')

		print(f'"{sector:d}": {{"sector": {sector:d}, "reference_time": {reference_time:.6f}, "ffi_cadence": {ffi_cadence:d}}},')
		sector += 1

	plt.show()
