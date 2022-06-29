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
from photometry.spice import TESS_SPICE
from photometry.plots import plt, plots_interactive

if __name__ == '__main__':
	plots_interactive()

	# Anchor-point which defines the "zeropoint" of the reference times:
	zp_sector = 7
	zp_reftime = 2458497.374306

	# The maximum sector to go up to (i.e. we dont have reliable SPICE kernels after this):
	max_sector = 55

	# Create time axis:
	launch = Time('2018-04-18T18:51:00', format='isot', scale='utc')
	time_end = Time.now()
	dt = 900/86400
	time = np.arange(launch.jd + 20, time_end.jd, dt)

	# Use SPICE kernels to get the distance between TESS and the Earth:
	radius_earth = 6378.1370 # km
	with TESS_SPICE() as knl:
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
	ax.plot(time, dist)
	ax.scatter(time[peaks], dist[peaks])
	ax.axvline(zp_reftime, c='r')
	ax.set_xlabel('Time (JD)')
	ax.set_ylabel('Distance (Earth radii)')

	# Go two orbit at a time forward and mark those peaks as the reference times for that sector:
	sector = zp_sector
	for i in range(indx, len(peaks), 2):
		ax.axvline(time[peaks][i], ls='--', c='g')

		if sector < 27:
			ffi_cadence = 1800
		elif sector < 55:
			ffi_cadence = 600
		else:
			ffi_cadence = 200

		print("\"%d\": {\"sector\": %d, \"reference_time\": %.6f, \"ffi_cadence\": %d}," % (sector, sector, time[peaks][i], ffi_cadence))
		sector += 1
		if sector > max_sector:
			break

	plt.show()
