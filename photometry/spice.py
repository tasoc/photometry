#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of SPICE kernels with TESS to find barycentric time correction for FFIs.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u
import os
import numpy as np
import spiceypy
from spiceypy.utils.support_types import SpiceyError
import hashlib
from .utilities import download_file

class InadequateSpiceException(Exception):
	pass

class TESS_SPICE(object):
	"""
	SPICE Kernel object.

	Attributes:
		planetary_ephemeris (string): Planetary ephemeris that is loaded. Can be passed to
		METAKERNEL (string): Path to meta-kernel currently loaded.
	"""

	def __init__(self, kernels_folder=None):
		"""
		Parameters:
			kernels_folder (string): If not provided, the path stored in the environment
				variable ``TESSPHOT_SPICE_KERNELS`` is used, and if that is not set, the
				``data/spice`` directory is used.
		"""

		# If no kernel folder is given, used the one stored in env.var. or the default location:
		if kernels_folder is None:
			kernels_folder = os.environ.get('TESSPHOT_SPICE_KERNELS', os.path.join(os.path.dirname(__file__), 'data', 'spice'))

		# Create list of kernels that should be loaded:
		files = (
			'tess2018338154046-41240_naif0012.tls',
			'tess2018338154429-41241_de430.bsp',
			#'TESS_EPH_PRE_2YEAR_2018171_01.bsp',
			#'TESS_EPH_DEF_2018004_01.bsp',
			'TESS_EPH_DEF_2018080_01.bsp',
			#'TESS_EPH_DEF_2018108_01.bsp',
			'TESS_EPH_DEF_2018108_02.bsp',
			'TESS_EPH_DEF_2018115_01.bsp',
			'TESS_EPH_DEF_2018124_01.bsp',
			'TESS_EPH_DEF_2018133_01.bsp',
			'TESS_EPH_DEF_2018150_01.bsp',
			'TESS_EPH_DEF_2018183_01.bsp',
			'TESS_EPH_DEF_2018186_01.bsp',
			'TESS_EPH_DEF_2018190_01.bsp',
			'TESS_EPH_DEF_2018193_01.bsp',
			'TESS_EPH_DEF_2018197_01.bsp',
			'TESS_EPH_DEF_2018200_01.bsp',
			'TESS_EPH_DEF_2018204_01.bsp',
			'TESS_EPH_DEF_2018207_01.bsp',
			'TESS_EPH_DEF_2018211_01.bsp',
			'TESS_EPH_DEF_2018214_01.bsp',
			'TESS_EPH_DEF_2018218_01.bsp',
			'TESS_EPH_DEF_2018221_01.bsp',
			'TESS_EPH_DEF_2018225_01.bsp',
			'TESS_EPH_DEF_2018228_01.bsp',
			'TESS_EPH_DEF_2018232_01.bsp',
			'TESS_EPH_DEF_2018235_01.bsp',
			'TESS_EPH_DEF_2018239_01.bsp',
			'TESS_EPH_DEF_2018242_01.bsp',
			'TESS_EPH_DEF_2018246_01.bsp',
			'TESS_EPH_DEF_2018249_01.bsp',
			'TESS_EPH_DEF_2018253_01.bsp',
			'TESS_EPH_DEF_2018256_01.bsp',
			'TESS_EPH_DEF_2018260_01.bsp',
			'TESS_EPH_DEF_2018263_01.bsp',
			'TESS_EPH_DEF_2018268_01.bsp',
			'TESS_EPH_DEF_2018270_01.bsp',
			'TESS_EPH_DEF_2018274_01.bsp',
			'TESS_EPH_DEF_2018277_01.bsp',
			'TESS_EPH_DEF_2018282_01.bsp',
			'TESS_EPH_DEF_2018285_01.bsp',
			'TESS_EPH_DEF_2018288_01.bsp',
			'TESS_EPH_DEF_2018291_01.bsp',
			'TESS_EPH_DEF_2018295_01.bsp',
			'TESS_EPH_DEF_2018298_01.bsp',
			'TESS_EPH_DEF_2018302_01.bsp',
			'TESS_EPH_DEF_2018305_01.bsp',
			'TESS_EPH_DEF_2018309_01.bsp',
			'TESS_EPH_DEF_2018312_01.bsp',
			'TESS_EPH_DEF_2018316_01.bsp',
			'TESS_EPH_DEF_2018319_01.bsp',
		)

		# Make sure the kernel directory exists:
		if not os.path.exists(kernels_folder):
			os.makedirs(kernels_folder)

		# Automatically download kernels from MAST, if they don't already exist?
		urlbase = 'https://archive.stsci.edu/missions/tess/models/'
		for fname in files:
			fpath = os.path.join(kernels_folder, fname)
			if not os.path.exists(fpath):
				download_file(urlbase + fname, fpath)

		# Path where meta-kernel will be saved:
		fileshash = hashlib.md5(','.join(files).encode()).hexdigest()
		self.METAKERNEL = os.path.join(kernels_folder, 'metakernel-' + fileshash + '.txt')

		# Write meta-kernel to file:
		if not os.path.exists(self.METAKERNEL):
			with open(self.METAKERNEL, 'w') as fid:
				fid.write("KPL/MK\n")
				fid.write(r"\begindata" + "\n")
				fid.write("PATH_VALUES = ('" + os.path.abspath(kernels_folder) + "')\n")
				fid.write("PATH_SYMBOLS = ('KERNELS')\n")
				fid.write("KERNELS_TO_LOAD = (\n")
				fid.write( ",\n".join(["'$KERNELS/" + fname + "'" for fname in files]) )
				fid.write(")\n")
				fid.write(r"\begintext" + "\n")
				fid.write("End of MK file.\n")

		# Define TESS and load kernels:
		spiceypy.boddef('TESS', -95)
		spiceypy.furnsh(self.METAKERNEL)

		# Let's make sure astropy is using the de430 kernels as well:
		# Default is to use the same as is being used by SPOC (de430).
		# TODO: Would be nice to also use the local one
		#self.planetary_ephemeris = 'de430'
		#self.planetary_ephemeris = 'file://' + os.path.join(kernels_folder, 'tess2018338154429-41241_de430.bsp').replace('\\', '/')
		self.planetary_ephemeris = 'https://archive.stsci.edu/missions/tess/models/tess2018338154429-41241_de430.bsp'
		self._old_solar_system_ephemeris = coord.solar_system_ephemeris.get()
		coord.solar_system_ephemeris.set(self.planetary_ephemeris)

	def close(self):
		"""Close SPICE object."""
		spiceypy.unload(self.METAKERNEL)
		coord.solar_system_ephemeris.set(self._old_solar_system_ephemeris)

	def __enter__(self):
		return self

	def __exit__(self, *args, **kwargs):
		self.close()

	def position(self, jd, relative_to='EARTH'):
		"""
		Returns position of TESS for the given timestamps as geocentric XYZ-coordinates in kilometers.

		Parameters:
			jd (ndarray): Time in Julian Days where position of TESS should be calculated.
			relative_to (string, optional): Object for which to calculate position relative to. Default='EARTH'.

		Returns:
			ndarray: Position of TESS as geocentric XYZ-coordinates in kilometers.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		# Convert JD to Ephemeris Time:
		jd = np.atleast_1d(jd)
		times = spiceypy.str2et(['JD %.16f' % j for j in jd])

		# Get positions as a 2D array of (x,y,z) coordinates in km:
		try:
			positions, lt = spiceypy.spkpos('TESS', times, 'J2000', 'NONE', relative_to)
			positions = np.atleast_2d(positions)
		except SpiceyError as e:
			if 'SPICE(SPKINSUFFDATA)' in e.value:
				raise InadequateSpiceException("Inadequate SPICE kernels available")
			else:
				raise

		return positions

	def EarthLocation(self, jd):
		"""
		Returns positions as an EarthLocation object, which can be feed
		directly into ``astropy.time.Time.light_travel_time``.

		Parameters:
			jd (ndarray): Time in Julian Days where position of TESS should be calculated.

		Returns:
			´astropy.coordinates.EarthLocation´ object: EarthLocation object that can be passed directly into ``astropy.time.Time``.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		# Get positions as 2D array:
		positions = self.position(jd, relative_to='EARTH')

		# Transform into appropiate Geocentric frame:
		obstimes = Time(jd, format='jd', scale='utc')
		cartrep = coord.CartesianRepresentation(positions, xyz_axis=1, unit=u.km)
		gcrs = coord.GCRS(cartrep, obstime=obstimes)
		itrs = gcrs.transform_to(coord.ITRS(obstime=obstimes))

		# Create EarthLocation object
		return coord.EarthLocation.from_geocentric(*itrs.cartesian.xyz, unit=u.km)

	def position_velocity(self, jd, relative_to='EARTH'):
		"""
		Returns position and velocity of TESS for the given timestamps as geocentric XYZ-coordinates in kilometers.

		Parameters:
			jd (ndarray): Time in Julian Days where position of TESS should be calculated.
			relative_to (string, optional): Object for which to calculate position relative to. Default='EARTH'.

		Returns:
			ndarray: Position and velocity of TESS as geocentric XYZ-coordinates in kilometers.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		# Convert JD to Ephemeris Time:
		jd = np.atleast_1d(jd)
		times = spiceypy.str2et(['JD %.16f' % j for j in jd])

		# Get state of spacecraft (position and velocity):
		try:
			pos_vel, lt = spiceypy.spkezr('TESS', times, 'J2000', 'NONE', relative_to)
			pos_vel = np.asarray(pos_vel).T
		except SpiceyError as e:
			if 'SPICE(SPKINSUFFDATA)' in e.value:
				raise InadequateSpiceException("Inadequate SPICE kernels available")
			else:
				raise

		return pos_vel
