#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of SPICE kernels with TESS to find barycentric time correction for FFIs.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import os.path
import logging
import numpy as np
from functools import lru_cache
import astropy.constants as const
import astropy.coordinates as coord
from astropy.time import Time
from astropy.table import Table
import astropy.units as u
from astropy.version import major as astropy_major_version
import spiceypy
import hashlib
from .utilities import download_parallel, to_tuple

#--------------------------------------------------------------------------------------------------
class InadequateSpiceError(Exception):
	pass

#--------------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_kernel_files_table():
	"""
	List of kernels that should be loaded:
	"""
	tab = Table.read(os.path.join(os.path.dirname(__file__), 'data', 'spice-kernels.ecsv'),
		format='ascii.ecsv')
	if not isinstance(tab['tmin'], Time):
		tab['tmin'] = Time(tab['tmin'], format='iso', scale='utc')
	if not isinstance(tab['tmax'], Time):
		tab['tmax'] = Time(tab['tmax'], format='iso', scale='utc')
	return tab

#--------------------------------------------------------------------------------------------------
@lru_cache(maxsize=10)
def _filter_kernels(intv):
	if len(intv) != 2:
		raise ValueError("Invalid interval")
	tmin, tmax = sorted(Time(intv, format='jd', scale='utc'))
	kernels = []
	for knl in load_kernel_files_table():
		no_overlap = (knl['tmax'] < tmin) or (knl['tmin'] > tmax)
		if knl['tmin'].mask or not no_overlap:
			kernels.append(knl['fname'])
	return kernels

#--------------------------------------------------------------------------------------------------
class TESS_SPICE(object):
	"""
	SPICE Kernel object.

	Attributes:
		kernel_files (tuple): List of kernel files to be loaded.
		planetary_ephemeris (string): Planetary ephemeris that is loaded. Can be passed to
			astropy.coord.solar_system_ephemeris.set.
		METAKERNEL (string): Path to meta-kernel currently loaded.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	#----------------------------------------------------------------------------------------------
	def __init__(self, intv=None, kernels_folder=None, download=True):
		"""
		Parameters:
			intv (Time):
			kernels_folder (string): If not provided, the path stored in the environment
				variable ``TESSPHOT_SPICE_KERNELS`` is used, and if that is not set, the
				``data/spice`` directory is used.
			download (bool): Allow download of missing SPICE kernels if needed. If ``False``,
				an error is raised in case of a missing kernel. Default=True.

		Raises:
			ValueError: If required SPICE kernels are not available.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		logger = logging.getLogger(__name__)

		# If no kernel folder is given, used the one stored in env.var. or the default location:
		if kernels_folder is None:
			kernels_folder = os.environ.get('TESSPHOT_SPICE_KERNELS', os.path.join(os.path.dirname(__file__), 'data', 'spice'))

		self.kernel_folder = kernels_folder

		# Make sure the kernel directory exists:
		kernels_folder = os.path.abspath(kernels_folder)
		os.makedirs(kernels_folder, exist_ok=True)

		if intv is None:
			self.kernel_files = list(load_kernel_files_table()['fname'])
		else:
			if isinstance(intv, Time):
				intv = intv.utc.jd
			self.kernel_files = _filter_kernels(to_tuple(intv))

		# Path where meta-kernel will be saved:
		hashkey = kernels_folder + ',' + ','.join(self.kernel_files)
		fileshash = hashlib.md5(hashkey.encode()).hexdigest()
		self.METAKERNEL = os.path.join(kernels_folder, 'metakernel-' + fileshash + '.txt')

		# Because SpiceyPy loads kernels into a global memory scope (BAAAAADDDD SpiceyPy!!!),
		# we first check if we have already loaded this into the global scope:
		# This is to attempt to avoid loading in the same kernels again and again when
		# running things in parallel.
		already_loaded = False
		for k in range(spiceypy.ktotal('META')):
			if os.path.abspath(spiceypy.kdata(k, 'META')[0]) == self.METAKERNEL:
				logger.debug("SPICE Meta-kernel already loaded.")
				already_loaded = True
				break

		# Load kernels if needed:
		if not already_loaded:
			# Automatically download kernels from TASOC, if they don't already exist?
			#urlbase = 'https://archive.stsci.edu/missions/tess/models/'
			urlbase = 'https://tasoc.dk/pipeline/spice/'
			downlist = []
			for fname in self.kernel_files:
				fpath = os.path.join(kernels_folder, fname)
				if not os.path.exists(fpath):
					downlist.append([urlbase + fname, fpath])

			if downlist:
				if download:
					download_parallel(downlist)
				else:
					raise ValueError("Some needed SPICE kernels are missing.")

			# Write meta-kernel to file:
			if not os.path.exists(self.METAKERNEL):
				with open(self.METAKERNEL, 'w') as fid:
					fid.write("KPL/MK\n")
					fid.write(r"\begindata" + "\n")
					fid.write("PATH_VALUES = ('" + kernels_folder + "')\n")
					fid.write("PATH_SYMBOLS = ('KERNELS')\n")
					fid.write("KERNELS_TO_LOAD = (\n")
					fid.write(",\n".join(["'$KERNELS/" + fname + "'" for fname in self.kernel_files]))
					fid.write(")\n")
					fid.write(r"\begintext" + "\n")
					fid.write("End of MK file.\n")

			# Define TESS object if it doesn't already exist:
			try:
				spiceypy.bodn2c('TESS')
			except spiceypy.utils.exceptions.NotFoundError:
				logger.debug("Defining TESS name in SPICE")
				spiceypy.boddef('TESS', -95)

			logger.debug("Loading SPICE Meta-kernel: %s", self.METAKERNEL)
			spiceypy.furnsh(self.METAKERNEL)

		# Let's make sure astropy is using the de430 kernels as well:
		# Default is to use the same as is being used by SPOC (de430).
		# If using astropy 4.0+, we can load the local one directly. Before this,
		# it needs to be downloaded and cached:
		# NOTE: https://github.com/astropy/astropy/pull/8767
		if astropy_major_version >= 4:
			self.planetary_ephemeris = os.path.join(kernels_folder, 'tess2018338154429-41241_de430.bsp')
		else:
			self.planetary_ephemeris = urlbase + 'tess2018338154429-41241_de430.bsp'
		self._old_solar_system_ephemeris = coord.solar_system_ephemeris.get()
		coord.solar_system_ephemeris.set(self.planetary_ephemeris)

	#----------------------------------------------------------------------------------------------
	def unload(self):
		"""Unload TESS SPICE kernels from memory."""
		try:
			spiceypy.unload(self.METAKERNEL)
		except spiceypy.exceptions.SpiceFILEOPENFAILED: # pragma: no cover
			pass

	#----------------------------------------------------------------------------------------------
	def unload_all(self):
		"""Unload all SPICE kernels from memory."""
		for fpath in self.loaded_kernels():
			try:
				spiceypy.unload(fpath)
			except spiceypy.exceptions.SpiceFILEOPENFAILED: # pragma: no cover
				pass

	#----------------------------------------------------------------------------------------------
	def close(self):
		"""Close SPICE object."""
		#self.unload() # Uhh, we are being naugthy here!
		coord.solar_system_ephemeris.set(self._old_solar_system_ephemeris)

	#----------------------------------------------------------------------------------------------
	def __enter__(self):
		return self

	#----------------------------------------------------------------------------------------------
	def __exit__(self, *args, **kwargs):
		self.close()

	#----------------------------------------------------------------------------------------------
	def loaded_kernels(self, kind='ALL'):
		"""
		Return load of currently loaded SPICE kernels.

		Parameters:
			kind (str, optional): Only return SPICE kernels of the given type. Default='ALL'.

		Returns:
			list: List of absolute paths to the SPICE kernels currently loaded into memory.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""
		knls = []
		for k in range(spiceypy.ktotal(kind)):
			knls.append(os.path.abspath(spiceypy.kdata(k, kind)[0]))
		return knls

	#----------------------------------------------------------------------------------------------
	def position(self, jd, of='TESS', relative_to='EARTH'):
		"""
		Returns position of TESS for the given timestamps as geocentric XYZ-coordinates
		in kilometers.

		Parameters:
			jd (ndarray): Time in Julian Days where position of TESS should be calculated.
			of (string, optional): Object for which to calculate position for. Default='TESS'.
			relative_to (string, optional): Object for which to calculate position relative to.
				Default='EARTH'.

		Returns:
			ndarray: Position of TESS as geocentric XYZ-coordinates in kilometers.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		# Convert JD to Ephemeris Time:
		jd = np.atleast_1d(jd)
		times = [spiceypy.unitim(j, 'JDTDB', 'ET') for j in jd]

		# Get positions as a 2D array of (x,y,z) coordinates in km:
		try:
			positions, _ = spiceypy.spkpos(of, times, 'J2000', 'NONE', relative_to)
			positions = np.atleast_2d(positions)
		except spiceypy.utils.exceptions.SpiceSPKINSUFFDATA:
			raise InadequateSpiceError("Inadequate SPICE kernels available")

		return positions

	#----------------------------------------------------------------------------------------------
	def EarthLocation(self, jd):
		"""
		Returns positions as an EarthLocation object, which can be feed
		directly into :func:`astropy.time.Time.light_travel_time`.

		Parameters:
			jd (ndarray): Time in Julian Days where position of TESS should be calculated.

		Returns:
			:class:`astropy.coordinates.EarthLocation`: EarthLocation object that can be passed
				directly into :py:class:`astropy.time.Time`.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		# Get positions as 2D array:
		positions = self.position(jd, relative_to='EARTH')

		# Transform into appropiate Geocentric frame:
		obstimes = Time(jd, format='jd', scale='tdb')
		cartrep = coord.CartesianRepresentation(positions, xyz_axis=1, unit=u.km)
		gcrs = coord.GCRS(cartrep, obstime=obstimes)
		itrs = gcrs.transform_to(coord.ITRS(obstime=obstimes))

		# Create EarthLocation object
		return coord.EarthLocation.from_geocentric(*itrs.cartesian.xyz, unit=u.km)

	#----------------------------------------------------------------------------------------------
	def position_velocity(self, jd, of='TESS', relative_to='EARTH'):
		"""
		Returns position and velocity of TESS for the given timestamps as geocentric
		XYZ-coordinates in kilometers.

		Parameters:
			jd (ndarray): Time in Julian Days where position of TESS should be calculated.
			of (string, optional): Object for which to calculate position for. Default='TESS'.
			relative_to (string, optional): Object for which to calculate position relative to.
				Default='EARTH'.

		Returns:
			ndarray: Position and velocity of TESS as geocentric XYZ-coordinates in kilometers.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		# Convert JD to Ephemeris Time:
		jd = np.atleast_1d(jd)
		times = [spiceypy.unitim(j, 'JDTDB', 'ET') for j in jd]

		# Get state of spacecraft (position and velocity):
		try:
			pos_vel, _ = spiceypy.spkezr(of, times, 'J2000', 'NONE', relative_to)
			pos_vel = np.asarray(pos_vel)
		except spiceypy.utils.exceptions.SpiceSPKINSUFFDATA:
			raise InadequateSpiceError("Inadequate SPICE kernels available")

		return pos_vel

	#----------------------------------------------------------------------------------------------
	def velocity(self, *args, **kwargs):
		"""
		Parameters:
			jd (ndarray): Time in Julian Days where position of TESS should be calculated.
			of (string, optional): Object for which to calculate position for. Default='TESS'.
			relative_to (string, optional): Object for which to calculate position relative to.
				Default='EARTH'.

		Returns:
			ndarray: Velocity of TESS as geocentric XYZ-coordinates in kilometers per seconds.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""
		return self.position_velocity(*args, **kwargs)[:, 3:]

	#----------------------------------------------------------------------------------------------
	def sclk2jd(self, sclk):
		"""
		Convert spacecraft time to TDB Julian Dates (JD).

		Parameters:
			sclk (ndarray): Timestamps in TESS Spacecraft Time.

		Returns:
			ndarray: Timestamps in TDB Julian Dates (TDB).
		"""

		sclk = np.atleast_1d(sclk)
		N = len(sclk)
		jd = np.empty(N, dtype='float64')
		for k in range(N):
			et = spiceypy.scs2e(-95, sclk[k])
			jd[k] = spiceypy.unitim(et, 'ET', 'JDTDB')

		return jd

	#----------------------------------------------------------------------------------------------
	def barycorr(self, tm, star_coord):
		"""
		Barycentric time correction.

		Using Astropys way of doing it.

		Parameters:
			tm (ndarray): Timestamps in TDB Julian Date (JD).
			star_coord (SkyCoord object): Coordinates of star.

		Returns:
			ndarray: Corrected timestamps in BJD.
			ndarray: Time corrections used to convert time into barycentric time in days.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		tess_position = self.EarthLocation(tm)
		times = Time(tm, format='jd', scale='tdb', location=tess_position)

		# TODO: Auto-advance the coordinates of the star to the given obstime, if possible
		# This is currently done in prepare/BasePhotometry to the reference-time for the sector
		#try:
		#	star_coord = star_coord.apply_space_motion(new_obstime=times)
		#except ValueError:
		#	pass
		#print(star_coord)

		# Calculate the light time travel correction for the stars coordinates:
		timecorr = times.light_travel_time(star_coord, kind='barycentric', ephemeris=self.planetary_ephemeris)

		# Calculate the corrected timestamps:
		time = times.tdb + timecorr

		return time.jd, timecorr.value

	#----------------------------------------------------------------------------------------------
	def barycorr2(self, times, star_coord):
		"""
		Barycentric time correction (experimental).

		Made from scratch and includes both Rømer, Einstein and Shapiro delays.

		Parameters:
			tm (ndarray): Timestamps in UTC Julian Date (JD).
			star_coord (SkyCoord object): Coordinates of star.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		# Constants in appropiate units:
		c = const.c.to('km/s').value
		GM_sun = const.GM_sun.to('km3/s2').value

		# Create unit-vector pointing to the star:
		star_coord = star_coord.transform_to('icrs')
		ra = star_coord.ra
		dec = star_coord.dec
		star_vector = np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])

		# Position of TESS spacecraft relative to solar system barycenter:
		tess_position = self.position(times, of='TESS', relative_to='0')

		# Assuming tess_position is in kilometers, this gives the correction in days
		delay_roemer = np.dot(tess_position, star_vector) / c

		# Shapiro delay:
		sun_pos = self.position(times, of='SUN', relative_to='TESS')
		sun_pos /= np.linalg.norm(sun_pos)
		costheta = np.array([np.dot(star_vector, sun_pos[k,:]) for k in range(len(times))])
		delay_shapiro = (2*GM_sun/c**3) * np.log(1 - costheta)
		#print(delay_shapiro)

		# Einstein delay:
		tess_position_geocenter = self.position(times, of='TESS', relative_to='EARTH')
		geocenter_velocity = self.velocity(times, of='EARTH', relative_to='0')
		delay_einstein = np.array([np.dot(tess_position_geocenter[k,:], geocenter_velocity[k,:]) for k in range(len(times))])
		delay_einstein /= c**2

		timecorr = ( delay_roemer + delay_shapiro + delay_einstein ) / 86400.0
		#print(timecorr)

		return timecorr

	#----------------------------------------------------------------------------------------------
	def time_coverage(self, of='TESS'):
		"""
		Return table of time-coverage of loaded kernels.

		Parameters:
			of (str, optional): Object to generate table for (default='TESS').

		Returns:
			:class:`astropy.table.Table`: Table containing one row per loaded kernel
				and the corresponding minimum and maximum time coverage.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		objid = spiceypy.bodn2c(of)

		rows = []
		for f in self.kernel_files:
			fpath = os.path.join(self.kernel_folder, f)

			try:
				cell = spiceypy.spkcov(fpath, objid)
			except spiceypy.exceptions.SpiceINVALIDARCHTYPE:
				continue

			ncell = spiceypy.wncard(cell)
			if ncell == 0:
				continue
			elif ncell > 1:
				raise RuntimeError("what?")

			tmin, tmax = spiceypy.wnfetd(cell, 0)
			tmin = Time(spiceypy.unitim(tmin, 'ET', 'JDTDB'), format='jd', scale='tdb')
			tmax = Time(spiceypy.unitim(tmax, 'ET', 'JDTDB'), format='jd', scale='tdb')

			rows.append([f, tmin, tmax])

		return Table(rows=rows, names=['fname', 'tmin', 'tmax'])
