#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of SPICE kernels with TESS to find barycentric time correction for FFIs.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import os.path
import logging
import numpy as np
import astropy.constants as const
import astropy.coordinates as coord
from astropy.time import Time
import astropy.units as u
from astropy.version import major as astropy_major_version
import spiceypy
from spiceypy.utils.support_types import SpiceyError
import hashlib
from .utilities import download_parallel

class InadequateSpiceException(Exception):
	pass

class TESS_SPICE(object):
	"""
	SPICE Kernel object.

	Attributes:
		kernel_files (tuple): List of kernel files to be loaded.
		planetary_ephemeris (string): Planetary ephemeris that is loaded. Can be passed to
			astropy.coord.solar_system_ephemeris.set.
		METAKERNEL (string): Path to meta-kernel currently loaded.
	"""

	# List of kernels that should be loaded:
	kernel_files = (
		# Planetary ephemeris and TESS clock kernels:
		'tess2018338154046-41240_naif0012.tls',
		'tess2018338154429-41241_de430.bsp',
		'tess2019113195500-41374_sclk.tsc',

		# Predictive kernels of TESS's expected position:
		#'TESS_EPH_PRE_2YEAR_2018171_01.bsp',
		#'TESS_EPH_PRE_LONG_2018004_01.bsp', # No informaion on TESS
		#'TESS_EPH_PRE_LONG_2018080_01.bsp', # Only (really) providing info before launch
		'TESS_EPH_PRE_LONG_2018109_02.bsp',
		'TESS_EPH_PRE_LONG_2019045_01.bsp',
		'TESS_EPH_PRE_LONG_2019364_21.bsp',

		# Definite kernels of TESS's actual position:
		#'TESS_EPH_DEF_2018004_01.bsp', # Does not contain any information
		#'TESS_EPH_DEF_2018080_01.bsp', # Only contains information from before launch of TESS (??)
		#'TESS_EPH_DEF_2018108_01.bsp', # Surpassed by never version below
		'TESS_EPH_DEF_2018108_02.bsp',
		'TESS_EPH_DEF_2018115_01.bsp',
		'TESS_EPH_DEF_2018124_01.bsp',
		'TESS_EPH_DEF_2018133_01.bsp',
		'TESS_EPH_DEF_2018150_01.bsp',
		#'TESS_EPH_DEF_2018183_01.bsp', # Overlaps 100% with other kernels
		'TESS_EPH_DEF_2018186_01.bsp',
		#'TESS_EPH_DEF_2018190_01.bsp', # Overlaps 100% with other kernels
		'TESS_EPH_DEF_2018193_01.bsp',
		#'TESS_EPH_DEF_2018197_01.bsp', # Overlaps 100% with other kernels
		#'TESS_EPH_DEF_2018200_01.bsp', # Overlaps 100% with other kernels
		#'TESS_EPH_DEF_2018204_01.bsp', # Overlaps 100% with other kernels
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
		'TESS_EPH_DEF_2018323_01.bsp',
		'TESS_EPH_DEF_2018327_01.bsp',
		'TESS_EPH_DEF_2018330_01.bsp',
		'TESS_EPH_DEF_2018333_01.bsp',
		'TESS_EPH_DEF_2018337_01.bsp',
		'TESS_EPH_DEF_2018340_01.bsp',
		'TESS_EPH_DEF_2018344_01.bsp',
		'TESS_EPH_DEF_2018347_01.bsp',
		'TESS_EPH_DEF_2018351_01.bsp',
		'TESS_EPH_DEF_2018354_01.bsp',
		'TESS_EPH_DEF_2018358_01.bsp',
		'TESS_EPH_DEF_2018361_01.bsp',
		'TESS_EPH_DEF_2018365_01.bsp',
		'TESS_EPH_DEF_2019003_01.bsp',
		'TESS_EPH_DEF_2019007_01.bsp',
		'TESS_EPH_DEF_2019010_01.bsp',
		'TESS_EPH_DEF_2019014_01.bsp',
		'TESS_EPH_DEF_2019017_01.bsp',
		'TESS_EPH_DEF_2019021_01.bsp',
		'TESS_EPH_DEF_2019024_01.bsp',
		'TESS_EPH_DEF_2019028_01.bsp',
		'TESS_EPH_DEF_2019031_01.bsp',
		'TESS_EPH_DEF_2019035_01.bsp',
		'TESS_EPH_DEF_2019038_01.bsp',
		'TESS_EPH_DEF_2019042_01.bsp',
		'TESS_EPH_DEF_2019045_01.bsp',
		'TESS_EPH_DEF_2019049_01.bsp',
		'TESS_EPH_DEF_2019052_01.bsp',
		'TESS_EPH_DEF_2019056_01.bsp',
		'TESS_EPH_DEF_2019059_01.bsp',
		'TESS_EPH_DEF_2019063_01.bsp',
		'TESS_EPH_DEF_2019066_01.bsp',
		'TESS_EPH_DEF_2019070_01.bsp',
		'TESS_EPH_DEF_2019073_01.bsp',
		'TESS_EPH_DEF_2019077_01.bsp',
		'TESS_EPH_DEF_2019080_01.bsp',
		'TESS_EPH_DEF_2019084_01.bsp',
		'TESS_EPH_DEF_2019087_01.bsp',
		'TESS_EPH_DEF_2019091_01.bsp',
		'TESS_EPH_DEF_2019094_01.bsp',
		'TESS_EPH_DEF_2019098_01.bsp',
		'TESS_EPH_DEF_2019102_01.bsp',
		'TESS_EPH_DEF_2019105_01.bsp',
		'TESS_EPH_DEF_2019108_01.bsp',
		'TESS_EPH_DEF_2019112_01.bsp',
		'TESS_EPH_DEF_2019115_01.bsp',
		'TESS_EPH_DEF_2019119_01.bsp',
		'TESS_EPH_DEF_2019122_01.bsp',
		'TESS_EPH_DEF_2019126_01.bsp',
		# New batch (2020-01-28):
		'TESS_EPH_DEF_2019129_01.bsp',
		'TESS_EPH_DEF_2019133_01.bsp',
		'TESS_EPH_DEF_2019136_01.bsp',
		'TESS_EPH_DEF_2019140_01.bsp',
		'TESS_EPH_DEF_2019143_01.bsp',
		'TESS_EPH_DEF_2019147_01.bsp',
		'TESS_EPH_DEF_2019150_01.bsp',
		'TESS_EPH_DEF_2019154_01.bsp',
		'TESS_EPH_DEF_2019157_01.bsp',
		'TESS_EPH_DEF_2019161_01.bsp',
		'TESS_EPH_DEF_2019164_01.bsp',
		'TESS_EPH_DEF_2019168_01.bsp',
		'TESS_EPH_DEF_2019171_01.bsp',
		'TESS_EPH_DEF_2019175_01.bsp',
		'TESS_EPH_DEF_2019178_01.bsp',
		'TESS_EPH_DEF_2019182_01.bsp',
		'TESS_EPH_DEF_2019185_01.bsp',
		'TESS_EPH_DEF_2019189_01.bsp',
		'TESS_EPH_DEF_2019192_01.bsp',
		'TESS_EPH_DEF_2019196_01.bsp',
		'TESS_EPH_DEF_2019199_01.bsp',
		'TESS_EPH_DEF_2019203_01.bsp',
		'TESS_EPH_DEF_2019206_01.bsp',
		'TESS_EPH_DEF_2019210_01.bsp',
		#'TESS_EPH_DEF_2019213_01.bsp', # Surpassed by never version below
		'TESS_EPH_DEF_2019213_02.bsp',
		#'TESS_EPH_DEF_2019217_01.bsp', # Surpassed by never version below
		'TESS_EPH_DEF_2019217_02.bsp',
		#'TESS_EPH_DEF_2019220_01.bsp', # Surpassed by never version below
		'TESS_EPH_DEF_2019220_02.bsp',
		#'TESS_EPH_DEF_2019224_01.bsp', # Surpassed by never version below
		'TESS_EPH_DEF_2019224_02.bsp',
		'TESS_EPH_DEF_2019227_02.bsp',
		'TESS_EPH_DEF_2019231_02.bsp',
		'TESS_EPH_DEF_2019234_02.bsp',
		'TESS_EPH_DEF_2019238_02.bsp',
		'TESS_EPH_DEF_2019241_21.bsp',
		'TESS_EPH_DEF_2019245_21.bsp',
		'TESS_EPH_DEF_2019248_21.bsp',
		'TESS_EPH_DEF_2019252_21.bsp',
		'TESS_EPH_DEF_2019255_21.bsp',
		'TESS_EPH_DEF_2019259_21.bsp',
		'TESS_EPH_DEF_2019262_21.bsp',
		'TESS_EPH_DEF_2019266_21.bsp',
		'TESS_EPH_DEF_2019269_21.bsp',
		'TESS_EPH_DEF_2019273_21.bsp',
		'TESS_EPH_DEF_2019276_21.bsp',
		'TESS_EPH_DEF_2019280_21.bsp',
		'TESS_EPH_DEF_2019283_21.bsp',
		'TESS_EPH_DEF_2019287_21.bsp',
		'TESS_EPH_DEF_2019290_21.bsp',
		'TESS_EPH_DEF_2019294_21.bsp',
		'TESS_EPH_DEF_2019297_21.bsp',
		'TESS_EPH_DEF_2019301_21.bsp',
		'TESS_EPH_DEF_2019304_21.bsp',
		'TESS_EPH_DEF_2019308_21.bsp',
		'TESS_EPH_DEF_2019311_21.bsp',
		'TESS_EPH_DEF_2019315_21.bsp',
		'TESS_EPH_DEF_2019318_21.bsp',
		'TESS_EPH_DEF_2019322_22.bsp',
		'TESS_EPH_DEF_2019325_22.bsp',
		'TESS_EPH_DEF_2019329_21.bsp',
		'TESS_EPH_DEF_2019332_21.bsp',
		'TESS_EPH_DEF_2019336_21.bsp',
		'TESS_EPH_DEF_2019339_21.bsp',
		'TESS_EPH_DEF_2019343_21.bsp',
		'TESS_EPH_DEF_2019346_21.bsp',
		'TESS_EPH_DEF_2019350_21.bsp',
		'TESS_EPH_DEF_2019353_21.bsp',
		'TESS_EPH_DEF_2019357_21.bsp',
		'TESS_EPH_DEF_2019360_21.bsp',
		'TESS_EPH_DEF_2019364_21.bsp',
	)

	def __init__(self, kernels_folder=None):
		"""
		Parameters:
			kernels_folder (string): If not provided, the path stored in the environment
				variable ``TESSPHOT_SPICE_KERNELS`` is used, and if that is not set, the
				``data/spice`` directory is used.
		"""

		logger = logging.getLogger(__name__)

		# If no kernel folder is given, used the one stored in env.var. or the default location:
		if kernels_folder is None:
			kernels_folder = os.environ.get('TESSPHOT_SPICE_KERNELS', os.path.join(os.path.dirname(__file__), 'data', 'spice'))

		# Make sure the kernel directory exists:
		os.makedirs(kernels_folder, exist_ok=True)

		# Automatically download kernels from TASOC, if they don't already exist?
		#urlbase = 'https://archive.stsci.edu/missions/tess/models/'
		urlbase = 'https://tasoc.dk/pipeline/spice/'
		downlist = []
		for fname in self.kernel_files:
			fpath = os.path.join(kernels_folder, fname)
			if not os.path.exists(fpath):
				downlist.append([urlbase + fname, fpath])

		if downlist:
			download_parallel(downlist)

		# Path where meta-kernel will be saved:
		fileshash = hashlib.md5(','.join(self.kernel_files).encode()).hexdigest()
		self.METAKERNEL = os.path.abspath(os.path.join(kernels_folder, 'metakernel-' + fileshash + '.txt'))

		# Write meta-kernel to file:
		if not os.path.exists(self.METAKERNEL):
			with open(self.METAKERNEL, 'w') as fid:
				fid.write("KPL/MK\n")
				fid.write(r"\begindata" + "\n")
				fid.write("PATH_VALUES = ('" + os.path.abspath(kernels_folder) + "')\n")
				fid.write("PATH_SYMBOLS = ('KERNELS')\n")
				fid.write("KERNELS_TO_LOAD = (\n")
				fid.write(",\n".join(["'$KERNELS/" + fname + "'" for fname in self.kernel_files]))
				fid.write(")\n")
				fid.write(r"\begintext" + "\n")
				fid.write("End of MK file.\n")

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

		# Define TESS object if it doesn't already exist:
		try:
			spiceypy.bodn2c('TESS')
		except SpiceyError:
			logger.debug("Defining TESS name in SPICE")
			spiceypy.boddef('TESS', -95)

		# Load kernels if needed:
		if not already_loaded:
			logger.debug("Loading SPICE Meta-kernel: %s", self.METAKERNEL)
			spiceypy.furnsh(self.METAKERNEL)

		# Let's make sure astropy is using the de430 kernels as well:
		# Default is to use the same as is being used by SPOC (de430).
		# If using astropy 4.0+, we can load the local one directly. Before this,
		# it needs to be downloaded and cached:
		# NOTE: https://github.com/astropy/astropy/pull/8767
		#self.planetary_ephemeris = 'de430'
		if astropy_major_version >= 4:
			self.planetary_ephemeris = os.path.abspath(os.path.join(kernels_folder, 'tess2018338154429-41241_de430.bsp'))
		else:
			self.planetary_ephemeris = urlbase + 'tess2018338154429-41241_de430.bsp'
		self._old_solar_system_ephemeris = coord.solar_system_ephemeris.get()
		coord.solar_system_ephemeris.set(self.planetary_ephemeris)

	def unload(self):
		"""Unload TESS SPICE kernels from memory."""
		spiceypy.unload(self.METAKERNEL)

	def close(self):
		"""Close SPICE object."""
		#self.unload() # Uhh, we are being naugthy here!
		coord.solar_system_ephemeris.set(self._old_solar_system_ephemeris)

	def __enter__(self):
		return self

	def __exit__(self, *args, **kwargs):
		self.close()

	#--------------------------------------------------------------------------
	def position(self, jd, of='TESS', relative_to='EARTH'):
		"""
		Returns position of TESS for the given timestamps as geocentric XYZ-coordinates in kilometers.

		Parameters:
			jd (ndarray): Time in Julian Days where position of TESS should be calculated.
			of (string, optional): Object for which to calculate position for. Default='TESS'.
			relative_to (string, optional): Object for which to calculate position relative to. Default='EARTH'.

		Returns:
			ndarray: Position of TESS as geocentric XYZ-coordinates in kilometers.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		# Convert JD to Ephemeris Time:
		jd = np.atleast_1d(jd)
		times = [spiceypy.unitim(j, 'JDTDB', 'ET') for j in jd]

		# Get positions as a 2D array of (x,y,z) coordinates in km:
		try:
			positions, lt = spiceypy.spkpos(of, times, 'J2000', 'NONE', relative_to)
			positions = np.atleast_2d(positions)
		except SpiceyError as e:
			if 'SPICE(SPKINSUFFDATA)' in e.value:
				raise InadequateSpiceException("Inadequate SPICE kernels available")
			else:
				raise

		return positions

	#--------------------------------------------------------------------------
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
		obstimes = Time(jd, format='jd', scale='tdb')
		cartrep = coord.CartesianRepresentation(positions, xyz_axis=1, unit=u.km)
		gcrs = coord.GCRS(cartrep, obstime=obstimes)
		itrs = gcrs.transform_to(coord.ITRS(obstime=obstimes))

		# Create EarthLocation object
		return coord.EarthLocation.from_geocentric(*itrs.cartesian.xyz, unit=u.km)

	#--------------------------------------------------------------------------
	def position_velocity(self, jd, of='TESS', relative_to='EARTH'):
		"""
		Returns position and velocity of TESS for the given timestamps as geocentric XYZ-coordinates in kilometers.

		Parameters:
			jd (ndarray): Time in Julian Days where position of TESS should be calculated.
			of (string, optional): Object for which to calculate position for. Default='TESS'.
			relative_to (string, optional): Object for which to calculate position relative to. Default='EARTH'.

		Returns:
			ndarray: Position and velocity of TESS as geocentric XYZ-coordinates in kilometers.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""

		# Convert JD to Ephemeris Time:
		jd = np.atleast_1d(jd)
		times = [spiceypy.unitim(j, 'JDTDB', 'ET') for j in jd]

		# Get state of spacecraft (position and velocity):
		try:
			pos_vel, lt = spiceypy.spkezr(of, times, 'J2000', 'NONE', relative_to)
			pos_vel = np.asarray(pos_vel)
		except SpiceyError as e:
			if 'SPICE(SPKINSUFFDATA)' in e.value:
				raise InadequateSpiceException("Inadequate SPICE kernels available")
			else:
				raise

		return pos_vel

	#--------------------------------------------------------------------------
	def velocity(self, *args, **kwargs):
		"""
		Parameters:
			jd (ndarray): Time in Julian Days where position of TESS should be calculated.
			of (string, optional): Object for which to calculate position for. Default='TESS'.
			relative_to (string, optional): Object for which to calculate position relative to. Default='EARTH'.

		Returns:
			ndarray: Velocity of TESS as geocentric XYZ-coordinates in kilometers per seconds.

		.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
		"""
		return self.position_velocity(*args, **kwargs)[:, 3:]

	#--------------------------------------------------------------------------
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

	#--------------------------------------------------------------------------
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

	#--------------------------------------------------------------------------
	def barycorr2(self, times, star_coord):
		"""
		Barycentric time correction (experimental).

		Made from scratch and includes both Rømer, Einstein and Shapiro delays.

		Parameters:
			tm (ndarray): Timestamps in UTC Julian Date (JD).
			star_coord (SkyCoord object): Coordinates of star.
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
