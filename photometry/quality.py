#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Handling of TESS data quality flags.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import

class TESSQualityFlags(object):
	"""
	This class encodes the meaning of the various TESS QUALITY bitmask flags.
	"""
	AttitudeTweak = 1
	SafeMode = 2
	CoarsePoint = 4
	EarthPoint = 8
	ZeroCrossing = 16
	Desat = 32
	ApertureCosmic = 64
	ManualExclude = 128
	SensitivityDropout = 256
	ImpulsiveOutlier = 512
	CollateralCosmic = 1024
	EarthMoonPlanetInFOV = 2048

	# Which is the recommended QUALITY mask to identify bad data?
	DEFAULT_BITMASK = (AttitudeTweak | SafeMode | CoarsePoint | EarthPoint |
					   Desat | ApertureCosmic | ManualExclude)

	# This bitmask includes flags that are known to identify both good and bad cadences.
	# Use it wisely.
	HARD_BITMASK = (DEFAULT_BITMASK | SensitivityDropout | CollateralCosmic)

	# Using this bitmask only QUALITY == 0 cadences will remain
	HARDEST_BITMASK = 2**32-1

	# Pretty string descriptions for each flag
	STRINGS = {
		1: "Attitude tweak",
		2: "Safe mode",
		4: "Spacecraft in Coarse point",
		8: "Spacecraft in Earth point",
		16: "Reaction wheel zero crossing",
		32: "Reaction wheel desaturation event",
		64: "Cosmic ray in optimal aperture pixel",
		128: "Manual exclude",
		256: "Sudden sensitivity dropout",
		512: "Impulsive outlier",
		1024: "Cosmic ray in collateral data",
		2048: "Earth, Moon or other planet in camera FOV"
	}

	@classmethod
	def decode(cls, quality):
		"""
		Converts a TESS QUALITY value into a list of human-readable strings.
		This function takes the QUALITY bitstring that can be found for each
		cadence in TESS data files and converts into a list of human-readable
		strings explaining the flags raised (if any).

		Parameters:
			quality (int): Value from the 'QUALITY' column of a TESS data file.

		Returns:
			list of str: List of human-readable strings giving a short
			             description of the quality flags raised.
						 Returns an empty list if no flags raised.
		"""
		result = []
		for flag in cls.STRINGS.keys():
			if quality & flag > 0:
				result.append(cls.STRINGS[flag])
		return result

	@staticmethod
	def filter(quality, flags=DEFAULT_BITMASK):
		"""
		Filter quality flags against a specific set of flags.

		Parameters:
			quality (integer or ndarray): Quality flags.
			flags (integer bitmask): Default=``TESSQualityFlags.DEFAULT_BITMASK``.

		Returns:
			ndarray: ``True`` if quality DOES NOT contain any of the specified ``flags``, ``False`` otherwise.

		"""
		return (quality & flags == 0)