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
	Argabrightening = 64
	ApertureCosmic = 128
	ManualExclude = 256
	SensitivityDropout = 1024
	ImpulsiveOutlier = 2048
	ArgabrighteningOnCCD = 4096
	CollateralCosmic = 8192
	DetectorAnomaly = 16384
	NoFinePoint = 32768
	NoData = 65536
	RollingBandInAperture = 131072
	RollingBandInMask = 262144
	PossibleThrusterFiring = 524288
	ThrusterFiring = 1048576

	# Which is the recommended QUALITY mask to identify bad data?
	DEFAULT_BITMASK = (AttitudeTweak | SafeMode | CoarsePoint | EarthPoint |
					   Desat | ApertureCosmic | ManualExclude |
					   DetectorAnomaly | NoData | ThrusterFiring)

	# This bitmask includes flags that are known to identify both good and bad cadences.
	# Use it wisely.
	HARD_BITMASK = (DEFAULT_BITMASK | SensitivityDropout | CollateralCosmic |
					PossibleThrusterFiring)

	# Using this bitmask only QUALITY == 0 cadences will remain
	HARDEST_BITMASK = 2096639

	# Pretty string descriptions for each flag
	STRINGS = {
		1: "Attitude tweak",
		2: "Safe mode",
		4: "Coarse point",
		8: "Earth point",
		16: "Zero crossing",
		32: "Desaturation event",
		64: "Argabrightening",
		128: "Cosmic ray in optimal aperture",
		256: "Manual exclude",
		1024: "Sudden sensitivity dropout",
		2048: "Impulsive outlier",
		4096: "Argabrightening on CCD",
		8192: "Cosmic ray in collateral data",
		16384: "Detector anomaly",
		32768: "No fine point",
		65536: "No data",
		131072: "Rolling band in optimal aperture",
		262144: "Rolling band in full mask",
		524288: "Possible thruster firing",
		1048576: "Thruster firing"
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