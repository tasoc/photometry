#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Handling of TESS data quality flags.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np

#--------------------------------------------------------------------------------------------------
class QualityFlagsBase(object):

	# Using this bitmask only QUALITY == 0 cadences will remain
	HARDEST_BITMASK = 2**32-1

	@classmethod
	def decode(cls, quality):
		"""
		Converts a QUALITY value into a list of human-readable strings.
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
			if quality & flag != 0:
				result.append(cls.STRINGS[flag])
		return result

	@classmethod
	def filter(cls, quality, flags=None):
		"""
		Filter quality flags against a specific set of flags.

		Parameters:
			quality (integer or ndarray): Quality flags.
			flags (integer bitmask): Default=``TESSQualityFlags.DEFAULT_BITMASK``.

		Returns:
			ndarray: ``True`` if quality DOES NOT contain any of the specified ``flags``, ``False`` otherwise.

		"""
		if flags is None: flags = cls.DEFAULT_BITMASK
		return (quality & flags == 0)

	@staticmethod
	def binary_repr(quality):
		"""
		Binary representation of the quality flag.

		Parameters:
			quality (int or ndarray): Quality flag.

		Returns:
			string: Binary representation of quality flag. String will be 32 characters long.

		"""
		if isinstance(quality, (np.ndarray, list, tuple)):
			return np.array([np.binary_repr(q, width=32) for q in quality])
		else:
			return np.binary_repr(quality, width=32)

#--------------------------------------------------------------------------------------------------
class CorrectorQualityFlags(QualityFlagsBase):
	"""
	This class encodes the meaning of the various TESS QUALITY bitmask flags.
	"""
	FlaggedBadData = 1
	ManualExclude = 2
	SigmaClip = 4
	JumpAdditiveConstant = 8
	JumpAdditiveLinear = 16
	JumpMultiplicativeConstant = 32
	JumpMultiplicativeLinear = 64
	Interpolated = 128
	BackgroundShenanigans = 256

	# Default bitmask
	DEFAULT_BITMASK = (FlaggedBadData | ManualExclude)

	# Pretty string descriptions for each flag
	STRINGS = {
		FlaggedBadData: "Bad data based on pixel flags",
		ManualExclude: "Manual exclude",
		SigmaClip: "Point removed due to sigma clipping",
		JumpAdditiveConstant: "Jump corrected using additive constant",
		JumpAdditiveLinear: "Jump corrected using additive linear trend",
		JumpMultiplicativeConstant: "Jumb corrected using multiplicative constant",
		JumpMultiplicativeLinear: "Jump corrected using multiplicative linear trend",
		Interpolated: "Point is interpolated",
		BackgroundShenanigans: "Background Shenanigans detected in stamp",
	}

#--------------------------------------------------------------------------------------------------
class TESSQualityFlags(QualityFlagsBase):
	"""
	This class encodes the meaning of the various TESS PIXEL_QUALITY bitmask flags.
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
	ScatteredLight = 4096

	# Which is the recommended QUALITY mask to identify bad data?
	DEFAULT_BITMASK = (AttitudeTweak | SafeMode | CoarsePoint | EarthPoint |
					   Desat | ApertureCosmic | ManualExclude)

	# This bitmask includes flags that are known to identify both good and bad cadences.
	# Use it wisely.
	HARD_BITMASK = (DEFAULT_BITMASK | SensitivityDropout | CollateralCosmic)

	# Bitmask of all of the flags that are relevant to FFIs:
	# We are on purpose not including ManualExclude here, as it would
	# result in many (~20%) of FFI data to be marked with ManualExclude
	# and therefore be rejected in the following processing.
	# There is also no reason for why a single timestamp in a TPF marked
	# as ManualExclude should necessarily cause the FFI timestamp to be invalid.
	FFI_RELEVANT_BITMASK = (AttitudeTweak | SafeMode | CoarsePoint | EarthPoint |
					   Desat | EarthMoonPlanetInFOV | ScatteredLight)

	# Pretty string descriptions for each flag
	STRINGS = {
		AttitudeTweak: "Attitude tweak",
		SafeMode: "Safe mode",
		CoarsePoint: "Spacecraft in Coarse point",
		EarthPoint: "Spacecraft in Earth point",
		ZeroCrossing: "Reaction wheel zero crossing",
		Desat: "Reaction wheel desaturation event",
		ApertureCosmic: "Cosmic ray in optimal aperture pixel",
		ManualExclude: "Manual exclude",
		SensitivityDropout: "Sudden sensitivity dropout",
		ImpulsiveOutlier: "Impulsive outlier",
		CollateralCosmic: "Cosmic ray in collateral data",
		EarthMoonPlanetInFOV: "Earth, Moon or other planet in camera FOV"
	}

#--------------------------------------------------------------------------------------------------
class PixelQualityFlags(QualityFlagsBase):
	"""
	This class encodes the meaning of the various TESS QUALITY bitmask flags.
	"""
	NotUsedForBackground = 1
	ManualExclude = 2
	BackgroundShenanigans = 4

	# Default bitmask
	DEFAULT_BITMASK = (ManualExclude)

	# Pretty string descriptions for each flag
	STRINGS = {
		NotUsedForBackground: "Pixel was not used in background calculation",
		ManualExclude: "Manual exclude",
		BackgroundShenanigans: "Background Shenanigans detected in pixel",
	}
