#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Difference Imaging Photometry.

.. codeauthor:: Isabel Colman <EMAIL@HERE>
"""

from __future__ import division, with_statement, print_function, absolute_import
from six.moves import range, zip
import numpy as np
import logging
from .. import BasePhotometry, STATUS

#------------------------------------------------------------------------------
class DiffImgPhotometry(BasePhotometry):
	"""Simple Aperture Photometry using K2P2 to define masks.

	.. codeauthor:: Isabel Colman <EMAIL@HERE>
	"""

	def __init__(self, *args, **kwargs):
		# Call the parent initializing:
		# This will set several default settings
		super(self.__class__, self).__init__(*args, **kwargs)

		# Here you could do other things that needs doing in the beginning
		# of the run on each target.


	def do_photometry(self):
		"""Perform photometry on the given target.

		This function needs to set
			* self.lightcurve
		"""

		# TODO: Perform magic here!

		# Return whether you think it went well:
		return STATUS.OK
