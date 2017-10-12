#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collection of utility functions that can be used throughout
the photometry package.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
from six.moves import range, zip

#------------------------------------------------------------------------------
def enum(*sequential, **named):
	"""
	Handy way to fake an enumerated type in Python
	http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
	"""
	enums = dict(zip(sequential, range(len(sequential))), **named)
	return type('Enum', (), enums)
