#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper around `astropy.nddata.CCDData` that allows for normal math.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from astropy.nddata import CCDData

class CalibImage(CCDData):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def __add__(self, other):
		return self.add(other, handle_meta='first_found')

	def __sub__(self, other):
		return self.subtract(other, handle_meta='first_found')

	def __mul__(self, other):
		return self.multiply(other, handle_meta='first_found')

	def __truediv__(self, other):
		return self.divide(other, handle_meta='first_found')

	def __neg__(self):
		return self.multiply(-1, handle_meta='first_found')

	def __radd__(self, other):
		return self.__add__(other)

	def __rsub__(self, other):
		return self.subtract(other, handle_meta='first_found').multiply(-1, handle_meta='first_found')

	def __rmul__(self, other):
		return self.__mul__(other)

	def __setitem__(self, index, value):
		self.data[index] = value.data
		self.uncertainty.array[index] = value.uncertainty.array
		#self.mask[index] = value.mask

	@property
	def rows(self):
		return self.meta['index_rows']

	@property
	def cols(self):
		return self.meta['index_columns']
