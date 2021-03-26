#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper around `astropy.nddata.CCDData` that allows for normal math.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
from astropy.nddata import CCDData

class CalibImage(CCDData):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		#if 'aperture' not in self.meta:
		#	raise ValueError("APERTURE not defined")
		#if self.aperture.shape != self.shape:
		#	raise ValueError("APERTURE has wrong shape")

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
		if value.uncertainty is not None:
			self.uncertainty.array[index] = value.uncertainty.array
		#self.mask[index] = value.mask

	def __getitem__(self, item):
		new = super().__getitem__(item)

		if 'aperture' in self.meta:
			new.meta['aperture'] = self.meta['aperture'][item]

		#print(item)
		#if 'index_rows' in self.meta:
		#	new.meta['index_rows'] = self.meta['index_rows'][item]
		#if 'index_columns' in self.meta:
		#	new.meta['index_columns'] = self.meta['index_rows'][item]

		return new

	@property
	def aperture(self):
		return self.meta.get('aperture')

	@property
	def outputs(self):
		return self.aperture & (32 + 64 + 128 + 256)

	@property
	def iter_outputs(self):
		outputs = self.aperture & (32 + 64 + 128 + 256)
		for out in np.unique(self.outputs):
			outname = {32: 'A', 64: 'B', 128: 'C', 256: 'D'}[out]
			outmask = (outputs == out)
			yield outname, outmask

	@property
	def rows(self):
		return self.meta['index_rows']

	@property
	def cols(self):
		return self.meta['index_columns']
