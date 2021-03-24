#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
from numpy.polynomial.polynomial import polyval
from numpy.polynomial.legendre import legval

class polynomial():
	def __init__(self, xmlpoly):

		self.type = xmlpoly.get('type', 'standard')
		self.offsetx = float(xmlpoly.get('offsetx', 0))
		self.scalex = float(xmlpoly.get('scalex', 1))
		self.originx = float(xmlpoly.get('originx', 0))
		#self.xindex = xmlpoly.get('xindex')
		self.maxDomain = float(xmlpoly.get('maxDomain'))

		N = int(xmlpoly.get('order')) + 1
		self.coeffs = np.zeros(N, dtype='float64')
		for k,coeff in enumerate(xmlpoly.findall('./coeffs/coeff')):
			self.coeffs[k] = coeff.get('value')

		self.covariances = np.zeros(N*N, dtype='float64')
		for k,cov in enumerate(xmlpoly.findall('./covariances/covariance')):
			self.covariances[k] = cov.get('value')
		self.covariances = self.covariances.reshape((N,N))

	def __call__(self, x):
		if self.type == 'standard':
			return polyval(self.offsetx + self.scalex * (x - self.originx), self.coeffs)

		elif self.type == 'legendre':
			return legval(self.offsetx + self.scalex * (x - self.originx), self.coeffs)

		elif self.type == 'NotScaled':
			#return polyval(x, self.coeffs)
			raise NotImplementedError("Polynomial of type 'NotScaled' is not implemented yet.")

		else:
			raise ValueError("Unknown polynomial type: '%s'", self.type)
