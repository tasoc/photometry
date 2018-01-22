#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 17:21:56 2018

@author: Jonas Svenstrup Hansen <jonas.svenstrup@gmail.com>
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import random

if __package__ is None:
	import sys
	from os import path
	sys.path.append( path.dirname( path.dirname( path.abspath(__file__))))
	
	from photometry.psf import PSF
	from photometry.utilities import mag2flux

class simulateFITS(object):
	def __init__(self):
		self.stamp = (-100,100,-100,100)





if __name__ == '__main__':
	sim = simulateFITS()
	KPSF = PSF(20, 1, sim.stamp)
	KPSF.plot()


