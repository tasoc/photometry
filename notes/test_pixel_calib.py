# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:11:53 2020

@author: au195407
"""

import os.path
import numpy as np
from astropy.io import fits
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from photometry.pixel_calib import PixelCalibrator
from photometry.plots import plt

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':

	plt.switch_backend('Qt5Agg')

	with fits.open('tess2018206045859-s0001-0000000008195886-0120-s_tp.fits.gz', mode='readonly') as tpf:

		with PixelCalibrator(camera=tpf[0].header['CAMERA'], ccd=tpf[0].header['CCD']) as pcal:
			print(pcal)

			pcal.plot_flatfield()
			pcal.plot_twodblack()

			tpf_cal = pcal.calibrate_tpf(tpf)

			plt.show()

