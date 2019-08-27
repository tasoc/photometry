#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:54:10 2017

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import sys
import os
try:
	from tempfile import TemporaryDirectory
except ImportError:
	from backports.tempfile import TemporaryDirectory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from photometry import HaloPhotometry, STATUS
import logging
import pytest

#------------------------------------------------------------------------------
@pytest.mark.skipif(os.environ.get('CI') == 'true' and os.environ.get('TRAVIS') == 'true',
					reason="This is simply too slow to run on Travis. We need to do something about that.'")
def test_halo():

	INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')

	with TemporaryDirectory() as OUTPUT_DIR:
		for datasource in ('tpf', 'ffi'):
			with HaloPhotometry(267211065, INPUT_DIR, OUTPUT_DIR, plot=True, datasource=datasource, sector=1, camera=3, ccd=2) as pho:

				pho.photometry()
				pho.save_lightcurve()
				print( pho.lightcurve )

				# It should set the status to one of these:
				print(pho.status)
				assert(pho.status in (STATUS.OK, STATUS.WARNING))

				# They shouldn't be exactly zero:
				assert( ~np.all(pho.lightcurve['flux'] == 0) )
				assert( ~np.all(pho.lightcurve['pos_centroid'][:,0] == 0) )
				assert( ~np.all(pho.lightcurve['pos_centroid'][:,1] == 0) )

				# They shouldn't be NaN (in this case!):
				assert( ~np.all(np.isnan(pho.lightcurve['flux'])) )
				assert( ~np.all(np.isnan(pho.lightcurve['pos_centroid'][:,0])) )
				assert( ~np.all(np.isnan(pho.lightcurve['pos_centroid'][:,1])) )

				print("Passed Tests for %s" % datasource)

#------------------------------------------------------------------------------
if __name__ == '__main__':

	# Setup logging:
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	console = logging.StreamHandler()
	console.setFormatter(formatter)
	logger_phot = logging.getLogger('photometry')
	if not logger_phot.hasHandlers(): logger_phot.addHandler(console)
	logger_phot.setLevel(logging.INFO)

	test_halo()
