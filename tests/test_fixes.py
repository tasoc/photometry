#!/bin/env python
# -*- coding: utf-8 -*-
"""

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import numpy as np
import os.path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from photometry import fixes

#---------------------------------------------------------------------------------------------------
def test_fixes_time_offset():

	# Early releases that should be corrected:
	assert fixes.time_offset_should_be_fixed(10, 20, 'shouldnt-matter') == True
	assert fixes.time_offset_should_be_fixed(19, 26, 'shouldnt-matter') == True

	# Sector 20 that should be corrected:
	for procver in ('spoc-4.0.14-20200108', 'spoc-4.0.15-20200114', 'spoc-4.0.17-20200130'):
		assert fixes.time_offset_should_be_fixed(20, 27, procver) == True

	# Sector 20 that should NOT be corrected:
	for procver in ('spoc-4.0.26-20200323', 'spoc-4.0.27-20200326'):
		assert fixes.time_offset_should_be_fixed(20, 27, procver) == False

	# Sector 20 that should not be corrected (hypothetical future data relase #99):
	assert fixes.time_offset_should_be_fixed(20, 99, 'shouldnt-matter') == False

	# Sector 21 that should be corrected:
	for procver in ('spoc-4.0.17-20200130', 'spoc-4.0.20-20200220', 'spoc-4.0.21-20200227'):
		assert fixes.time_offset_should_be_fixed(21, 29, procver) == True

	# Sector 21 that should NOT be corrected:
	#for procver in ():
	#	assert fixes.time_offset_should_be_fixed(21, 29, procver) == False

	# Sector 21 that should not be corrected (hypothetical future data relase #99):
	assert fixes.time_offset_should_be_fixed(21, 99, 'shouldnt-matter') == False

#---------------------------------------------------------------------------------------------------
def test_fixes_time_offset_apply():

	time = np.linspace(1000, 2000, 200, dtype='float64')

	time_corrected_mid1 = fixes.time_offset_apply(time)
	time_corrected_mid2 = fixes.time_offset_apply(time, 'mid')

	# Calling with and without timepos should yield the same result:
	np.testing.assert_allclose(time_corrected_mid1, time_corrected_mid2)

	# Test mid-time correction:
	print( (time - time_corrected_mid1)*86400 )
	np.testing.assert_allclose((time_corrected_mid1 - time)*86400, -1.979)

	# Test start-time correction:
	time_corrected_start = fixes.time_offset_apply(time, 'start')
	print( (time - time_corrected_start)*86400 )
	np.testing.assert_allclose((time_corrected_start - time)*86400, -1.969)

	# Test end-time correction:
	time_corrected_end = fixes.time_offset_apply(time, 'end')
	print( (time - time_corrected_end)*86400 )
	np.testing.assert_allclose((time_corrected_end - time)*86400, -1.989)

	# Check that it raises an ValueError:
	with pytest.raises(ValueError):
		fixes.time_offset_apply(time, 'invalid-input')
