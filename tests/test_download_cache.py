#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test download_cache and the corresponding CLI program.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os
import tempfile
from conftest import capture_cli
import photometry

#--------------------------------------------------------------------------------------------------
@pytest.mark.web
def test_download_cache(PRIVATE_SPICE_DIR):

	# Delete a couple of small SPICE kernels:
	os.remove(os.path.join(PRIVATE_SPICE_DIR, 'TESS_EPH_DEF_2018108_02.bsp'))
	os.remove(os.path.join(PRIVATE_SPICE_DIR, 'TESS_EPH_DEF_2018115_01.bsp'))

	# Run download_cache directly:
	photometry.download_cache()

	# The two SPICE kernels should now be back in place:
	assert os.path.isfile(os.path.join(PRIVATE_SPICE_DIR, 'TESS_EPH_DEF_2018108_02.bsp'))
	assert os.path.isfile(os.path.join(PRIVATE_SPICE_DIR, 'TESS_EPH_DEF_2018115_01.bsp'))

#--------------------------------------------------------------------------------------------------
@pytest.mark.web
def test_run_download_cache(PRIVATE_SPICE_DIR):

	# Delete a couple of small SPICE kernels:
	os.remove(os.path.join(PRIVATE_SPICE_DIR, 'TESS_EPH_DEF_2018108_02.bsp'))
	os.remove(os.path.join(PRIVATE_SPICE_DIR, 'TESS_EPH_DEF_2018115_01.bsp'))

	# Run download_cache CLI program:
	out, err, exitcode = capture_cli('run_download_cache.py', ['--debug'])
	assert ' - ERROR - ' not in err
	assert exitcode == 0
	assert 'All cache data downloaded.' in out

	# The two SPICE kernels should now be back in place:
	assert os.path.isfile(os.path.join(PRIVATE_SPICE_DIR, 'TESS_EPH_DEF_2018108_02.bsp'))
	assert os.path.isfile(os.path.join(PRIVATE_SPICE_DIR, 'TESS_EPH_DEF_2018115_01.bsp'))

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
