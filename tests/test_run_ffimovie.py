#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of FFI Movies.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
from conftest import capture_cli
import photometry.plots # noqa: F401
from matplotlib import animation

HAS_FFMPEG = ('ffmpeg' in animation.writers)

#--------------------------------------------------------------------------------------------------
@pytest.mark.ffmpeg
@pytest.mark.skipif(not HAS_FFMPEG, reason="FFMpeg not available")
def test_run_ffimovie(SHARED_INPUT_DIR):

	out, err, exitcode = capture_cli('run_ffimovie.py', params=[SHARED_INPUT_DIR])

	assert exitcode == 0

	for fname in (
		'sector001_camera3_ccd2.mp4',
		'sector001_combined_backgrounds.mp4',
		'sector001_combined_flags.mp4',
		'sector001_combined_images.mp4',
		'sector001_combined_originals.mp4'):

		mp4file = os.path.join(SHARED_INPUT_DIR, fname)
		assert os.path.isfile(mp4file), "MP4 was not created: " + fname

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
