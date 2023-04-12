#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests of photometry.utilities.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
import random
import logging
import tempfile
import numpy as np
import requests
import responses
import httpretty
import conftest # noqa: F401
import photometry.utilities as u

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')

#--------------------------------------------------------------------------------------------------
def test_move_median_central():

	x_1d = np.array([4, 2, 2, 0, 0, np.NaN, 0, 2, 2, 4])
	result_1d = u.move_median_central(x_1d, 3)
	expected_1d = np.array([3, 2, 2, 0, 0, 0, 1, 2, 2, 3])
	np.testing.assert_allclose(result_1d, expected_1d)

	x_2d = np.tile(x_1d, (10, 1))
	result_2d = u.move_median_central(x_2d, 3, axis=1)
	expected_2d = np.tile(expected_1d, (10, 1))
	np.testing.assert_allclose(result_2d, expected_2d)

#--------------------------------------------------------------------------------------------------
def test_sphere_distance():
	np.testing.assert_allclose(u.sphere_distance(0, 0, 90, 0), 90)
	np.testing.assert_allclose(u.sphere_distance(90, 0, 0, 0), 90)
	np.testing.assert_allclose(u.sphere_distance(0, -90, 0, 90), 180)
	np.testing.assert_allclose(u.sphere_distance(45, 45, 45, 45), 0)
	np.testing.assert_allclose(u.sphere_distance(33.2, 45, 33.2, -45), 90)
	np.testing.assert_allclose(u.sphere_distance(337.5, 0, 22.5, 0), 45)
	np.testing.assert_allclose(u.sphere_distance(22.5, 0, 337.5, 0), 45)
	np.testing.assert_allclose(u.sphere_distance(0, 0, np.array([0, 90]), np.array([90, 90])), np.array([90, 90]))

#--------------------------------------------------------------------------------------------------
def test_coordtransforms():

	inp = np.array([[0, 0], [0, 90], [0, -90], [30, 0]], dtype='float64')

	expected_xyz = np.array([
		[1, 0, 0],
		[0, 0, 1],
		[0, 0, -1],
		[np.cos(30*np.pi/180), np.sin(30*np.pi/180), 0]
	], dtype='float64')

	xyz = u.radec_to_cartesian(inp)
	print( xyz )
	print( expected_xyz )

	print( xyz - expected_xyz )

	np.testing.assert_allclose(xyz, expected_xyz, atol=1e-7)

	# Transform back:
	radec2 = u.cartesian_to_radec(xyz)
	print( radec2 )

	# Test that we recoved the input:
	np.testing.assert_allclose(radec2, inp, atol=1e-7)

#--------------------------------------------------------------------------------------------------
def test_rms_timescale():

	time = np.linspace(0, 27, 100)
	flux = np.zeros(len(time))

	rms = u.rms_timescale(time, flux)
	print(rms)
	np.testing.assert_allclose(rms, 0)

	rms = u.rms_timescale(time, flux*np.nan)
	print(rms)
	assert np.isnan(rms), "Should return nan on pure nan input"

	rms = u.rms_timescale([], [])
	print(rms)
	assert np.isnan(rms), "Should return nan on empty input"

	# Pure nan in the time-column should raise ValueError:
	with pytest.raises(ValueError):
		rms = u.rms_timescale(time*np.nan, flux)

	# Test with timescale longer than timespan should return zero:
	flux = np.random.randn(1000)
	time = np.linspace(0, 27, len(flux))
	rms = u.rms_timescale(time, flux, timescale=30.0)
	print(rms)
	np.testing.assert_allclose(rms, 0)

	# Time with infinity should give ValueError:
	time[1] = np.Inf
	with pytest.raises(ValueError):
		u.rms_timescale(time, flux)

	# Time with negative infinity should give ValueError:
	time[1] = np.NINF
	with pytest.raises(ValueError):
		u.rms_timescale(time, flux)

	# All timestamps being the same should give ValueError:
	time[:] = 1.2
	with pytest.raises(ValueError):
		u.rms_timescale(time, flux)

#--------------------------------------------------------------------------------------------------
def test_find_nearest():

	a = np.arange(10, dtype='float64')
	print(a)

	assert u.find_nearest(a, 0) == 0
	assert u.find_nearest(a, -0.4) == 0
	assert u.find_nearest(a, 5) == 5
	assert u.find_nearest(a, 4.5) == 4 # should return the first match
	assert u.find_nearest(a, 9) == 9
	assert u.find_nearest(a, 9.4) == 9
	assert u.find_nearest(a, 1.2) == 1

	# With strange search values:
	assert u.find_nearest(a, np.Inf) == 9
	assert u.find_nearest(a, -np.Inf) == 0
	with pytest.raises(ValueError):
		u.find_nearest(a, np.NaN)

	# with NaN:
	a[1] = np.NaN
	print(a)

	assert u.find_nearest(a, 0) == 0
	assert u.find_nearest(a, -0.4) == 0
	assert u.find_nearest(a, 5) == 5
	assert u.find_nearest(a, 4.5) == 4 # should return the first match
	assert u.find_nearest(a, 9) == 9
	assert u.find_nearest(a, 9.4) == 9
	assert u.find_nearest(a, 1.2) == 2

	# With MaskedArray:
	a = np.ma.masked_array(a, mask=np.isnan(a))
	print(a)

	assert u.find_nearest(a, 0) == 0
	assert u.find_nearest(a, -0.4) == 0
	assert u.find_nearest(a, 5) == 5
	assert u.find_nearest(a, 4.5) == 4 # should return the first match
	assert u.find_nearest(a, 9) == 9
	assert u.find_nearest(a, 9.4) == 9
	assert u.find_nearest(a, 1.2) == 2

#--------------------------------------------------------------------------------------------------
def test_mag2flux():

	mags = np.linspace(-1, 20, 30)
	flux = u.mag2flux(mags)

	assert np.all(np.isfinite(flux)), "MAG2FLUX should give finite fluxes on finite mags"
	assert np.all(np.diff(flux) < 0), "MAG2FLUX should give montomical decreasing values"
	assert np.isnan(u.mag2flux(np.NaN)), "MAG2FLUX should return NaN on NaN input"

#--------------------------------------------------------------------------------------------------
def test_download_file(caplog):
	"""Test downloading of file, with mocking of HTTP requests"""
	with tempfile.TemporaryDirectory() as tmpdir:

		fpath = os.path.join(tmpdir, 'nevercreated.txt')
		assert not os.path.exists(fpath), "File already exists"

		url = 'https://tasoc.dk/foobar.json'
		with responses.RequestsMock() as rsps:
			rsps.add(responses.GET, url, status=500)

			with caplog.at_level(logging.CRITICAL): # Silence logger error
				with pytest.raises(requests.exceptions.RetryError):
					u.download_file(url, fpath)

		# The output file should not exist after failure:
		assert not os.path.exists(fpath), "File shouldn't exist"

		# Test server not providing correct content-length header:
		with responses.RequestsMock() as rsps:
			rsps.add(responses.GET, url, status=200,
				body='{}',
				content_type='application/json',
				headers={'content-length': '1234567890'})

			with caplog.at_level(logging.CRITICAL): # Silence logger error
				with pytest.raises(RuntimeError):
					u.download_file(url, fpath)

		# The output file should not exist after failure:
		assert not os.path.exists(fpath), "File shouldn't exist"

		# The body text that should be returned by a successful response:
		body = '{"random":' + str(random.randrange(1, 100000000)) + '}'

		# Try with a successful response:
		url = 'https://tasoc.dk/foobar.json'
		fpath = os.path.join(tmpdir, 'foobar.json')
		assert not os.path.exists(fpath), "File already exists"
		with responses.RequestsMock() as rsps:
			rsps.add(responses.GET, url, status=200,
				body=body,
				content_type='application/json',
				auto_calculate_content_length=True)
			u.download_file(url, fpath)

		# The file should now exist and contain the body of the response:
		assert os.path.isfile(fpath)

		# Check the file contents:
		with open(fpath, 'r') as fid:
			filecontents = fid.read()
		assert filecontents == body

#--------------------------------------------------------------------------------------------------
def test_download_file_retries(caplog):
	"""
	Test retrying downloading of file, with mocking of HTTP requests

	Because of an issue with the "responses" package, we are instead
	using the "httpretty" package here:
	https://github.com/getsentry/responses/issues/135

	Ideally we would like to avoid this dependence, but there is no
	other way arround it right now.
	"""

	# The number of retries allowed in utilities.download_file:
	max_retries = 3

	with tempfile.TemporaryDirectory() as tmpdir:

		fpath = os.path.join(tmpdir, 'nevercreated.txt')
		assert not os.path.exists(fpath), "File already exists"

		url = 'https://tasoc.dk/nevercreated.json'
		with httpretty.enabled(allow_net_connect=False):
			httpretty.register_uri(httpretty.GET, url, responses=[
				httpretty.Response(body='{"message": "HTTPretty :)"}', status=429),
				httpretty.Response(body='{"message": "HTTPretty :)"}', status=200),
			])

			u.download_file(url, fpath)

			# The file should now have been downloaded:
			assert os.path.exists(fpath), "File should exist"

			# And we should have made 2 HTTP requests, becuse we did one retry:
			assert len(httpretty.latest_requests()) == 2

			# Reset and make "server" even less responsive:
			httpretty.reset()
			httpretty.register_uri(httpretty.GET, url, responses=[
				httpretty.Response(body='{"message": "HTTPretty :)"}', status=429),
				httpretty.Response(body='{"message": "HTTPretty :)"}', status=429),
				httpretty.Response(body='{"message": "HTTPretty :)"}', status=500),
				httpretty.Response(body='{"message": "HTTPretty :)"}', status=500),
				httpretty.Response(body='{"message": "HTTPretty :)"}', status=200),
			])

			# Try again, but now we should end up with a RetryError:
			with caplog.at_level(logging.CRITICAL): # Silence logger error
				with pytest.raises(requests.exceptions.RetryError):
					u.download_file(url, fpath)

			# We should have done one more request than the maximum number of retries:
			assert len(httpretty.latest_requests()) == max_retries + 1

#--------------------------------------------------------------------------------------------------
def test_download_parallel():
	"""Test parallel downloading of files, with mocking of HTTP requests"""

	# The body text that should be returned by a successful response:
	body_success = '{"random":' + str(random.randrange(1, 100000000)) + '}'

	with tempfile.TemporaryDirectory() as tmpdir:

		url = 'https://tasoc.dk/foobar.json'
		urls = [
			[url, os.path.join(tmpdir, 'foobar1.json')],
			[url, os.path.join(tmpdir, 'foobar2.json')],
		]

		with responses.RequestsMock() as rsps:
			rsps.add(responses.GET, url, status=200,
				body=body_success,
				content_type='application/json',
				auto_calculate_content_length=True)

			u.download_parallel(urls)

		# The files should now exist and contain the body of the response:
		for url, fpath in urls:
			assert os.path.isfile(fpath)

			# Check the file contents:
			with open(fpath, 'r') as fid:
				filecontents = fid.read()
			assert filecontents == body_success

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
