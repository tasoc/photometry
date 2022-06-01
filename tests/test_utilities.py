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
import sqlite3
import configparser
import tempfile
import numpy as np
import requests
import responses
import httpretty
from urllib3.exceptions import MaxRetryError
import conftest # noqa: F401
import photometry.utilities as u

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')

#--------------------------------------------------------------------------------------------------
def test_load_settings():

	settings = u.load_settings()
	assert isinstance(settings, configparser.ConfigParser)
	assert settings.getboolean('fixes', 'time_offset', fallback=True) # Actually checking value

	# Try to run the exact same query, and it should now be taken over by the cache:
	hits_before = u.load_settings.cache_info().hits
	settings2 = u.load_settings()
	assert settings2 == settings
	assert u.load_settings.cache_info().hits == hits_before+1

#--------------------------------------------------------------------------------------------------
def test_load_sector_settings():

	settings = u.load_sector_settings(2)
	print(settings)
	assert isinstance(settings, dict)
	assert int(settings['sector']) == 2

	# Try to run the exact same query, and it should now be taken over by the cache:
	hits_before = u.load_sector_settings.cache_info().hits
	settings2 = u.load_sector_settings(2)
	assert settings2 == settings
	assert u.load_sector_settings.cache_info().hits == hits_before+1

	settings = u.load_sector_settings()
	print(settings)
	assert isinstance(settings, dict)
	assert 'sectors' in settings

	sectors = []
	for key, value in settings['sectors'].items():
		# Make sure they contain what they should:
		assert isinstance(value, dict)
		assert 'sector' in value
		assert 'reference_time' in value
		assert 'ffi_cadence' in value
		assert value['ffi_cadence'] in (1800, 600), "Invalid FFI Cadence"

		# Ensure that sector numbers are unique:
		assert value['sector'] not in sectors
		sectors.append(value['sector'])

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
def test_find_ffi_files(SHARED_INPUT_DIR):

	files = u.find_ffi_files(SHARED_INPUT_DIR)
	assert(len(files) == 10)

	files = u.find_ffi_files(SHARED_INPUT_DIR, camera=1)
	assert(len(files) == 4)

	files = u.find_ffi_files(SHARED_INPUT_DIR, camera=3)
	assert(len(files) == 4)

	files = u.find_ffi_files(SHARED_INPUT_DIR, sector=27)
	assert(len(files) == 2)

#--------------------------------------------------------------------------------------------------
def test_find_tpf_files(SHARED_INPUT_DIR):
	"""
	The current list of test-files are:

	starid    sector  camera  ccd  cadence
	267211065      1       3    2      120
	260795451      1       3    2      120
	 25155310      1       1    4      120  alert
	 25155310     27       4    1      120
	 25155310     27       4    1       20
	"""

	# Find all TPF files in input dir:
	files = u.find_tpf_files(SHARED_INPUT_DIR)
	print(files)
	assert(len(files) == 5)

	# Find file with specific starid (only one exists)
	files = u.find_tpf_files(SHARED_INPUT_DIR, starid=267211065)
	print(files)
	assert(len(files) == 1)

	# Find TPFs from sector 1 (2 regular, 1 alert-data):
	files = u.find_tpf_files(SHARED_INPUT_DIR, sector=1)
	print(files)
	assert(len(files) == 3)

	# Limit with findmax (only the first one should be returned):
	files2 = u.find_tpf_files(SHARED_INPUT_DIR, sector=1, findmax=1)
	print(files2)
	assert len(files2) == 1
	assert files2[0] == files[0]

	# Find TPFs for starid with both 120 and 20s cadences and a alert data file:
	files = u.find_tpf_files(SHARED_INPUT_DIR, starid=25155310)
	print(files)
	assert(len(files) == 3)

	# Test files from sector 27, with both 120 and 20s cadences:
	files = u.find_tpf_files(SHARED_INPUT_DIR, sector=27)
	print(files)
	assert(len(files) == 2)

	# Test files from sector 27, with only 120s cadence:
	files = u.find_tpf_files(SHARED_INPUT_DIR, sector=27, cadence=120)
	print(files)
	assert(len(files) == 1)

	# Test files from sector 27, with only 20s cadence:
	files = u.find_tpf_files(SHARED_INPUT_DIR, sector=27, cadence=20)
	print(files)
	assert(len(files) == 1)

	files = u.find_tpf_files(SHARED_INPUT_DIR, sector=1, camera=3)
	print(files)
	assert(len(files) == 2)

	# Find TPFs from sector 1, on camera 2, which should have no files:
	files = u.find_tpf_files(SHARED_INPUT_DIR, sector=1, camera=2)
	print(files)
	assert(len(files) == 0)

	# Test files from sector 27 on CCD 4, which should have no files:
	files = u.find_tpf_files(SHARED_INPUT_DIR, sector=27, ccd=4)
	print(files)
	assert(len(files) == 0)

	# Test files from sector 27, with both 120 and 20s cadences:
	files = u.find_tpf_files(SHARED_INPUT_DIR, sector=27, ccd=1)
	print(files)
	assert(len(files) == 2)

	# Test files from sector 27, with both 120 and 20s cadences:
	files = u.find_tpf_files(SHARED_INPUT_DIR, sector=27, ccd=1, findmax=1)
	print(files)
	assert(len(files) == 1)

	# Test the cache:
	hits_before = u._find_tpf_files.cache_info().hits
	files2 = u.find_tpf_files(SHARED_INPUT_DIR, sector=27, ccd=1, findmax=1)
	assert files2 == files
	assert u._find_tpf_files.cache_info().hits == hits_before+1

	# Test with invalid cadence:
	with pytest.raises(ValueError) as e:
		u.find_tpf_files(SHARED_INPUT_DIR, sector=27, cadence=123345)
	assert str(e.value) == "Invalid cadence. Must be either 20 or 120."

#--------------------------------------------------------------------------------------------------
def test_find_hdf5_files(SHARED_INPUT_DIR):

	files = u.find_hdf5_files(SHARED_INPUT_DIR)
	assert(len(files) == 2)

	files = u.find_hdf5_files(SHARED_INPUT_DIR, sector=1)
	assert(len(files) == 2)

	files = u.find_hdf5_files(SHARED_INPUT_DIR, camera=1)
	assert(len(files) == 1)

	files = u.find_hdf5_files(SHARED_INPUT_DIR, sector=1, camera=3)
	assert(len(files) == 1)

	files = u.find_hdf5_files(SHARED_INPUT_DIR, sector=1, camera=3, ccd=2)
	assert(len(files) == 1)

	# Try to run the exact same query, and it should now be taken over by the cache:
	hits_before = u.find_hdf5_files.cache_info().hits
	files2 = u.find_hdf5_files(SHARED_INPUT_DIR, sector=1, camera=3, ccd=2)
	assert files2 == files
	assert u.find_hdf5_files.cache_info().hits == hits_before+1

#--------------------------------------------------------------------------------------------------
def test_find_catalog_files(SHARED_INPUT_DIR):

	files = u.find_catalog_files(SHARED_INPUT_DIR)
	print(files)
	assert(len(files) == 3)

	files = u.find_catalog_files(SHARED_INPUT_DIR, sector=1)
	print(files)
	assert(len(files) == 2)

	files = u.find_catalog_files(SHARED_INPUT_DIR, sector=27)
	print(files)
	assert(len(files) == 1)

	files = u.find_catalog_files(SHARED_INPUT_DIR, camera=1)
	print(files)
	assert(len(files) == 1)

	files = u.find_catalog_files(SHARED_INPUT_DIR, sector=1, camera=3, ccd=2)
	print(files)
	assert(len(files) == 1)

	# Try to run the exact same query, and it should now be taken over by the cache:
	hits_before = u.find_catalog_files.cache_info().hits
	files2 = u.find_catalog_files(SHARED_INPUT_DIR, sector=1, camera=3, ccd=2)
	assert files2 == files
	assert u.find_catalog_files.cache_info().hits == hits_before+1

#--------------------------------------------------------------------------------------------------
def test_load_ffi_files(SHARED_INPUT_DIR):

	files = u.find_ffi_files(SHARED_INPUT_DIR, camera=1)

	img = u.load_ffi_fits(files[0])
	assert(img.shape == (2048, 2048))

	img, hdr = u.load_ffi_fits(files[0], return_header=True)
	assert(img.shape == (2048, 2048))

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
@pytest.mark.parametrize('factory', [None, sqlite3.Row])
@pytest.mark.parametrize('foreign_keys', [1, 0])
def test_sqlite_drop_column(factory, foreign_keys):

	with sqlite3.connect(':memory:') as conn:
		conn.row_factory = factory
		cursor = conn.cursor()

		cursor.execute("PRAGMA foreign_keys=%s;" % foreign_keys)

		# Make sure the foreign key is set as we want it:
		cursor.execute("PRAGMA foreign_keys;")
		assert cursor.fetchone()[0] == foreign_keys

		# Create test-table:
		cursor.execute("""CREATE TABLE tbl (
			col_a INTEGER,
			col_b REAL,
			col_c REAL NOT NULL,
			col_d REAL,
			col_e REAL
		);""")
		for k in range(1000):
			cursor.execute("INSERT INTO tbl VALUES (%d,RANDOM(),RANDOM(),RANDOM(),RANDOM());" % k)
		cursor.execute("CREATE UNIQUE INDEX col_a_idx ON tbl (col_a);")
		cursor.execute("CREATE INDEX col_c_idx ON tbl (col_c);")
		cursor.execute("CREATE\t INDEX  col_de_idx ON tbl (col_d, col_e);") # with strange whitespace
		conn.commit()

		# Check names of columns before we remove anything:
		cursor.execute("PRAGMA table_info(tbl);")
		s = set([row[1] for row in cursor.fetchall()])
		assert s == set(['col_a', 'col_b', 'col_c', 'col_d', 'col_e'])

		# Remove col_b:
		u.sqlite_drop_column(conn, 'tbl', 'col_b')

		# Check names of columns after we removed col_b:
		cursor.execute("PRAGMA table_info(tbl);")
		s = set([row[1] for row in cursor.fetchall()])
		assert s == set(['col_a', 'col_c', 'col_d', 'col_e'])

		# Make sure the number of rows has not changed:
		cursor.execute("SELECT COUNT(*) FROM tbl;")
		assert cursor.fetchone()[0] == 1000

		# Make sure the foreign_keys setting has not changed:
		cursor.execute("PRAGMA foreign_keys;")
		assert cursor.fetchone()[0] == foreign_keys

		# Wrong table or column name should give a ValueError:
		with pytest.raises(ValueError):
			u.sqlite_drop_column(conn, 'tbl_wrong', 'col_e')

		with pytest.raises(ValueError):
			u.sqlite_drop_column(conn, 'tbl', 'col_wrong')

		# Attempting to drop a column associated with an index should
		# cause an Exception:
		with pytest.raises(RuntimeError):
			u.sqlite_drop_column(conn, 'tbl', 'col_c')

		with pytest.raises(RuntimeError):
			u.sqlite_drop_column(conn, 'tbl', 'col_e')

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
				with pytest.raises(MaxRetryError): # requests.exceptions.HTTPError
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
