#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of photometry.utilities.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
import numpy as np
import sqlite3
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
def test_find_ffi_files(SHARED_INPUT_DIR):

	files = u.find_ffi_files(SHARED_INPUT_DIR)
	assert(len(files) == 8)

	files = u.find_ffi_files(SHARED_INPUT_DIR, camera=1)
	assert(len(files) == 4)

	files = u.find_ffi_files(SHARED_INPUT_DIR, camera=3)
	assert(len(files) == 4)

#--------------------------------------------------------------------------------------------------
def test_find_tpf_files(SHARED_INPUT_DIR):

	files = u.find_tpf_files(SHARED_INPUT_DIR)
	assert(len(files) == 2)

	files = u.find_tpf_files(SHARED_INPUT_DIR, starid=267211065)
	assert(len(files) == 1)

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

#--------------------------------------------------------------------------------------------------
def test_find_catalog_files(SHARED_INPUT_DIR):

	files = u.find_catalog_files(SHARED_INPUT_DIR)
	assert(len(files) == 2)

	files = u.find_catalog_files(SHARED_INPUT_DIR, sector=1)
	assert(len(files) == 2)

	files = u.find_catalog_files(SHARED_INPUT_DIR, camera=1)
	assert(len(files) == 1)

	files = u.find_catalog_files(SHARED_INPUT_DIR, sector=1, camera=3, ccd=2)
	assert(len(files) == 1)

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
		with pytest.raises(Exception):
			u.sqlite_drop_column(conn, 'tbl', 'col_c')

		with pytest.raises(Exception):
			u.sqlite_drop_column(conn, 'tbl', 'col_e')

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
