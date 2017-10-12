#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:54:10 2017

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, print_function, with_statement, absolute_import
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from photometry import BasePhotometry

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')
DUMMY_TARGET = 143159

def test_stamp():
	with BasePhotometry(DUMMY_TARGET, INPUT_DIR) as pho:

		pho._stamp = (0, 10, 0, 20)
		pho.set_stamp()

		cols, rows = pho.get_pixel_grid()
		print('Rows:')
		print(rows)
		print(rows.shape)
		print('Cols:')
		print(cols)
		print(cols.shape)

		assert(rows.shape == (10, 20))
		assert(cols.shape == (10, 20))
		assert(rows[0,0] == 1)
		assert(cols[0,0] == 1)
		assert(rows[-1,0] == 10)
		assert(cols[-1,0] == 1)
		assert(rows[-1,-1] == 10)
		assert(cols[-1,-1] == 20)

		pho.resize_stamp(up=12)
		cols, rows = pho.get_pixel_grid()
		print('Rows:')
		print(rows)
		print(rows.shape)
		print('Cols:')
		print(cols)
		print(cols.shape)
		assert(rows.shape == (22, 20))
		assert(cols.shape == (22, 20))

def test_images():
	with BasePhotometry(DUMMY_TARGET, INPUT_DIR) as pho:

		pho._stamp = (0, 10, 0, 20)
		pho.set_stamp()

		for img in pho.images:
			assert(img.shape == (10, 20))

def test_backgrounds():
	with BasePhotometry(DUMMY_TARGET, INPUT_DIR) as pho:

		pho._stamp = (0, 10, 0, 20)
		pho.set_stamp()

		for img in pho.backgrounds:
			assert(img.shape == (10, 20))

def test_catalog():
	with BasePhotometry(DUMMY_TARGET, INPUT_DIR) as pho:
		print(pho.catalog)
		assert(DUMMY_TARGET in pho.catalog['starid'])

if __name__ == '__main__':
	test_stamp()
	test_images()
	test_backgrounds()
	test_catalog()
