#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:54:10 2017

@author: au195407
"""

from __future__ import division, print_function, with_statement, absolute_import
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from photometry import BasePhotometry

def test_stamp():

	with BasePhotometry(143159) as pho:

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

if __name__ == '__main__':
	test_stamp()