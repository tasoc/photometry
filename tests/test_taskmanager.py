#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, print_function, with_statement, absolute_import
import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from photometry import TaskManager

def test_taskmanager():
	"""Test of background estimator"""

	# Load the first image in the input directory:
	INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')

	# Find the shape of the original image:
	with TaskManager(INPUT_DIR) as tm:
		# Get the first task in the TODO file:
		task = tm.get_task()
		print(task)
		
		# Check that it contains what we know it should:
		# The first priority in the TODO file is the following:
		assert(task['priority'] == 1)
		assert(task['starid'] == 284853659)
		assert(task['camera'] == 2)
		assert(task['ccd'] == 2)
	

if __name__ == '__main__':
	test_taskmanager()
