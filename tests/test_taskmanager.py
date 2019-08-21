#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from photometry import TaskManager, STATUS

def test_taskmanager():
	"""Test of background estimator"""

	# Load the first image in the input directory:
	INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')

	# Find the shape of the original image:
	with TaskManager(INPUT_DIR, overwrite=True) as tm:
		# Get the number of tasks:
		numtasks = tm.get_number_tasks()
		print(numtasks)
		assert(numtasks == 168642)

		# Get the first task in the TODO file:
		task1 = tm.get_task()
		print(task1)

		# Check that it contains what we know it should:
		# The first priority in the TODO file is the following:
		assert(task1['priority'] == 1)
		assert(task1['starid'] == 267211065)
		assert(task1['camera'] == 3)
		assert(task1['ccd'] == 2)
		assert(task1['datasource'] == 'tpf')
		assert(task1['sector'] == 1)

		# Start task with priority=1:
		tm.start_task(1)

		# Get the next task, which should be the one with priority=2:
		task2 = tm.get_task()
		print(task2)

		assert(task2['priority'] == 2)
		assert(task2['starid'] == 267211065)
		assert(task2['camera'] == 3)
		assert(task2['ccd'] == 2)
		assert(task2['datasource'] == 'ffi')
		assert(task2['sector'] == 1)

		# Check that the status did actually change in the todolist:
		tm.cursor.execute("SELECT status FROM todolist WHERE priority=1;")
		task1_status = tm.cursor.fetchone()['status']
		print(task1_status)

		assert(task1_status == STATUS.STARTED.value)

if __name__ == '__main__':
	test_taskmanager()
