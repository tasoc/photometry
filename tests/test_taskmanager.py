#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of TaskManager.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import sys
import os.path
import json
import tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from photometry import TaskManager, STATUS

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')

#--------------------------------------------------------------------------------------------------
def test_taskmanager(PRIVATE_TODO_FILE):
	"""Test of TaskManager"""
	with TaskManager(PRIVATE_TODO_FILE, overwrite=True) as tm:
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

#--------------------------------------------------------------------------------------------------
def test_taskmanager_invalid():
	with pytest.raises(FileNotFoundError):
		TaskManager(os.path.join(INPUT_DIR, 'does-not-exists'))

#--------------------------------------------------------------------------------------------------
def test_get_tasks(PRIVATE_TODO_FILE):
	with TaskManager(PRIVATE_TODO_FILE, overwrite=True) as tm:
		task = tm.get_task(starid=267211065)
		assert task['priority'] == 1

		# Call with non-existing starid:
		task = tm.get_task(starid=-1234567890)
		assert task is None

#--------------------------------------------------------------------------------------------------
def test_taskmanager_summary(PRIVATE_TODO_FILE):
	with tempfile.TemporaryDirectory() as tmpdir:
		summary_file = os.path.join(tmpdir, 'summary.json')
		with TaskManager(PRIVATE_TODO_FILE, overwrite=True, summary=summary_file) as tm:
			# Load the summary file:
			with open(summary_file, 'r') as fid:
				j = json.load(fid)

			# Everytning should be really empty:
			print(j)
			assert j['numtasks'] == 168642
			assert j['UNKNOWN'] == 0
			assert j['OK'] == 0
			assert j['ERROR'] == 0
			assert j['WARNING'] == 0
			assert j['ABORT'] == 0
			assert j['STARTED'] == 0
			assert j['SKIPPED'] == 0
			assert j['tasks_run'] == 0
			assert j['slurm_jobid'] is None
			assert j['last_error'] is None
			assert j['mean_elaptime'] is None

			# Start task with priority=1:
			task = tm.get_random_task()
			print(task)
			tm.start_task(task['priority'])
			tm.write_summary()

			with open(summary_file, 'r') as fid:
				j = json.load(fid)

			print(j)
			assert j['numtasks'] == 168642
			assert j['UNKNOWN'] == 0
			assert j['OK'] == 0
			assert j['ERROR'] == 0
			assert j['WARNING'] == 0
			assert j['ABORT'] == 0
			assert j['STARTED'] == 1
			assert j['SKIPPED'] == 0
			assert j['tasks_run'] == 0
			assert j['slurm_jobid'] is None
			assert j['last_error'] is None
			assert j['mean_elaptime'] is None

			# Make a fake result we can save;
			result = task.copy()
			result['status'] = STATUS.OK
			result['time'] = 3.14
			result['method_used'] = 'aperture'

			# Save the result:
			tm.save_result(result)
			tm.write_summary()

			# Check that the diagnostics were updated:
			tm.cursor.execute("SELECT * FROM diagnostics WHERE priority=?;", [task['priority']])
			row = tm.cursor.fetchone()
			print(dict(row))
			assert row['priority'] == task['priority']
			assert row['starid'] == task['starid']
			assert row['elaptime'] == 3.14
			assert row['method_used'] == 'aperture'

			# Load the summary file after "running the task":
			with open(summary_file, 'r') as fid:
				j = json.load(fid)

			print(j)
			assert j['numtasks'] == 168642
			assert j['UNKNOWN'] == 0
			assert j['OK'] == 1
			assert j['ERROR'] == 0
			assert j['WARNING'] == 0
			assert j['ABORT'] == 0
			assert j['STARTED'] == 0
			assert j['SKIPPED'] == 0
			assert j['tasks_run'] == 1
			assert j['slurm_jobid'] is None
			assert j['last_error'] is None
			assert j['mean_elaptime'] == 3.14

			task = tm.get_random_task()
			tm.start_task(task['priority'])

			# Make a fake result we can save;
			result = task.copy()
			result['status'] = STATUS.ERROR
			result['time'] = 6.14
			result['method_used'] = 'halo'
			result['details'] = {
				'errors': ['dummy error 1', 'dummy error 2']
			}

			# Save the result:
			tm.save_result(result)
			tm.write_summary()

			# Check that the diagnostics were updated:
			tm.cursor.execute("SELECT * FROM diagnostics WHERE priority=?;", [task['priority']])
			row = tm.cursor.fetchone()
			print(dict(row))
			assert row['priority'] == task['priority']
			assert row['starid'] == task['starid']
			assert row['elaptime'] == 6.14
			assert row['method_used'] == 'halo'
			assert row['errors'] == "dummy error 1\ndummy error 2"

			# Load the summary file after "running the task":
			with open(summary_file, 'r') as fid:
				j = json.load(fid)

			print(j)
			assert j['numtasks'] == 168642
			assert j['UNKNOWN'] == 0
			assert j['OK'] == 1
			assert j['ERROR'] == 1
			assert j['WARNING'] == 0
			assert j['ABORT'] == 0
			assert j['STARTED'] == 0
			assert j['SKIPPED'] == 0
			assert j['tasks_run'] == 2
			assert j['slurm_jobid'] is None
			assert j['last_error'] == "dummy error 1\ndummy error 2"
			assert j['mean_elaptime'] == 3.44

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
