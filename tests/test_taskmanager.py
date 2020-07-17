#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of TaskManager.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import pytest
import os.path
import json
import tempfile
import conftest # noqa: F401
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
def test_taskmanager_constraints(PRIVATE_TODO_FILE):

	constraints = {'datasource': 'tpf', 'priority': 1}
	with TaskManager(PRIVATE_TODO_FILE, overwrite=True, cleanup_constraints=constraints) as tm:
		task = tm.get_task(**constraints)
		numtasks = tm.get_number_tasks(**constraints)
		print(task)
		assert task['starid'] == 267211065, "Task1 should be None"
		assert numtasks == 1, "Task1 search should give no results"

	constraints = {'datasource': 'tpf', 'priority': 1, 'camera': None}
	with TaskManager(PRIVATE_TODO_FILE, overwrite=True, cleanup_constraints=constraints) as tm:
		task2 = tm.get_task(**constraints)
		numtasks2 = tm.get_number_tasks(**constraints)
		print(task2)
		assert task2 == task, "Tasks should be identical"
		assert numtasks2 == 1, "Task2 search should give no results"

	constraints = {'datasource': 'ffi', 'priority': 2}
	with TaskManager(PRIVATE_TODO_FILE, overwrite=True, cleanup_constraints=constraints) as tm:
		task = tm.get_task(**constraints)
		numtasks = tm.get_number_tasks(**constraints)
		print(task)
		assert task['priority'] == 2, "Task2 should be #2"
		assert task['datasource'] == 'ffi'
		assert task['camera'] == 3
		assert task['ccd'] == 2
		assert numtasks == 1, "Priority search should give one results"

	constraints = {'datasource': 'ffi', 'priority': 2, 'camera': 3, 'ccd': 2}
	with TaskManager(PRIVATE_TODO_FILE, overwrite=True, cleanup_constraints=constraints) as tm:
		task3 = tm.get_task(**constraints)
		numtasks3 = tm.get_number_tasks(**constraints)
		print(task3)
		assert task3 == task, "Tasks should be identical"
		assert numtasks3 == 1, "Task3 search should give one results"

	constraints = ['priority=17']
	with TaskManager(PRIVATE_TODO_FILE, cleanup_constraints=constraints) as tm:
		task4 = tm.get_task(priority=17)
		numtasks4 = tm.get_number_tasks(priority=17)
		print(task4)
		assert task4['priority'] == 17, "Task4 should be #17"
		assert numtasks4 == 1, "Priority search should give one results"

	constraints = {'starid': 267211065}
	with TaskManager(PRIVATE_TODO_FILE, cleanup_constraints=constraints) as tm:
		numtasks5 = tm.get_number_tasks(**constraints)
		assert numtasks5 == 2
		task5 = tm.get_task(**constraints)
		assert task5['priority'] == 1

#--------------------------------------------------------------------------------------------------
def test_taskmanager_constraints_invalid(PRIVATE_TODO_FILE):
	with pytest.raises(ValueError) as e:
		with TaskManager(PRIVATE_TODO_FILE, cleanup_constraints='invalid'):
			pass
	assert str(e.value) == 'cleanup_constraints should be dict or list'

#--------------------------------------------------------------------------------------------------
def test_taskmanager_no_more_tasks(PRIVATE_TODO_FILE):
	with TaskManager(PRIVATE_TODO_FILE) as tm:
		# Set all the tasks as completed:
		tm.cursor.execute("UPDATE todolist SET status=1;")
		tm.conn.commit()

		# When we now ask for a new task, there shouldn't be any:
		assert tm.get_task() is None
		assert tm.get_random_task() is None
		assert tm.get_number_tasks() == 0

#--------------------------------------------------------------------------------------------------
def test_taskmanager_summary(PRIVATE_TODO_FILE):
	with tempfile.TemporaryDirectory() as tmpdir:
		summary_file = os.path.join(tmpdir, 'summary.json')
		with TaskManager(PRIVATE_TODO_FILE, overwrite=True, summary=summary_file, summary_interval=2) as tm:
			# Load the summary file:
			with open(summary_file, 'r') as fid:
				j = json.load(fid)

			assert tm.summary_counter == 0  # Counter should start at zero

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
			result['worker_wait_time'] = 1.0
			result['method_used'] = 'aperture'

			# Save the result:
			tm.save_result(result)
			assert tm.summary_counter == 1 # We saved once, so counter should have gone up one
			tm.write_summary()

			# Check that the diagnostics were updated:
			tm.cursor.execute("SELECT * FROM diagnostics WHERE priority=?;", [task['priority']])
			row = tm.cursor.fetchone()
			print(dict(row))
			assert row['priority'] == task['priority']
			assert 'starid' not in row # It should no longer be there (after version 4.6)
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
			assert j['mean_worker_waittime'] == 1.0

			task = tm.get_random_task()
			tm.start_task(task['priority'])

			# Make a fake result we can save;
			result = task.copy()
			result['status'] = STATUS.ERROR
			result['time'] = 6.14
			result['worker_wait_time'] = 2.0
			result['method_used'] = 'halo'
			result['details'] = {
				'errors': ['dummy error 1', 'dummy error 2']
			}

			# Save the result:
			tm.save_result(result)
			assert tm.summary_counter == 0 # We saved again, so summary_counter should be zero
			tm.write_summary()

			# Check that the diagnostics were updated:
			tm.cursor.execute("SELECT * FROM diagnostics WHERE priority=?;", [task['priority']])
			row = tm.cursor.fetchone()
			print(dict(row))
			assert row['priority'] == task['priority']
			assert 'starid' not in row # It should no longer be there (after version 4.6)
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
			assert j['mean_worker_waittime'] == 1.1

#--------------------------------------------------------------------------------------------------
def test_taskmanager_skip_targets(PRIVATE_TODO_FILE):
	with TaskManager(PRIVATE_TODO_FILE, overwrite=True, cleanup=True) as tm:

		# Start task with a random task:
		task = tm.get_task(starid=267211065, datasource='ffi') # Tmag = 2.216
		#task = tm.get_task(starid=261522674) # Tmag = 14.574
		print(task)

		# Make a fake result we can save:
		tm.start_task(task['priority'])
		result = task.copy()
		result['status'] = STATUS.OK
		result['time'] = 6.14
		result['worker_wait_time'] = 2.0
		result['method_used'] = 'aperture'
		result['details'] = {
			'skip_targets': [261522674] # Tmag = 14.574
		}

		tm.save_result(result)

		tm.cursor.execute("SELECT * FROM todolist LEFT JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE todolist.priority=?;", [task['priority']])
		row = tm.cursor.fetchall()
		assert len(row) == 1
		row = row[0]
		print(dict(row))
		assert row['priority'] == task['priority']
		assert row['starid'] == task['starid']
		assert row['sector'] == task['sector']
		assert row['camera'] == task['camera']
		assert row['ccd'] == task['ccd']
		assert row['elaptime'] == 6.14
		assert row['method_used'] == 'aperture'
		assert row['status'] == STATUS.OK.value

		tm.cursor.execute("SELECT * FROM todolist LEFT JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE starid=261522674 AND datasource='ffi';")
		row2 = tm.cursor.fetchall()
		assert len(row2) == 1
		row2 = row2[0]
		print(dict(row2))
		assert row2['status'] == STATUS.SKIPPED.value

		# There should now be exactly one entry in the photometry_skipped table:
		tm.cursor.execute("SELECT * FROM photometry_skipped;")
		row = tm.cursor.fetchall()
		assert len(row) == 1
		row = row[0]
		print(dict(row))
		assert row['priority'] == row2['priority']
		assert row['skipped_by'] == task['priority']

		#================================================================
		# RESET THE TODO-FILE:
		tm.cursor.execute("UPDATE todolist SET status=NULL;")
		tm.cursor.execute("DELETE FROM diagnostics;")
		tm.cursor.execute("DELETE FROM photometry_skipped;")
		tm.conn.commit()
		#================================================================

		# Start task with a random task:
		task = tm.get_task(starid=261522674, datasource='ffi') # Tmag = 14.574
		print(task)

		# Make a fake result we can save;
		tm.start_task(task['priority'])
		result = task.copy()
		result['status'] = STATUS.OK
		result['time'] = 6.14
		result['worker_wait_time'] = 2.0
		result['method_used'] = 'aperture'
		result['details'] = {
			'skip_targets': [267211065] # Tmag = 2.216
		}

		tm.save_result(result)

		# This time the processed target (the faint one) should end up marked as SKIPPED:
		tm.cursor.execute("SELECT * FROM todolist LEFT JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE todolist.priority=?;", [task['priority']])
		row = tm.cursor.fetchall()
		assert len(row) == 1
		row = row[0]
		print(dict(row))
		assert row['priority'] == task['priority']
		assert row['starid'] == task['starid']
		assert row['sector'] == task['sector']
		assert row['camera'] == task['camera']
		assert row['ccd'] == task['ccd']
		assert row['elaptime'] == 6.14
		assert row['method_used'] == 'aperture'
		assert row['status'] == STATUS.SKIPPED.value

		# And the bright target should not have STATUS set, so it can be processed later on:
		tm.cursor.execute("SELECT * FROM todolist LEFT JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE starid=267211065 AND datasource='ffi';")
		row3 = tm.cursor.fetchall()
		assert len(row3) == 1
		row3 = row3[0]
		print(dict(row3))
		assert row3['status'] is None

		# There should now be exactly one entry in the photometry_skipped table:
		tm.cursor.execute("SELECT * FROM photometry_skipped;")
		row = tm.cursor.fetchall()
		assert len(row) == 1
		row = row[0]
		print(dict(row))
		assert row['priority'] == task['priority']
		assert row['skipped_by'] == row3['priority']

	assert False

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	pytest.main([__file__])
