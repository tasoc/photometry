#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scheduler using MPI for running the TESS photometry
pipeline on a large scale multicore computer.

The setup uses the task-pull paradigm for high-throughput computing
using ``mpi4py``. Task pull is an efficient way to perform a large number of
independent tasks when there are more tasks than processors, especially
when the run times vary for each task.

The basic example was inspired by
http://math.acadiau.ca/ACMMaC/Rmpi/index.html

Example
-------

>> mpiexec -n 4 python mpi_scheduler.py

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import with_statement, print_function
from mpi4py import MPI
import sys
import os
import enum
import logging

#------------------------------------------------------------------------------
class TaskManager(object):
	def __init__(self, todo_file):
		self.conn = sqlite3.connect(todo_file)
		self.conn.row_factory = sqlite3.Row
		self.cursor = self.conn.cursor()

		# Reset the status of everything for a new run:
		# TODO: This should obviously be removed once we start running for real
		self.cursor.execute("UPDATE todolist SET status=NULL,elaptime=NULL;")
		self.cursor.execute("DROP TABLE IF EXISTS diagnostics;")
		self.conn.commit()

		self.cursor.execute("""CREATE TABLE IF NOT EXISTS diagnostics (
			priority INT PRIMARY KEY NOT NULL,
			starid BIGINT NOT NULL,
			mean_flux DOUBLE PRECISION,
			mask_size INT,
			pos_row REAL,
			pos_column REAL,
			contamination REAL,
			stamp_resizes INT,
			errors TEXT
		)""")
		self.conn.commit()

		# Setup logging:
		formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
		console = logging.StreamHandler()
		console.setFormatter(formatter)
		self.logger = logging.getLogger(__name__)
		self.logger.addHandler(console)
		self.logger.setLevel(logging.INFO)

	def close(self):
		self.cursor.close()
		self.conn.close()

	def __exit__(self, *args):
		self.close()

	def __enter__(self):
		return self

	def get_number_tasks(self):
		self.cursor.execute("SELECT COUNT(*) AS num FROM todolist WHERE status IS NULL;")
		num = int(self.cursor.fetchone()['num'])
		return num

	def get_task(self):
		self.cursor.execute("SELECT priority,starid,method FROM todolist WHERE status IS NULL ORDER BY priority LIMIT 1;")
		task = self.cursor.fetchone()
		if task: return dict(task)
		return None

	def save_result(self, result):
		self.cursor.execute("UPDATE todolist SET status=?,elaptime=? WHERE priority=?;", (result['status'].value, result['time'], result['priority']))

		if 'skip_targets' in result['details'] and len(result['details']['skip_targets']) > 0:
			# Create unique list of starids to be masked as skipped:
			skip_starids = [str(starid) for starid in set(result['details']['skip_targets'])]
			skip_starids = ','.join(skip_starids)
			# Mark them as SKIPPED in the database:
			self.cursor.execute("UPDATE todolist SET status=5 WHERE status IS NULL AND starid IN (" + skip_starids + ");")

		# Save diagnostics:
		error_msg = result['details'].get('errors', None)
		if error_msg: error_msg = '\n'.join(error_msg)
		self.cursor.execute("INSERT INTO diagnostics (priority, starid, pos_column, pos_row, mean_flux, mask_size, contamination, stamp_resizes, errors) VALUES (?,?,?,?,?,?,?,?,?);", (
			result['priority'],
			result['starid'],
			result['details'].get('pos_centroid', (None, None))[0],
			result['details'].get('pos_centroid', (None, None))[1],
			result['details'].get('mean_flux', None),
			result['details'].get('mask_size', None),
			result['details'].get('contamination', None),
			result['details'].get('stamp_resizes', 0),
			error_msg
		))
		self.conn.commit()

	def start_task(self, taskid):
		self.cursor.execute("UPDATE todolist SET status=6 WHERE priority=?;", (taskid,))
		self.conn.commit()


#------------------------------------------------------------------------------
if __name__ == '__main__':

	# Get paths to input and output files from environment variables:
	input_folder = os.environ['TESSPHOT_INPUT']
	output_folder = os.environ['TESSPHOT_OUTPUT']
	todo_file = os.path.join(input_folder, 'todo.sqlite')

	# Define MPI message tags
	tags = enum.IntEnum('tags', ('READY', 'DONE', 'EXIT', 'START'))

	# Initializations and preliminaries
	comm = MPI.COMM_WORLD   # get MPI communicator object
	size = comm.size        # total number of processes
	rank = comm.rank        # rank of this process
	status = MPI.Status()   # get MPI status object

	if rank == 0:
		# Master process executes code below
		import sqlite3

		with TaskManager(todo_file) as tm:
			# Get list of tasks:
			numtasks = tm.get_number_tasks()
			tm.logger.info("%d tasks to be run", numtasks)

			# Start the master loop that will assing tasks
			# to the workers:
			num_workers = size - 1
			closed_workers = 0
			tm.logger.info("Master starting with %d workers", num_workers)
			while closed_workers < num_workers:
				# Ask workers for information:
				data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
				source = status.Get_source()
				tag = status.Get_tag()

				if tag == tags.READY:
					# Worker is ready, so send it a task
					task = tm.get_task()

					if task:
						task_index = task['priority']
						tm.start_task(task_index)
						comm.send(task, dest=source, tag=tags.START)
						tm.logger.info("Sending task %d to worker %d", task_index, source)
					else:
						comm.send(None, dest=source, tag=tags.EXIT)

				elif tag == tags.DONE:
					# The worker is done with a task
					tm.logger.info("Got data from worker %d: %s", source, data)
					tm.save_result(data)

				elif tag == tags.EXIT:
					# The worker has exited
					tm.logger.info("Worker %d exited.", source)
					closed_workers += 1

				else:
					# This should never happen, but just to
					# make sure we don't run into an infinite loop:
					raise Exception("Master recieved an unknown tag: '{0}'".format(tag))

		tm.logger.info("Master finishing")

	else:
		# Worker processes execute code below
		from photometry import tessphot
		from timeit import default_timer

		while True:
			# Send signal that we are ready for task,
			# and recieve a task from the master:
			comm.send(None, dest=0, tag=tags.READY)
			task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
			tag = status.Get_tag()

			if tag == tags.START:
				# Do the work here
				result = task.copy()
				del task['priority']

				t1 = default_timer()
				pho = tessphot(input_folder=input_folder, output_folder=output_folder, **task)
				t2 = default_timer()

				# Construct result message:
				result.update({
					'status': pho.status,
					'time': t2 - t1,
					'details': pho._details
				})

				# Send the result back to the master:
				comm.send(result, dest=0, tag=tags.DONE)

			elif tag == tags.EXIT:
				# We were told to EXIT, so lets do that
				break

			else:
				# This should never happen, but just to
				# make sure we dont run into an infinite loop:
				raise Exception("Worker recieved an unknown tag: '{0}'".format(tag))

		comm.send(None, dest=0, tag=tags.EXIT)
