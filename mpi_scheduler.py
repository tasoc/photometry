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

Example:

>> mpiexec -n 4 python mpi_scheduler.py

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import with_statement, print_function
from mpi4py import MPI
import sys
import os
import enum
import logging

# Get paths to input and output files from environment variables:
input_folder = os.environ['TESSPHOT_INPUT']
output_folder = os.environ['TESSPHOT_OUTPUT']
todo_file = os.path.join(input_folder, 'todo.sqlite')

#------------------------------------------------------------------------------
class TaskManager(object):
	def __init__(self, todo_file):
		self.conn = sqlite3.connect(todo_file)
		self.conn.row_factory = sqlite3.Row
		self.cursor = self.conn.cursor()

		# Reset the status of everything for a new run:
		# TODO: This should obviously be removed once we start running for real
		self.cursor.execute("UPDATE todolist SET status=NULL,elaptime=NULL;")
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

	def get_tasks(self):
		self.cursor.execute("SELECT starid,method FROM todolist WHERE status IS NULL ORDER BY priority;")
		tasks = self.cursor.fetchall()
		tasks = [dict(t) for t in tasks]
		return tasks

	def save_result(self, result):
		self.cursor.execute("UPDATE todolist SET status=?,elaptime=? WHERE starid=?;", (result['status'].value, result['time'], result['starid']))
		self.conn.commit()


#------------------------------------------------------------------------------

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
		tasks = tm.get_tasks()
		tm.logger.info("%d tasks to be run", len(tasks))

		# Start the master loop that will assing tasks
		# to the workers:
		task_index = 0
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
				if task_index < len(tasks):
					comm.send(tasks[task_index], dest=source, tag=tags.START)
					tm.logger.info("Sending task %d to worker %d", task_index, source)
					task_index += 1
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
				# make sure we dont run into an infinite loop:
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
			task['input_folder'] = input_folder
			task['output_folder'] = output_folder

			t1 = default_timer()
			pho = tessphot(**task)
			t2 = default_timer()

			# Construct result message:
			result = {
				'starid': pho.starid,
				'status': pho.status,
				'time': t2 - t1
			}

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
