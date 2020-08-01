#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scheduler using MPI for running the TESS photometry
pipeline on a large scale multi-core computer.

The setup uses the task-pull paradigm for high-throughput computing
using ``mpi4py``. Task pull is an efficient way to perform a large number of
independent tasks when there are more tasks than processors, especially
when the run times vary for each task.

The basic example was inspired by
https://github.com/jbornschein/mpi4py-examples/blob/master/09-task-pull.py

Example
-------
To run the program using four processes (one master and three workers) you can
execute the following command:

>>> mpiexec -n 4 python run_tessphot_mpi.py

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from mpi4py import MPI
import argparse
import logging
import traceback
import os
import enum
from timeit import default_timer
import photometry

#--------------------------------------------------------------------------------------------------
def main():
	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Run TESS Photometry in parallel using MPI.')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('-o', '--overwrite', help='Overwrite existing results.', action='store_true')
	parser.add_argument('-p', '--plot', help='Save plots when running.', action='store_true')
	parser.add_argument('--camera', type=int, choices=(1,2,3,4), default=None, help='TESS Camera. Default is to run all cameras.')
	parser.add_argument('--ccd', type=int, choices=(1,2,3,4), default=None, help='TESS CCD. Default is to run all CCDs.')
	parser.add_argument('--datasource', type=str, choices=('ffi','tpf'), default=None, help='Data source or cadence. Default is to run all.')
	parser.add_argument('--version', type=int, help='Data release number to store in output files.', nargs='?', default=None)
	parser.add_argument('input_folder', type=str, help='Input directory. This directory should contain a TODO-file.', nargs='?', default=None)
	args = parser.parse_args()

	# Set logging level:
	logging_level = logging.INFO
	if args.quiet:
		logging_level = logging.WARNING
	elif args.debug:
		logging_level = logging.DEBUG

	# Get paths to input and output files from environment variables:
	input_folder = args.input_folder
	if input_folder is None:
		input_folder = os.environ.get('TESSPHOT_INPUT', os.path.join(os.path.dirname(__file__), 'tests', 'input'))
	if not input_folder:
		parser.error("Please specify an INPUT_FOLDER.")
	output_folder = os.environ.get('TESSPHOT_OUTPUT', os.path.join(input_folder, 'lightcurves'))

	# Define MPI message tags
	tags = enum.IntEnum('tags', ('READY', 'DONE', 'EXIT', 'START'))

	# Initializations and preliminaries
	comm = MPI.COMM_WORLD   # get MPI communicator object
	size = comm.size        # total number of processes
	rank = comm.rank        # rank of this process
	status = MPI.Status()   # get MPI status object

	if rank == 0:
		# Master process executes code below
		try:
			# Constraints on which targets to process:
			constraints = {
				'camera': args.camera,
				'ccd': args.ccd,
				'datasource': args.datasource
			}

			with photometry.TaskManager(input_folder, cleanup=True, overwrite=args.overwrite,
				cleanup_constraints=constraints, summary=os.path.join(output_folder, 'summary.json')) as tm:

				# Set level of TaskManager logger:
				tm.logger.setLevel(logging_level)

				# Get list of tasks:
				numtasks = tm.get_number_tasks(**constraints)
				tm.logger.info("%d tasks to be run", numtasks)

				# Start the master loop that will assign tasks
				# to the workers:
				num_workers = size - 1
				closed_workers = 0
				tm.logger.info("Master starting with %d workers", num_workers)
				while closed_workers < num_workers:
					# Ask workers for information:
					data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
					source = status.Get_source()
					tag = status.Get_tag()

					if tag == tags.DONE:
						# The worker is done with a task
						tm.logger.debug("Got data from worker %d: %s", source, data)
						tm.save_result(data)

					if tag in (tags.DONE, tags.READY):
						# Worker is ready, so send it a task
						task = tm.get_task(**constraints)
						if task:
							task_index = task['priority']
							tm.start_task(task_index)
							comm.send(task, dest=source, tag=tags.START)
							tm.logger.debug("Sending task %d to worker %d", task_index, source)
						else:
							comm.send(None, dest=source, tag=tags.EXIT)

					elif tag == tags.EXIT:
						# The worker has exited
						tm.logger.info("Worker %d exited.", source)
						closed_workers += 1

					else: # pragma: no cover
						# This should never happen, but just to
						# make sure we don't run into an infinite loop:
						raise Exception("Master received an unknown tag: '{0}'".format(tag))

				tm.logger.info("Master finishing")

		except: # noqa: E722
			# If something fails in the master
			print(traceback.format_exc().strip())
			comm.Abort(1)

	else:
		# Worker processes execute code below
		# Configure logging within photometry:
		formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
		console = logging.StreamHandler()
		console.setFormatter(formatter)
		logger = logging.getLogger('photometry')
		logger.addHandler(console)
		logger.setLevel(logging.WARNING)

		try:
			# Send signal that we are ready for task:
			comm.send(None, dest=0, tag=tags.READY)

			while True:
				# Receive a task from the master:
				tic = default_timer()
				task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
				tag = status.Get_tag()
				toc = default_timer()

				if tag == tags.START:
					# Do the work here
					result = task.copy()
					del task['priority'], task['tmag']

					t1 = default_timer()
					pho = photometry.tessphot(input_folder=input_folder, output_folder=output_folder, plot=args.plot, version=args.version, **task)
					t2 = default_timer()

					# Construct result message:
					result.update({
						'status': pho.status,
						'method_used': pho.method,
						'time': t2 - t1,
						'worker_wait_time': toc - tic,
						'details': pho._details
					})

					# Send the result back to the master:
					comm.send(result, dest=0, tag=tags.DONE)

				elif tag == tags.EXIT:
					# We were told to EXIT, so lets do that
					break

				else: # pragma: no cover
					# This should never happen, but just to
					# make sure we don't run into an infinite loop:
					raise Exception("Worker received an unknown tag: '{0}'".format(tag))

		except: # noqa: E722, pragma: no cover
			logger.exception("Something failed in worker")

		finally:
			comm.send(None, dest=0, tag=tags.EXIT)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	main()
