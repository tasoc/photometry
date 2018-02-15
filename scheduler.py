#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic scheduler for running photometry on Kepler data of NGC 6819, Q11.

@author: Jonas Svenstrup Hansen, jonas.svenstrup@gmail.com
"""

import os
import sqlite3
import numpy as np
import multiprocessing
import argparse
import logging


def load_starids():
	""" 
	Load todo list starids and tmags for use with run_tessphot.py.
	
	Returns:
		(list of strings): Priority-sorted list with starids as strings.
	"""
	# Get input folder from environment variable and load todo list:
	input_folder = os.environ['TESSPHOT_INPUT']
	todo_file = os.path.join(input_folder, 'todo.sqlite')
	
	# With sqlite, load and save priority and starid to list of tuples:
	conn = sqlite3.connect(todo_file)
	cursor = conn.cursor()
	cursor.execute("""SELECT priority, starid FROM todolist""")
	stars = cursor.fetchall()
	
	# Sort stars by priority:
	stars.sort(key=lambda star: star[0])
	
	# Get list of starids as strings:
	return [np.str(star[1]) for star in stars]


def loop_through_starids(starids, method='aperture'):
	""" 
	Call run_tessphot in a multiprocessing loop inspired by prepare_photometry.
	
	Parameters:
		starids (list of strings): Priority-sorted list with starids as strings.
		method (string): Photometry method. Can be either ``'aperture'`` 
		(default), ``'psf'`` or ``'linpsf'``.
	"""
	
	logger = logging.getLogger(__name__)
	
	# Get number of available threads on CPU. Use 1 if not accessible:
	threads = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))

	# Set up multiprocessing if multiple threads are available:
	if threads > 1:
		pool = multiprocessing.Pool(threads)
		m = pool.imap
	else:
		m = map

	# Set up run_tessphot commands:
	base_cmd = "python run_tessphot.py "
	args = " --method='"+method+"' --quiet"
	cmds = [base_cmd+starid+args for starid in starids]

	# Call run_tessphot using multiprocess mapping for each starid:
	for cmd in m(run_cmd, cmds):
		logger.info(cmd)

	if threads > 1:
		# Close multithreading:
		pool.close()
		pool.join()


def run_cmd(cmd):
	"""
	Run command. Designed to be used in a multiprocessing call.
	
	Parameters:
		cmd (string): Command to run.
	
	Returns:
		cmd (string): Command that has been run.
	"""
	os.system(cmd)
	return cmd



if __name__ == '__main__':
	
	# Parse arguments:
	parser = argparse.ArgumentParser(description='Run TESS Photometry pipeline on every star in the todo list.')
	parser.add_argument('-m', '--method', help='Photometric method to use.', default=None, choices=('aperture', 'psf', 'linpsf'))
	args = parser.parse_args()
	method = args.method

	# Setup logging:
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

	logger = logging.getLogger(__name__)
	logger.setLevel(logging.INFO)
	if not logger.hasHandlers():
		console = logging.StreamHandler()
		console.setFormatter(formatter)
		logger.addHandler(console)

	# Load starids to priority sorted list of strings:
	starids = load_starids()
	
	# Call run_tessphot on each starid:
	loop_through_starids(starids, method=method)