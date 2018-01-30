#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Timer script for the various photometry methods.

@author: Jonas Svenstrup Hansen
"""

import os
import timeit

# Get input and output folder from enviroment variables:
input_folder = os.environ.get('TESSPHOT_INPUT', os.path.abspath(os.path.join(os.path.dirname(__file__), 'tests', 'input')))
output_folder = os.environ.get('TESSPHOT_OUTPUT', os.path.abspath('.'))

# Number of runs for each method:
Nruns = 1
print("Number of runs: %s" % Nruns)

# Time the methods:
methods = ('aperture', 'linpsf', 'psf')
for method in methods:
	setup = "from photometry import tessphot"
	code = "tessphot(5, '"+method+"', input_folder='" + \
			input_folder + "', output_folder='" + \
			output_folder + "')"
	extime = timeit.timeit(stmt = code, setup = setup, number=Nruns)
	out = float(extime)/float(Nruns)
	print("Execution time of "+method+": %s seconds" % out)

#print("Running aperture...")
#for i in range(Nruns):
#	start_time = time.time()
#	subprocess.call(["python","run_tessphot.py","5","-m","aperture"])
#	stop_time = time.time() - start_time
#	times[i] = stop_time
#print("--- %s seconds, uncertainty = %s ---" % (np.mean(times), np.std(times)/np.sqrt(Nruns)))
#
#print("Running linpsf...")
#for i in range(Nruns):
#	start_time = time.time()
#	subprocess.call(["python","run_tessphot.py","5","-m","linpsf"])
#	stop_time = time.time() - start_time
#	times[i] = stop_time
#print("--- %s seconds, uncertainty = %s ---" % (np.mean(times), np.std(times)/np.sqrt(Nruns)))
#
#print("Running psf...")
#for i in range(Nruns):
#	start_time = time.time()
#	subprocess.call(["python","run_tessphot.py","5","-m","psf"])
#	stop_time = time.time() - start_time
#	times[i] = stop_time
#print("--- %s seconds, uncertainty = %s ---" % (np.mean(times), np.std(times)/np.sqrt(Nruns)))
#
