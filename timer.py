#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Timer script for the various photometry methods.

@author: Jonas Svenstrup Hansen
"""

import subprocess
import time
import numpy as np

# Number of runs of each method:
Nruns = 10
times = np.empty(Nruns)


print("Running aperture...")
for i in range(Nruns):
	start_time = time.time()
	subprocess.call(["python","run_tessphot.py","5","-m","aperture"])
	stop_time = time.time() - start_time
	times[i] = stop_time
print("--- %s seconds, uncertainty = %s ---" % (np.mean(times), np.std(times)/np.sqrt(Nruns)))

print("Running linpsf...")
for i in range(Nruns):
	start_time = time.time()
	subprocess.call(["python","run_tessphot.py","5","-m","linpsf"])
	stop_time = time.time() - start_time
	times[i] = stop_time
print("--- %s seconds, uncertainty = %s ---" % (np.mean(times), np.std(times)/np.sqrt(Nruns)))

print("Running psf...")
for i in range(Nruns):
	start_time = time.time()
	subprocess.call(["python","run_tessphot.py","5","-m","psf"])
	stop_time = time.time() - start_time
	times[i] = stop_time
print("--- %s seconds, uncertainty = %s ---" % (np.mean(times), np.std(times)/np.sqrt(Nruns)))

