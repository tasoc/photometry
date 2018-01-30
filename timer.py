#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Timer script for the various photometry methods.

@author: Jonas Svenstrup Hansen
"""

import subprocess
import time

print("Running aperture...")
start_time = time.time()
subprocess.call(["python","run_tessphot.py","5","-m","aperture"])
print("--- %s seconds ---" % (time.time() - start_time))

print("Running linpsf...")
start_time = time.time()
subprocess.call(["python","run_tessphot.py","5","-m","linpsf"])
print("--- %s seconds ---" % (time.time() - start_time))

print("Running psf...")
start_time = time.time()
subprocess.call(["python","run_tessphot.py","5","-m","psf"])
print("--- %s seconds ---" % (time.time() - start_time))