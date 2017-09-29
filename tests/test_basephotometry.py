#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:54:10 2017

@author: au195407
"""

from __future__ import division, print_function, with_statement, absolute_import
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from BasePhotometry import BasePhotometry
from AperturePhotometry import AperturePhotometry

def test_something():
	assert(0 == 1)