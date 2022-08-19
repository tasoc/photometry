#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing for TaskMaster memory loading.

Can be used with line_profiler or similar to test execution time.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
import sys
import os
import timeit
import shutil
import tempfile
if sys.path[0] != os.path.abspath('..'):
	sys.path.insert(0, os.path.abspath('..'))
from photometry import TaskManager, STATUS

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':

	original_todo_file = os.path.abspath('../tests/input/todo.sqlite')

	num = 10000
	result = {}

	constraints = {
		'tmag_min': 3.0,
	}

	for in_memory in (True, False):
		with tempfile.TemporaryDirectory() as tmpdir:
			todo_file = os.path.join(tmpdir, 'test.sqlite')
			shutil.copy(original_todo_file, todo_file)

			print('-'*30)
			print(f"Running memory={in_memory}")
			with TaskManager(todo_file, overwrite=True, load_into_memory=in_memory, cleanup_constraints=constraints) as tm:

				t1 = timeit.timeit("tm.get_task()", globals=globals(), number=num)
				print(f"get_task: {t1:.6f} s, {t1/num:.6e} s/it")

				def func():
					task = tm.get_task()
					#print(task)
					if task is None:
						return
					tm.start_task(task['priority'])
					r = task.copy()
					r.update({
						'method_used': 'aperture',
						'status': STATUS.OK,
						'time': np.random.randn()
					})
					tm.save_result(r)

				t2 = timeit.timeit("func()", globals=globals(), number=num)
				print(f"get+save: {t2:.6f} s, {t2/num:.6e} s/it")

		result[in_memory] = {'t1': t1, 't2': t2}

	print(result[True]['t1'] / result[False]['t1'])
	print(result[True]['t2'] / result[False]['t2'])

	print('-'*30)
