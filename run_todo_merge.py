#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sqlite3
import os.path
import tempfile
from contextlib import closing
import shutil
import shlex
import subprocess
import sys

if __name__ == '__main__':
	# Parse command line arguments:
	parser = argparse.ArgumentParser(description="Merge TODO-files after photometry has be re-run.")
	parser.add_argument('-o', '--overwrite', help='Overwrite existing files.', action='store_true')
	parser.add_argument('todo', type=str, help="TODO-file from photometry.")
	parser.add_argument('derived', type=str, help="TODO-file derived from corrections.")
	parser.add_argument('combined', type=str, nargs='?', default=None)
	args = parser.parse_args()

	fname_todo = args.todo
	fname_derived = args.derived

	# Check that the files exists:
	if not os.path.isfile(fname_todo):
		parser.error("Not a valid TODO-file")
	if not os.path.isfile(fname_derived):
		parser.error("Not a valid derived TODO-file")

	# Decide where the final output file will be placed:
	fname_final = args.combined
	if fname_final is None:
		fname_final = os.path.join(os.path.abspath(os.path.dirname(fname_derived)), 'todo-combined.sqlite')

	# Check if the final output file already exists:
	if os.path.exists(fname_final):
		if args.overwrite:
			os.remove(fname_final)
		else:
			parser.error("File already exists")

	# Read the tables that are available from the derived TODO-file:
	with closing(sqlite3.connect(fname_derived)) as conn:
		cursor = conn.cursor()
		cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
		dump_tables = [r[0] for r in cursor.fetchall()]

		# TODO: Do checks of they can (or should!) be merged!

	# Start working in a tempoary directory to not risk leaving files lying around:
	with tempfile.TemporaryDirectory() as tmpdir:
		# The working copy of the file:
		fname_combined = os.path.join(tmpdir, 'working.sqlite')

		print("Copying existing file...")
		shutil.copy(fname_todo, fname_combined)

		# Clean out the list of tables to be dumped:
		dump_tables.remove('todolist')
		dump_tables.remove('diagnostics')
		dump_tables.remove('photometry_skipped')
		dump_tables.remove('datavalidation_raw')
		for d in dump_tables:
			if d.startswith('sqlite_'):
				dump_tables.remove(d)

		# Using temporary directory to dump the needed tables
		# to raw SQL file and insert them into the combined SQLite file afterwards:
		tmpfile = os.path.join(tmpdir, 'dump.sql')
		open(tmpfile, 'w').close()
		for tab in dump_tables:
			print("Dumping {0:s}...".format(tab))

			cmd = shlex.split('sqlite3 -bail -batch -readonly "{input:s}" ".dump {table:s}" >> "{output:s}"'.format(
				table=tab,
				input=fname_derived,
				output=tmpfile
			))
			subprocess.check_call(cmd, shell=True)

		print("Inserting tables...")
		cmd = shlex.split('cat "{output:s}" | sqlite3 -bail -batch "{input:s}"'.format(
			input=fname_combined,
			output=tmpfile
		))
		subprocess.check_call(cmd, shell=True)

		print("Deleting SQL dump file...")
		os.remove(tmpfile)

		print("Transferring correction status...")
		with closing(sqlite3.connect(fname_combined)) as conn:
			conn.row_factory = sqlite3.Row
			cursor = conn.cursor()

			cursor.execute("PRAGMA table_info(todolist)")
			todolist_columns = [r['name'] for r in cursor.fetchall()]
			if 'corr_status' not in todolist_columns:
				print("Creating corr_status column...")
				cursor.execute("ALTER TABLE todolist ADD COLUMN corr_status INTEGER DEFAULT NULL;")
				conn.commit()

			with closing(sqlite3.connect(fname_derived)) as conn_corr:
				conn_corr.row_factory = sqlite3.Row
				cursor_corr = conn_corr.cursor()
				cursor_corr.execute("SELECT priority,corr_status FROM todolist;")
				for row in cursor_corr.fetchall():
					cursor.execute("UPDATE todolist SET corr_status=? WHERE priority=?;", (
						row['corr_status'],
						row['priority']
					))

			conn.commit()

			cursor.execute("CREATE INDEX IF NOT EXISTS corr_status_idx ON todolist (corr_status);")
			conn.commit()

			print("Analyzing database...")
			cursor.execute("ANALYZE;")
			conn.commit()

			print("Vacuuming database...")
			conn.isolation_level = None
			cursor.execute("VACUUM;")

		# Move the finished file to the final destination:
		print("Moving file to final destination...")
		shutil.move(fname_combined, fname_final)
