#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import sqlite3
import os.path
import tempfile
from contextlib import closing
import shutil
import shlex
import subprocess

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	# Parse command line arguments:
	parser = argparse.ArgumentParser(description="Merge TODO-files after photometry has be re-run.")
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('-o', '--overwrite', help='Overwrite existing files.', action='store_true')
	parser.add_argument('todo', type=str, help="TODO-file from photometry.")
	parser.add_argument('derived', type=str, help="TODO-file derived from corrections.")
	parser.add_argument('combined', type=str, nargs='?', default=None)
	args = parser.parse_args()

	fname_todo = args.todo
	fname_derived = args.derived

	# Set logging level:
	logging_level = logging.INFO
	if args.quiet:
		logging_level = logging.WARNING
	elif args.debug:
		logging_level = logging.DEBUG

	# Setup logging:
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	console = logging.StreamHandler()
	console.setFormatter(formatter)
	logger = logging.getLogger(__name__)
	logger.addHandler(console)
	logger.setLevel(logging_level)

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
	fname_derived = os.path.abspath(fname_derived)
	with closing(sqlite3.connect('file:%s?mode=ro' % fname_derived, uri=True)) as conn:
		cursor = conn.cursor()
		cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
		dump_tables = [r[0] for r in cursor]

		# Do checks of they can (or should!) be merged!
		cursor.execute("ATTACH DATABASE '%s' AS original;" % os.path.abspath(fname_todo))
		cursor.execute("SELECT COUNT(*) FROM main.todolist;")
		count_table1 = cursor.fetchone()[0]
		cursor.execute("SELECT COUNT(*) FROM original.todolist;")
		count_table2 = cursor.fetchone()[0]
		if count_table1 != count_table2:
			raise RuntimeError("The two TODO-files are incompatible")
		# Columns that should be the same:
		cursor.execute("""SELECT COUNT(*) FROM main.todolist t1 LEFT JOIN original.todolist t2 ON t1.priority=t2.priority WHERE
			t2.priority IS NULL
			OR t1.starid != t2.starid
			OR t1.sector != t2.sector
			OR t1.camera != t2.camera
			OR t1.ccd != t2.ccd
			OR t1.cbv_area != t2.cbv_area
		;""")
		if cursor.fetchone()[0] != 0:
			raise RuntimeError("The two TODO-files are incompatible")

		# TODO: Create list of priorities where corrections should be re-run.
		# This is if the method has changed, or the status have changed:
		cursor.execute("""SELECT t1.priority FROM main.todolist t1 LEFT JOIN original.todolist t2 ON t1.priority=t2.priority WHERE
			(COALESCE(t1.method, t2.method) IS NOT NULL AND t1.method != t2.method)
			OR t2.status IS NULL
			OR t1.status != t2.status
		;""")
		delete_these = set([row[0] for row in cursor])

		# CHECK IF TABLES EXIST in TODO.SQLITE and are empty
		cursor.execute("SELECT name FROM original.sqlite_master WHERE type='table';")
		existing_tables = [r[0] for r in cursor]

	# Start working in a temporary directory to not risk leaving files lying around:
	with tempfile.TemporaryDirectory() as tmpdir:
		# The working copy of the file:
		fname_combined = os.path.join(tmpdir, 'working.sqlite')

		logger.info("Copying existing file...")
		shutil.copy(fname_todo, fname_combined)

		# Clean out the list of tables to be dumped:
		dump_tables.remove('todolist')
		dump_tables.remove('diagnostics')
		dump_tables.remove('photometry_skipped')
		dump_tables.remove('datavalidation_raw')
		for d in dump_tables:
			if d.startswith('sqlite_'):
				dump_tables.remove(d)

		# Check if there were empty tables in the todo.sqlite file that we
		# should delete before we merge in the tables from todo-method.sqlite:
		delete_tables = list(set(existing_tables) & set(dump_tables))
		if delete_tables:
			with closing(sqlite3.connect(fname_combined)) as conn:
				cursor = conn.cursor()
				for tbl in delete_tables:
					cursor.execute("SELECT COUNT(*) FROM %s;" % tbl)
					if cursor.fetchone()[0] == 0:
						cursor.execute("DROP TABLE %s;" % tbl)
						conn.commit()
					else:
						raise RuntimeError("bla bla bla: %s" % tbl)

		# Using temporary directory to dump the needed tables
		# to raw SQL file and insert them into the combined SQLite file afterwards:
		tmpfile = os.path.join(tmpdir, 'dump.sql')
		open(tmpfile, 'w').close()
		for tab in dump_tables:
			logger.info("Dumping %s...", tab)

			cmd = shlex.split('sqlite3 -bail -batch -readonly "{input:s}" ".dump {table:s}" >> "{output:s}"'.format(
				table=tab,
				input=fname_derived,
				output=tmpfile
			))
			logger.debug("Running command: %s", cmd)
			subprocess.check_call(cmd, shell=True)

		logger.info("Inserting tables...")
		cmd = shlex.split('cat "{output:s}" | sqlite3 -bail -batch "{input:s}"'.format(
			input=fname_combined,
			output=tmpfile
		))
		logger.debug("Running command: %s", cmd)
		subprocess.check_call(cmd, shell=True)

		logger.info("Deleting SQL dump file...")
		os.remove(tmpfile)

		logger.info("Transferring correction status...")
		with closing(sqlite3.connect(fname_combined)) as conn:
			conn.row_factory = sqlite3.Row
			cursor = conn.cursor()

			cursor.execute("PRAGMA table_info(todolist)")
			todolist_columns = [r['name'] for r in cursor]
			if 'corr_status' not in todolist_columns:
				logger.info("Creating corr_status column...")
				cursor.execute("ALTER TABLE todolist ADD COLUMN corr_status INTEGER DEFAULT NULL;")
				conn.commit()

			with closing(sqlite3.connect('file:%s?mode=ro' % fname_derived, uri=True)) as conn_corr:
				conn_corr.row_factory = sqlite3.Row
				cursor_corr = conn_corr.cursor()
				cursor_corr.execute("SELECT priority,corr_status FROM todolist;")
				for row in cursor_corr:
					update_status = row['corr_status']
					if row['priority'] in delete_these:
						update_status = None

					cursor.execute("UPDATE todolist SET corr_status=? WHERE priority=?;", (
						update_status,
						row['priority']
					))

			conn.commit()

			cursor.execute("CREATE INDEX IF NOT EXISTS corr_status_idx ON todolist (corr_status);")
			conn.commit()

			# Clean up tables for left-overs:
			if 'diagnostics_corr' in dump_tables:
				cursor.execute("DELETE FROM diagnostics_corr WHERE priority IN (SELECT priority FROM todolist WHERE corr_status IS NULL);")
				conn.commit()

			logger.info("Analyzing database...")
			cursor.execute("ANALYZE;")
			conn.commit()

			logger.info("Vacuuming database...")
			conn.isolation_level = None
			cursor.execute("VACUUM;")

		# Move the finished file to the final destination:
		logger.info("Moving file to final destination...")
		shutil.move(fname_combined, fname_final)
