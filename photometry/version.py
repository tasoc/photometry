#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get version identification from git

If the script is located within an active git repository,
git-describe is used to get the version information.
If this is not a git repository, then it is reasonable to
assume that the version is not being incremented and the
version returned will be the release version as read from
the VERSION file, which holds the version information.

The file VERSION will need to be changed by manually. This should be done
before running git tag (set to the same as the version in the tag).

Inspired by
https://github.com/aebrahim/python-git-version

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import subprocess
import os

__all__ = ("get_version",)

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
VERSION_FILE = os.path.abspath(os.path.join(CURRENT_DIRECTORY, '..', 'VERSION'))

#--------------------------------------------------------------------------------------------------
# Find the "git" command to run depending on the OS:
GIT_COMMAND = "git"
if os.name == "nt":
	def find_git_on_windows():
		"""find the path to the git executable on windows"""
		# first see if git is in the path
		try:
			subprocess.check_output(["where", "/Q", "git"])
			# if this command succeeded, git is in the path
			return "git"
		# catch the exception thrown if git was not found
		except subprocess.CalledProcessError:
			pass
		# There are several locations git.exe may be hiding
		possible_locations = []
		# look in program files for msysgit
		if "PROGRAMFILES(X86)" in os.environ:
			possible_locations.append("%s/Git/cmd/git.exe" % os.environ["PROGRAMFILES(X86)"])
		if "PROGRAMFILES" in os.environ:
			possible_locations.append("%s/Git/cmd/git.exe" % os.environ["PROGRAMFILES"])
		# look for the github version of git
		if "LOCALAPPDATA" in os.environ:
			github_dir = "%s/GitHub" % os.environ["LOCALAPPDATA"]
			if os.path.isdir(github_dir):
				for subdir in os.listdir(github_dir):
					if not subdir.startswith("PortableGit"):
						continue
					possible_locations.append("%s/%s/bin/git.exe" % (github_dir, subdir))
		for possible_location in possible_locations:
			if os.path.isfile(possible_location):
				return possible_location
		# git was not found
		return "git"

	GIT_COMMAND = find_git_on_windows()

#--------------------------------------------------------------------------------------------------
def git_describe(pep440=False, abbrev=7):
	"""return the string output of git desribe"""
	arguments = [GIT_COMMAND, "describe", "--tags", "--abbrev=%d" % abbrev]
	try:
		git_str = subprocess.check_output(arguments, cwd=CURRENT_DIRECTORY,
			stderr=subprocess.DEVNULL).decode("ascii").strip()
	except (OSError, subprocess.CalledProcessError):
		return None

	if "-" not in git_str:  # currently at a tag
		return git_str
	else:
		# formatted as version-N-githash
		# want to convert to version.postN-githash
		git_str = git_str.replace("-", ".post", 1)
		if pep440:  # does not allow git hash afterwards
			return git_str.split("-")[0]
		else:
			return git_str.replace("-g", "+git")

#--------------------------------------------------------------------------------------------------
def git_getbranch():
	arguments = [GIT_COMMAND, "symbolic-ref", "--short", "HEAD"]
	try:
		return subprocess.check_output(arguments, cwd=CURRENT_DIRECTORY,
			stderr=subprocess.DEVNULL).decode("ascii").strip()
	except (OSError, subprocess.CalledProcessError):
		return None

#--------------------------------------------------------------------------------------------------
def read_release_version():
	"""Read version information from VERSION file"""
	try:
		with open(VERSION_FILE, "r") as infile:
			version = str(infile.read().strip())
		if len(version) == 0:
			version = None
		return version
	except IOError:
		return None

#--------------------------------------------------------------------------------------------------
def update_release_version():
	"""Update VERSION file"""
	version = get_version(pep440=True)
	with open(VERSION_FILE, "w") as outfile:
		outfile.write(version)

#--------------------------------------------------------------------------------------------------
def get_version(pep440=False, include_branch=True):
	"""
	Tracks the version number.

	The file VERSION holds the version information. If this is not a git
	repository, then it is reasonable to assume that the version is not
	being incremented and the version returned will be the release version as
	read from the file.

	However, if the script is located within an active git repository,
	git-describe is used to get the version information.

	The file VERSION will need to be changed by manually. This should be done
	before running git tag (set to the same as the version in the tag).

	Parameters:
		pep440 (bool): When True, this function returns a version string suitable for
		a release as defined by PEP 440. When False, the githash (if
		available) will be appended to the version string.

	Returns:
		string: Version sting.
	"""

	git_version = git_describe(pep440=pep440)
	if git_version is None: # not a git repository
		return read_release_version()

	if include_branch:
		git_branch = git_getbranch()
		if git_branch is not None:
			git_version = git_branch + '-' + git_version

	return git_version

#--------------------------------------------------------------------------------------------------
if __name__ == "__main__":
	print(get_version())
