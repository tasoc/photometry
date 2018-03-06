The TASOC Photometry module
===============================
.. image:: https://travis-ci.org/tasoc/photometry.svg?branch=devel
    :target: https://travis-ci.org/tasoc/photometry

This module provides the basic photometry setup for TASOC

Installation instructions
-------------------------
* Start by making sure that you have `Git Large File Storage (LFS) <https://git-lfs.github.com/>`_ installed. You can verify that is installed by running the command:

  >>> git lfs version

* Go to the directory where you want the Python code to be installed and simply download it or clone it via *git* as::

  >>> git clone https://github.com/tasoc/photometry.git .

* All dependencies can be installed using the following command. It is recommended to do this in a dedicated `virtualenv <https://virtualenv.pypa.io/en/stable/>`_ or similar:

  >>> pip install -r requirements.txt



How to run tests
----------------
You can test your installation by going to the root directory where you cloned the repository and run the command::

 >>> pytest

Running the program
-------------------
The next thing to do is to set up the directories where input data is stored, and where output data (e.g. lightcurves) should be put. This is done by setting the enviroment variables ``TESSPHOT_INPUT`` and ``TESSPHOT_OUTPUT``.
Depending on your operating system and shell this is done in slightly different ways.

The photometry program can by run on a single star by running the program::

  >>> python run_tessphot.py 182092046

Here, the number gives the TIC identifier of the star. The program accepts various other command-line parameters - Try running::

  >>> python run_tessphot.py --help

This is very usefull for testing different methods and settings.

Contributing to the code
------------------------
You are more than welcome to contribute to this code!
Please contact `Rasmus Handberg <rasmush@phys.au.dk>`_ or `Mikkel Lund <mikkelnl@phys.au.dk>`_ if you wish to contribute.
