The ``TASOC Photometry`` module
===============================

This module provides the basic photometry setup for TASOC


**Installation instructions**
-----------------------------
* Go to the directory where you want the Python code to be installed and simply download it or clone it via *git* as::

   >>> git clone https://github.com/tasoc/photometry.git .

* All dependencies can be installed using::

       >>> pip install -r requirements.txt

  Currently, this includes the following packages:

      - `six <https://pypi.python.org/pypi/six>`_
      - `pytest <https://docs.pytest.org/en/latest/>`_
      - `numpy <http://www.numpy.org/>`_
      - `scipy <https://www.scipy.org/>`_
      - `matplotlib <http://matplotlib.org/>`_
      - `astropy <http://www.astropy.org/>`_
      - `Bottleneck <https://pypi.python.org/pypi/Bottleneck>`_
      - `h5py <http://www.h5py.org/>`_
      - `scikit-image (version 0.13.0) <http://scikit-image.org/>`_
      - `scikit-learn (version 0.19.0) <http://scikit-learn.org/stable/>`_
      - `statsmodels (version 0.8.0) <http://www.statsmodels.org/stable/index.html>`_

* It is recommended to do this in a dedicated `virtualenv <https://virtualenv.pypa.io/en/stable/>`_ or similar.

**How to run tests**
---------------------
You can test your installation by going to the root directory where you cloned the repository and run the command::

 >>> pytest

**Running the program**
-----------------------
The next thing to do is to set up the directories where input data is stored, and where output data (e.g. lightcurves) should be put. This is done by setting the enviroment variables ``TESSPHOT_INPUT`` and ``TESSPHOT_OUTPUT``.
Depending on your operating system and shell this is done in slightly different ways.

The photometry program can by run on a single star by running the program::

  >>> python run_tessphot.py 143159

Here, the number gives the TIC identifier of the star. The program accepts various other command-line parameters - Try runnng::

  >>> python run_tessphot.py --help

This is very usefull for testing different methods and settings.

**Contributing to the code**
----------------------------
You are more than welcome to contribute to this code!
Please contact `Rasmus Handberg <rasmush@phys.au.dk>`_ or `Mikkel Lund <mikkelnl@phys.au.dk>`_ if you wish to contribute.