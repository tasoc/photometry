# TASOC Photometry [![Build Status](https://travis-ci.org/tasoc/photometry.svg?branch=master)](https://travis-ci.org/tasoc/photometry)
The basic photometry setup for TASOC

## Installation instructions
Go to the directory where you want the Python code to be installed and simply download it or clone it via ``git``:
```
git clone https://github.com/tasoc/photometry.git .
```

All dependencies can be installed using 
```
pip install -r requirements.txt
```
I would recommend to do this in a dedicated ``virtualenv`` or similar.

## How to run tests
You can test your installation by going to the root directory where you cloned the repository and run the command
```
pytest
```

## Running the program
Next thing to do is to set up the directories where input data is stored, and where output data (e.g. lightcurves) should be put. This is done by setting the enviroment variables ``TESSPHOT_INPUT`` and ``TESSPHOT_OUTPUT``. Depending on your operating system and shell this is done in slightly different ways.

The photometry program can by run on a single star by running the program
```
python run_tessphot.py 143159
```
where the number is the TIC identifier of the star. The program accepts various other command-line parameters - Try runnng ``python run_tessphot.py --help``. This is very usefull for testing different methods and settings.

## Contributing to the code
You are more than welcome to contribute to this code!
Please contact Rasmus Handberg (<rasmush@phys.au.dk>) if you wish to contribute.
