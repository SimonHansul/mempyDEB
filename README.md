# mempyDEB

mempyDEB is a Python package that provides the basics for doing DEB-TKTD modelling in Python. 
It mostly defines a baseline model, default parameters and functions to run the model.

## Installation

To install `mempyDEB`, use the command

`pip install git+https://github.com/simonhansul/mempyDEB.git`

(e.g. in Anaconda prompt with the desired environment activated). <br>

If you want to try out mempyDEB but don't want to install a Python environment, you could do so in a [Google Colab notebook](https://colab.research.google.com).
You can run `%pip install git+https://github.com/simonhansul/mempyDEB.git` directly in the notebook cell to install mempyDEB on colab.

## Getting started

The examples directory contains a notebook which demonstrates the basic functionality of this package. <br>
In short, you can run a default simulation using

```Python
from mempyDEB.DEBODE.simulators import * # imports functions to run models
from mempyDEB.DEBODE.defaultparams import * # imports default parameters
sim = simulate_DEBBase(defaultparams_DEBBase) # runs the baseline model (a variant of DEBkiss) with default parameters
```

Generally, `mempyDEB` is a fairly slim package, designed as a low-barrier entry point to DEB-TKTD modelling. <br>
There are not many functions, but the code can be adapted to be used for more extensive (research) tasks. <br>
The built-in model is currently missing starvation rules, aging, and some more things, but the package will receive updates whenever it is needed for teaching purposes (or students are contributing through their projects!). The next update will probably be some built-in functionality for model fitting.

## Info & Acknowledgements

This pacakge was developed for the course "Mechanistic Effect Modelling" at Osnabrück University, as well as the postgraduate course "Dynamic Modelling of Ecotoxicological Effects" organized at University of Copenhagen. <br><br>



## Changelog 


## v0.1.0-0.1.1

These were the initial versions, providing an implemntation of the full DEBkiss model, without routines for data fitting.

### v0.2.0

This update was made after the 2024 edition of the course "Dynamic Modelling of Ecotoxicological Effects".<br>
During the course, I learned that students need some more pre-cooked examples for fitting models to *standard* data, e.g. using the DEBtox2019 model. <br>
After all, most of us want to spend time learning about the effects of chemicals in the environment, and not some much about the nitty-gritty details of getting things implemented. This is what the BYOM package by Tjalling Jager does so greatly, and where we still have some catching up to do in the Python world.<br><br>
I hope that this update makes it easier for learners to get acquainted with the model fitting process, and I am indebted to the 2024 students for *biting the bullet* with me and learning things the hard way. Your sacrifice will not be forgotten by future generations of Python modellers.<br>

Conesquently, this update mostly focussed on the debtox2019 model and some pre-cooked functionality for model fiting.

- Added implementation of debtox2019 (basic version without pre-implemented feedbacks, constant exposure only)
- Added `ModelFit` object (equivalent to `ModelFittingAssistant.py` used in the Roskilde 2024 course)
- Added `setup_fit` function for the debtox2019 model. Semi-automatically sets up the ModelFit object for a debtox2019 fitting routine. Requires paths pointing to data files as input, as well as info on which column name should be mapped to which debtox2019 state variable.
- Added functions for plotting debtox2019-compatible data.

