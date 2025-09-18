# mempyDEB

mempyDEB is a Python package that provides the basics for doing DEB-TKTD modelling in Python. 
It mostly defines a baseline model, default parameters and functions to run the model.

If you find this useful/interesting, also consider checking out the Julia package [EcotoxSystems.jl](https://github.com/SimonHansul/EcotoxSystems/jl.git) and the corresponding [Python wrapper](https://github.com/SimonHansul/pyEcotoxSystems/tree/main).

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

This pacakge was developed for the course "Mechanistic Effect Modelling" at Osnabrück University, as well as the postgraduate course "Dynamic Modelling of Ecotoxicological Effects" organized at University of Copenhagen. 
