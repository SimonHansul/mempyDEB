
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os

import pandas as pd 

def load_data(paths: dict):
    """
    Load data from a dictionary of paths and optional keyword arguments for reading the data. <br>
    Data *has* to be in tidy (column-oriented) format. <br>
    Metadata can be added to CSV files, by the skiprows argument has to be supllied (see examples).

    ## Examples

    The `paths` dict can contain just the paths to the individual data files: 

    ```
    data = load_data({
        'growth' : 'data/my_growth_data.csv',
        'repro' : 'data/my_repro_data.csv'
    })
    ```

    Optionally, we can supply keyword arguments for each data file to read. <br>
    For example, if a CSV has metadata (which is encouraged), 
    we have to supply the skiprows argument to skip the number of rows containing metadata:

       ```
    data = load_data({
        'growth' : ('data/my_growth_data.csv', {'skiprows' : 4}),
        'repro' : ('data/my_repro_data.csv', {'skiprows' : 4})
    })
    ```

    All kwargs are passed down to `pandas.read_csv`.
    """

    data = {}
    
    for key,info in paths.items():

        # if the info for each data is a list or tuple, 
        # we assume that it contains the path and a dict with kwargs for reading the data
        if type(info) in (list,tuple):
            path = info[0]
            kwargs = info[1]
            data[key] = pd.read_csv(path, **kwargs)
        # otherwise, we assume that just the path has been given
        else:
            path = info
            data[key] = pd.read_csv(path)

    return data


def setup_fit(
        paths: dict,
        colnames: dict,
        intguess: dict = None
        ):
    
    """
    Set-up `ModelFit` object to perform fitting of DEBtox2019 to growth, reproduction, and/or survival data.<br>
    Loads the data and defines the appropriate simulator and loss functions. <br>
    Data is expected to be in *tidy* format sensu Wickham (https://www.jstatsoft.org/article/view/v059i10).

    ## Parameters

    - paths: A dictionary of file paths pointing to data for length-growth, cumulative reproduction and survival. Possible keys are `growth`, `repro`, `survival`.
    - colnames: Maps the variables `time`, `exposure`, `length`, `survival` and `cum_repro` to columns in the corresponding data tables. Endpoints which are not provided can be skipped. 

    ## Examples 

    ```Python 
    fit = setup_fit(
        paths = {'growth' : 'data/growthdata.csv', 'repro' : 'data/reprodata.csv'},
        colnames = {'time' : 't_day', 'length' : 'length_mm', 'cum_repro' : 'cum_repro'}

    )
    ```
    """

    fit = ModelFit()
    
    fit.data = load_data(paths)
    
    if 'growth' in fit.data.keys():
        fit.data['growth'].rename({
            colnames['time'] : 't',
            colnames['length'] : 'L'
            })
        
        # TODO: add optional variable for food level
        # TODO: add optional variable for temperature
    
    if 'repro' in fit.data.keys():
        fit.data['repro'].rename({
            colnames['time'] : 't',
            colnames['cum_repro'] : 'cum_repro'
            })
        
        # TODO: add optional variable for food level
        # TODO: add optional variable for temperature 

    #fit.data.simulator = simulate_DEBtox # TODO: add 
    #fit.data.loss = fit_data.define_loss()


def SSQ(a, b): 
    """
    Sum of squared errors. Example for a loss function.
    """
    return np.sum(np.power(a - b), 2)

class ModelFit:
    """
    A class to guide modellers through the model fitting process. 

    Use `fit = ModelFit()` to initialize an empty model fitting object. 
    Use `fit.guide()` to get an update on which components are still missing.

    After all components have been defined, use either `fit.Bayesian_inference()` 
    to perform Bayesian inference of parameters using SMC-ABC from the pyabc package, 
    or `fit.optimization()` to perform optimization, which internally calls `scipy.optimize.minimize()`.
    """

    def __init__(self):

        self.data: dict = None
        self.simulator: function = None
        self.loss: function = None
        self.prior = None
        self.intguess: dict = None
        self.default_params: dict = None
        
        self.optimization_result = None
        self.abc_history = None
        self.accepted = None

    def plot_priors(self, **kwargs):
        """
        Plot pdfs of the prior distributions. Kwargs are passed down to the plot command.
        """
        
        nrows = int(np.ceil(len(self.prior.keys())/3))
        ncols = np.minimum(3, len(self.prior.keys()))

        fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize = (12,6*nrows))
        ax = np.ravel(ax)

        for i,p in enumerate(self.prior.keys()):

            xrange = np.geomspace(self.prior[p].ppf(0.0001),self.prior[p].ppf(0.9999), 10000)
            ax[i].plot(xrange, self.prior[p].pdf(xrange), **kwargs)
            ax[i].set(xlabel = p)

        ax[0].set(ylabel = "Prior density")
        sns.despine()

        return fig, ax
    
    def prior_sample(self):
        """
        Draw a sample from the priors.
        """
        samples = [self.prior[p].rvs() for p in self.prior.keys()]
        return dict(zip(self.prior.keys(), samples))
    

    def define_lognorm_prior(self, p_int = None, sigma = 1):
        """
        Define log-normal priors with median equal to initial guess and constant sigma (SD of log values).
        """

        if not p_int:
            p_int = self.intguess

        self.prior = pyabc.Distribution()

        for (par,val) in zip(p_int.keys(), p_int.values()):
            self.prior[par] = pyabc.RV("lognorm", sigma, 0, val)
    
    def prior_predictive_check(self, n = 100):
        """
        Evaluates n prior samples. 
        """
        
        self.prior_predictions = []

        for i in range(n): 
            sim = self.simulator(self.prior_sample())
            self.prior_predictions.append(sim)

    def run_bayesian_inference(
            self, 
            popsize = 1000,
            max_total_nr_simulations = 10_000, 
            max_nr_populations = 10,
            temp_database = "data.db"
            ):
        
        """
        Apply Bayesian inference, using Sequential Monte Carlo Approximate Bayesian Computation (SMC-ABC) 
        from the `pyABC` package.
        """
        
        # pyabc expects the data in dict format 
        # if the data is not already given in dict format, 
        # we define a dict with a single entry, 
        # and define a wrapper around the loss function 
        if type(self.data)!=dict: 
            data_smc = {'data' : self.data}

            def loss_SMC(predicted, data):
                return self.loss(predicted, data["data"])
            
        # if the data is already given in dict format, we assume that we can use it as is, 
        # same for the loss function
        else:
            data_smc = self.data
            loss_SMC = self.loss

        # setting things up
        abc = pyabc.ABCSMC( 
            self.simulator, 
            self.prior, 
            loss_SMC, 
            population_size=popsize
            )
         
        db_path = os.path.join(tempfile.gettempdir(), temp_database) # pyABC stores some information in a temporary file, this is set up here
        abc.new("sqlite:///" + db_path, data_smc) # the data is defined as a database entry
        history = abc.run( # running the SMC-ABC
            max_total_nr_simulations = max_total_nr_simulations, # we set a limit on the maximum number of simulations to run
            max_nr_populations = max_nr_populations # and a limit on the maximum number of populations, i.e. successive updates of the probability distributions
            )
        
        # constructing a data frame with accepted parameter values and weights
        accepted, weights = history.get_distribution()
        accepted = accepted.reset_index().assign(weight = weights)

        print("Conducted Bayesian inference using SMC-ABC. Results are in `abc_history` and `accepted`")

        self.abc_history = history
        self.accepted = accepted

    def posterior_sample(self):
        """ 
        Draw a posterior sample from accepted particles.
        """

        sample_ar = self.accepted.sample(weights = 'weight')[list(self.prior.keys())].iloc[0]
        return dict(zip(self.prior.keys(), sample_ar))


    def retrodict(self, n = 100):
        """ 
        Generate retrodictions based on `n` posterior samples.
        """

        self.retrodictions = []

        for i in range(n): 
            sim = self.simulator(self.posterior_sample())
            self.retrodictions.append(sim)

    def define_objective_function(self, for_use_with = 'scipy'):
        
        if for_use_with == 'scipy':
            def objective_function(parvals):
                """
                Objective function for the model contained in a ModelFit object
                """
                
                pfit = self.intguess.copy()
                pfit.update(dict(zip(self.intguess.keys(), parvals)))
                
                # calling the simulator function

                prediction = self.simulator(pfit)

                # calling the loss function
                
                return self.loss(prediction, self.data)
            
            return objective_function
        
        if for_use_with == 'lmfit':
            def objective_function(parvals):
                """
                Objective function for the model contained in a ModelFit object
                """

                parslist = [param.value for param in list(parvals.values())]
                
                pfit = self.intguess.copy()
                pfit.update(dict(zip(self.intguess.keys(), parslist)))
    
                # calling the simulator function

                prediction = self.simulator(pfit)

                # calling the loss function
                
                return self.loss(prediction, self.data)
            
            return objective_function


    def run_optimization(
            self,
            method = 'Nelder-Mead',
            **kwargs
        ): 
        """ 
        Apply an optimization algorithm using `scipy.optimize.minimize`.
        """

        objective_function = self.define_objective_function()

 
        opt = minimize(
            objective_function, # objective function 
            list(self.intguess.values()), # initial guesses
            method = method, # optimization method to use
            **kwargs
            )
               
        print(f"Fitted model using {method} method. Results stored in `optimization_result`")

        self.optimization_result = opt
        self.p_opt = dict(zip(self.intguess.keys(), opt.x))

    def guide(self):
        """
        Provide some information on what has been defined, and what is missing.
        """
    
        defined = []
        undefined = []
        fields = {
            'simulator': 'simulator',
            'data': 'data',
            'loss': 'loss function',
            'prior': 'prior',
            'default_params' : 'default parameters',
            'intguess': 'initial guess'
        }

        for key, description in fields.items():
            value = getattr(self, key)
            if value is not None:
                defined.append(description)
            else:
                undefined.append(description)

        if len(undefined)>0:
            joinstr = ", ".join(undefined)
            print(f"The following fields are so far undefined: \n {joinstr}")
               
            print(f"Minimum needed for optimization: {REQUIRED_FOR_OPTIM}")
            print(f"Minimum needed for Bayesian inference: {REQUIRED_FOR_BAYES}")
        else:
            print("All fields are defined, ready to run optimization or Bayesian inference")


    def __repr__(self):
        return f"ModelFit(data={self.data}, simulator={self.simulator}, prior={self.prior}, intguess={self.intguess})"

