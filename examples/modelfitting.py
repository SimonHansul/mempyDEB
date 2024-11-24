
from scipy.optimize import minimize
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os

import pandas as pd 

from debtox2019 import *

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

        data[key].sort_values('C_W')

    return data

def SSQ(a, b): 
    """
    Sum of squared errors.
    """
    return np.sum(np.power(a - b, 2))

def log_relerr(a, b):
    """
    Log relative error, computed as ln((a/b)+1).
    """

    return np.sum(np.log(((a+1e-10)/(b+1e-10))+1e-10))

def loss_debtox(
        predicted: pd.DataFrame,
        observed: dict, 
        error_model = log_relerr
        ):
    """
    Default loss function for the DEBtox model. <br>
    At the moment, this uses the sum of squares and
    """
    #TODO: add loss for survival
    #TODO: deal with missing endpoints

    scale_L = np.max(observed['growth'].L)
    scale_R = np.max(observed['repro'].cum_repro)

    loss_L = pd.merge(
        observed['growth'], predicted, 
        on = ['t', 'C_W'], 
        suffixes=['_observed', '_predicted']).groupby('C_W').apply(
            lambda df : pd.DataFrame({'loss' : [error_model(df.L_predicted/scale_L, df.L_observed/scale_L)]}),
            include_groups=False
        ).loss.sum()

    loss_R = pd.merge(
        observed['repro'], predicted, 
        on = ['t', 'C_W'], 
        suffixes=['_observed', '_predicted']).groupby('C_W').apply(
            lambda df : pd.DataFrame({'loss' : [error_model(df.cum_repro_predicted/scale_L, df.cum_repro_observed/scale_R)]}),
            include_groups=False
        ).loss.sum()

    return (loss_L/loss_R)/2


def setup_fit(
        paths: dict,
        colnames: dict
        ):
    
    """
    Set-up `ModelFit` object to perform fitting of DEBtox2019 to growth, reproduction, and/or survival data.<br>
    Loads the data and defines the appropriate simulator and loss functions. <br>
    Data is expected to be in *tidy* format sensu Wickham (https://www.jstatsoft.org/article/view/v059i10).

    ## Parameters

    - paths: A dictionary of file paths pointing to data for length-growth, cumulative reproduction and survival. Possible keys are `growth`, `repro`, `survival`.
    - colnames: Maps the variables `time`, `exposure`, `length`, `survival`, `cum_repro` to columns in the corresponding data tables. Endpoints which are not provided can be skipped. Optional additional colname fields are `temperature`.

    ## Examples 

    ```Python 
    fit = setup_fit(
        paths = {'growth' : 'data/growthdata.csv', 'repro' : 'data/reprodata.csv'},
        colnames = {'time' : 't_day', 'length' : 'length_mm', 'cum_repro' : 'cum_repro'}

    )
    ```
    """

    fit = ModelFit()

    #### loading and processing data ####
    
    fit.data = load_data(paths) # collects data from different csv files
    fit.adjust_colnames(colnames) # rename columns in the data to match columns in the simulation output
    
    # keeping track of unique tested exposures and observed time-points across endpoints
    fit.unique_C_Wvals = np.unique(np.concat([df.C_W.unique() for df in fit.data.values()]))
    fit.unique_timepoints = np.unique(np.concat([df.t.unique() for df in fit.data.values()]))
    
    # TODO: here we should add some data checks. e.g. monotonic increase/decrease in cum repro and survival, respectively

    # defining a function to plot the data
    def plot_data(**kwargs):
        fig, ax = plot_debtox2019_data(fit.data, **kwargs)
        return fig,ax

    fit.plot_data = plot_data

    #### configuring simulations ####

    # we start with the defualt debtox2019 parameters
    fit.default_params = defaultparams_debtox2019

    # adjusting exposure concentrations and time-range
    fit.default_params.update({
        't_max' : np.max(fit.unique_timepoints)+1,
        'C_W' : fit.unique_C_Wvals
    })

    # here we define the simulator, 
    # i.e. the function which generates a prediction of the data from a parameter proposal
    
    def simulator(p, t_eval=fit.unique_timepoints, **kwargs):
        psim = fit.default_params.copy()
        psim.update(p)
        sim = simulate_debtox(psim, t_eval=t_eval, **kwargs)


        return sim

    fit.simulator = simulator

    return fit


def isolate_controls(data: dict):
    """
    Isolate the controls from a data dict, based on the minimum C_W values.
    """

    control_data = {} # initialize an empty dict

    # for all data frames in the data dict
    for key,df in data.items():
        # get the subset with the minimum C_W values and add them to the previously initialized dict
        control_data[key] = df.loc[df.C_W==df.C_W.min()]

    return control_data

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

    def adjust_colnames(self, colnames: dict):
        if 'growth' in self.data.keys():
            self.data['growth'].rename(columns={
                colnames['time'] : 't',
                colnames['length'] : 'L'
                }, inplace=True)
            

            if 'temp' in self.data['growth'].keys():
                self.data['growth'].rename(columns={
                    colnames['temp'] : 'T'
                }, inplace=True)

            self.data['growth'].t = self.data['growth'].t.astype(float)
            self.data['growth'].sort_values('C_W', inplace=True)
            
            # TODO: add optional variable for food level
        
        if 'repro' in self.data.keys():
            self.data['repro'].rename(columns={
                colnames['time'] : 't',
                colnames['cum_repro'] : 'cum_repro'
                }, inplace=True)
            
            self.data['repro'].t = self.data['repro'].t.astype(float)
            self.data['repro'].sort_values('C_W', inplace=True)
            
            if 'temperature' in self.data['repro'].keys():
                self.data['repro'].rename(columns={
                    colnames['temp'] : 'T'
                }, inplace=True)
            # TODO: add optional variable for food level
    
        if 'survival' in self.data.keys():
            self.data['survival'].rename(columns={
                colnames['time'] : 't',
                colnames['survival'] : 'S'
                }, inplace=True)
            

            if 'temperature' in self.data['repro'].keys():
                self.data['repro'].rename(columns={
                    colnames['temp'] : 'T'
                }, inplace=True)
            # TODO: add optional variable for food level
    

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

    def define_objective_function(self, data = None):

        if not data: 
            data = self.data
        
        def objective_function(parvals):
            """
            Objective function for the model contained in a ModelFit object. <br>
            Can be directly applied to fitting functions which expect the parameters as list-like input.
            """
            
            pfit = self.intguess.copy()
            pfit.update(dict(zip(self.intguess.keys(), parvals)))
            
            # calling the simulator function

            prediction = self.simulator(pfit)

            # calling the loss function
            
            return self.loss(prediction, data)
        
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

        x0 = list(self.intguess.values())

        bounds = [(0,None) for _ in self.intguess.items()]

        opt = minimize(
            objective_function, # objective function 
            x0, # initial guesses
            method = method, # optimization method to use
            bounds = bounds,
            **kwargs
            )
               
        print(f"Fitted model using {method} method. Results stored in `optimization_result`")

        self.optimization_result = opt
        self.p_opt = dict(zip(self.intguess.keys(), opt.x))

    def fit_controls(self, method = 'Nelder-mead', update_defaults=False, **kwargs):
        """
        Rudimentary method to estimate just the baseline parameters from the controls. <br>
        Requires that intguess is defined by the user accordingly (no TKTD parameters in the intguess).

        - method: Local optimization method to use
        - update_defaults: Wether or not to update default_params entry in the ModelFit object. Default is False and will return the optimzation object from scipy.
        """
        
        self.default_params['C_W'] = [0.] # temporarily turning off simulation of exposures

        objective_function = self.define_objective_function(
            data = isolate_controls(self.data) 
            )
        
        x0 = list(self.intguess.values())
        bounds = [(0,None) for _ in self.intguess.items()]

        opt =  minimize(
                objective_function, # objective function 
                x0, # initial guesses
                method = method, # optimization method to use
                bounds = bounds,
                **kwargs
                )
        
        self.default_params['C_W'] = self.unique_C_Wvals # turning simulation of exposures back on
        
        print(f"Fitted model using {method} method. Results stored in `optimization_result`")
        
        if not update_defaults:
            print("update_defaults is set to False, returning optimization object")
            return opt
        else:
            print("update_defaults is set to True, updating default_params in the ModelFit object based on the estimated parameters")
            self.default_params.update(dict(zip(self.intguess.keys(), opt.x)))
            return None
    

    #def run_global_optimization(
    #        self,
    #        bounds,
    #        method: function = brute
    #        ):
    #    """
    #    EXPERIMENTAL. Appply global optimization. <br>
    #    This will currently directly return the optimization result and not save the results anywhere in the ModelFit object. <br>
    #    We need to do more work on finding good methods for global optim in the context of DEB-TKTD models.
    #    """
    #
    #    assert method in [brute, differential_evolution], "Your provided method is currently not considered for global optimization. Use brute or differential_evolution."
    #
    #    opt = self.define_objective_function()
    #    res = method(opt, bounds)
    #
    #    return res

    def __repr__(self):
        return f"ModelFit(data={self.data}, simulator={self.simulator}, prior={self.prior}, intguess={self.intguess})"

