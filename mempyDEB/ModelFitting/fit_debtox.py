from ModelFittingAssistant import ModelFit
import pandas as pd
from .DEBODE.simulators import simulate_debtox

fit = ModelFit()

def load_data(paths: dict):

    data = {}
    
    for key,path in paths.items():
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
    
    
    