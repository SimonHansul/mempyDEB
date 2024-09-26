


import yaml
import pandas as pd
from collections import OrderedDict

class AbstractDataset:
    pass

class Dataset(AbstractDataset):
    """
    Datatype to store calibration data, consisting of `time_resolved` and `scalar_data`. 
    This is most useful if data from different sources and/or of different types is pulled 
    together for calibration.
    
    - time_resolved: Dictionary of time-resolved (tabular) data
    - scalar: Dictionary of scalar data (tabular or in dict-form)
    - time_vars: Dictionary indicating the time-column for each time-resolved dataset
    - grouping_vars: Dictionary indicating additional grouping variables for time-resolved and scalar datasets
    - response_vars: Dictionary indicating response variables for time-resolved and scalar datasets
    - weights: Dictionary indicating weights for time-resolved and scalar datasets
    """
    def __init__(self):
        self.time_resolved = OrderedDict()
        self.scalar = OrderedDict()
        self.time_vars = {}
        self.grouping_vars = {"time_resolved": {}, "scalar": {}}
        self.response_vars = {"time_resolved": {}, "scalar": {}}
        self.weights = {"time_resolved": {}, "scalar": {}}

def read_from_path(path: str):
    """
    Read data from a CSV or YAML file based on file extension.
    """
    file_extension = path.split('.')[-1]
    assert file_extension in ["csv", "yml"], f"Unknown file extension {file_extension}, expecting csv or yml"
    
    if file_extension == "csv":
        return pd.read_csv(path)
    
    if file_extension == "yml":
        with open(path, 'r') as file:
            return yaml.safe_load(file)

def load_time_resolved_data(data: Dataset, config: dict):
    """
    Load time-resolved data into the Dataset object based on the configuration.
    """
    for ts_data in config["time_resolved"]:
        df = pd.read_csv(ts_data["path"])
        data.time_resolved[ts_data["name"]] = df

        assert "grouping_vars" in ts_data, "Independent variables entry missing for time-resolved data in config file"
        assert "response_vars" in ts_data, "Response variables entry missing for time-resolved data in config file"

        data.time_vars[ts_data["name"]] = ts_data["time_var"]

        data.grouping_vars["time_resolved"][ts_data["name"]] = [group_var for group_var in ts_data["grouping_vars"]]
        data.response_vars["time_resolved"][ts_data["name"]] = [response_var for response_var in ts_data["response_vars"]]

        data.weights["time_resolved"][ts_data["name"]] = ts_data["weight"]

def load_scalar_data(data: Dataset, config: dict):
    """
    Load scalar data into the Dataset object based on the configuration.
    """
    for sc_data in config["scalar"]:
        data.scalar[sc_data["name"]] = read_from_path(sc_data["path"])

        assert "response_vars" in sc_data, "Response variables entry missing for scalar data in config file"

        if "grouping_vars" in sc_data:
            data.grouping_vars["scalar"][sc_data["name"]] = [group_var for group_var in sc_data["grouping_vars"]]
        else:
            data.grouping_vars["scalar"][sc_data["name"]] = []

        data.response_vars["scalar"][sc_data["name"]] = [response_var for response_var in sc_data["response_vars"]]

        data.weights["scalar"][sc_data["name"]] = sc_data["weight"]

def data_from_config(path_to_config: str, datatype=Dataset):
    """
    Load a dataset from a configuration file (YAML).
    """
    with open(path_to_config, 'r') as file:
        config = yaml.safe_load(file)

    data = datatype()
    load_time_resolved_data(data, config)
    load_scalar_data(data, config)

    return data
