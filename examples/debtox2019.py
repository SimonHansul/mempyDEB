import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt 
import seaborn as sns

sns.set_palette('viridis')

def debtox2019(t, y, p, interp = None):
    """
    Definition of the DEBtox2019 derivatives.
    """
    L,R,D_j = y

    # damage dynamcs - without any feedbacks for now
    dD = p['k_D']*(p['C_W']-D_j) 

    # calculation of the stress functions 

    s_G = (p['pmoa']=='G') * p['b'] * np.maximum(0, D_j - p['z']) 
    s_M = (p['pmoa']=='M') * p['b'] * np.maximum(0, D_j - p['z'])  
    s_A = (p['pmoa']=='A') * p['b'] * np.maximum(0, D_j - p['z']) 
    s_R = (p['pmoa']=='R') * p['b'] * np.maximum(0, D_j - p['z']) 

    # derivatives for length and cumulative reproduction

    dL = p['r_B']*(1+s_M)/(1+s_G)*(p['f']*p['L_m']*(1-s_A)/(1+s_M)-L)
    dR = (L>p['L_p']) * np.maximum(0, p['R_m']*(1/(1+s_R))*(p['f']*p['L_m']*L**2*(1-s_A)-(p['L_p']**3)*(1+s_M))/(p['L_m']**3 - p['L_p']**3))

    return dL,dR,dD


def simulate_debtox(
        p: dict, 
        t_eval = None, 
        rtol = 1e-4,
        **kwargs):
    """
    Simulates the debtox2019 model. 

    - `p`: Dictionary of parameters and forcings. 
    - `t_eval`: Time-points to be evaluated by the ODE solver. Default is `None`, for which one solution per unit of time will be returned (depending on the unit of t_max).
    - `rtol`: Relative tolerance. Default is 1e-4. Higher tolerances lead to artefacts during discrete events (e.g. puberty).
    - `**kwargs`: Optional keyword arguments for the ODE solver (`scipy.integrate.solve_IVP`).
    
    """

    y0 = [p['L_0'], 0, 0]
    psim = p.copy()
    
    sim = pd.DataFrame()

    # if t_eval is not provided, evaluate once per unit of time
    if not (type(t_eval) in [np.ndarray, list]):
        t_eval = np.arange(0, p['t_max']+1)

    for (i,C_W) in enumerate(p['C_W']):

        # update current exposure
        psim['C_W'] = C_W

        # solve the módel
        sol = solve_ivp(
            debtox2019, 
            (0,psim['t_max']), 
            y0, 
            args = (psim,), 
            t_eval = t_eval, 
            rtol = rtol, 
            **kwargs
            )
        
        # output to data frame
        sim_i = pd.DataFrame({
                              't':sol.t, 
                              'treatment' : f"T{i}",
                              'C_W' : C_W,
                              'L' : sol.y[0], 
                              'R' : sol.y[1], 
                              'D_j' : sol.y[2]
                              }).assign(
                                  cum_repro = lambda df: df.R.shift(psim['emb_dev_time'], fill_value=0)
                               )
        # collect results
        sim = pd.concat([sim, sim_i])

    return sim

defaultparams_debtox2019 = {
    't_max' : 21, # max simulated timespan 
    'C_W' : [0,1,2,4], # exposure concentrations
    'k_D' : 1, # dominant rate constant 
    'b' : 0.5, # slope of damage-response relationship
    'z' : 1, # damage threshold
    'pmoa' : 'G', # physiological mode of action
    'f' : 1.0, # scaled functional response
    'r_B' : 0.2, # von Bertalanffy growth rat 
    'L_0' : 0.1, # initial length
    'L_m' : 1.0, # maximum length
    'L_p' : 0.7, # length at puberty
    'R_m' : 10, # maximum reproduction rate
    'emb_dev_time' : 2, # embryonic development time, used to shift the observations when comparing model output with observed reproduction
}

def plot_debtox2019_sim(sim: pd.DataFrame, group = 'C_W'):
    """
    Plot debtox2019 simulation output without observations.

    - `sim`: Simulation output
    - `group`: Grouping variable. Default is `C_W` (separate line for each exposure concentration, for constant exposure).
    """

    fig, ax = plt.subplots(ncols=3, figsize = (12,5))

    for i,y in enumerate(['D_j', 'L', 'cum_repro']):
        sns.lineplot(sim, x = 't', y = y, hue = group, ax=ax[i])

        ax[i].set_xlabel('Time (d)')
        ax[i].set_ylabel(y)

        if i>0:
            ax[i].legend().remove()

    plt.tight_layout()

    return fig, ax

def plot_debtox2019_data_cols(data: dict, group: str = 'C_W'):
    """
    Plot observed data compatible with a debtox2019 model. 
    
    - ``data``: Data dictionary as constructed by the `load_data` function
    - ``group``: 
    """
    
    fig, ax = plt.subplots(ncols=4, figsize = (16,5))

    if 'growth' in data.keys():
        sns.scatterplot(data['growth'], x = 't', y = 'L', hue = group, ax = ax[1])
    if 'repro' in data.keys():
        sns.scatterplot(data['repro'], x = 't', y = 'cum_repro', hue = group, ax = ax[2])
    if 'survival' in data.keys():
        sns.scatterplot(data['survival'], x = 't', y = 'S', hue = group, ax = ax[3])

    plt.tight_layout()
    [a.set(xlabel = 'Time (d)') for a in ax]

    return fig, ax

def select_group(df, group, groupval):
    return df.loc[df[group]==groupval]

def plot_debtox2019_data_grid(data, group):
    # FIXME: I have forgotten how slow python can become...
    # this is not a very practical way to plot data because it takes a couple of seconds even for a small dataset

    unqiue_group_vals = np.unique(np.concat([d[group] for d in data.values()]))
    num_group_vals = len(unqiue_group_vals)
    fig, ax = plt.subplots(ncols=num_group_vals, nrows = 4, figsize = (4*num_group_vals,5), sharey = "row")

    # iterate over groups (e.g. exposure concentrations)
    for i,val in enumerate(unqiue_group_vals):
        # set title based on group value and name of the variable
        ax[0,i].set_title(f'{group}={val}') 

        # if available, plot growth data (length over time)
        if 'growth' in data.keys():
            sns.scatterplot(
                select_group(data['growth'], group, val),
                x = 't', y = 'L', 
                ax = ax[1,i]
                )
            
        # if available, plot repro data (cumulative reproduction over time)
        if 'repro' in data.keys():
            sns.scatterplot(
                select_group(data['repro'], group, val), 
                 x = 't', y = 'cum_repro', 
                 ax = ax[2,i]
            )

        # if available, plot survival data (survival fraction over time)
        if 'survival' in data.keys():
            sns.scatterplot(
                select_group(data['survival'], group, val), 
                x = 't', y = 'S', 
                ax = ax[3,i]
                )
    # set axis labels

    [a.set(xlabel = 'Time (d)') for a in ax[-1,:]]

    ax[0,0].set_ylabel('$D_j$')
    ax[1,0].set_ylabel('$L$')
    ax[2,0].set_ylabel('$R$')
    ax[3,0].set_ylabel('$s$')

    plt.tight_layout()

    return fig, ax

def plot_debtox2019_data(data: dict, group: str= 'C_W', layout='cols'):
    """"
    Function to plot data compatible with debtox2019 model. 

    - data: dictionary with data sources as constructed by `load_data`
    - group: name of the treatment column, default is 'C_W' (could also be food, temperature for modified use-cases)
    """

    if layout=='cols':
        fig, ax = plot_debtox2019_data_cols(data, group)
        return fig, ax
    
    if layout=='grid':
        fig, ax = plot_debtox2019_data_grid(data, group)
        return fig, ax