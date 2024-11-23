import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt 
import seaborn as sns

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
        p, 
        t_eval = None, 
        rtol = 1e-4,
        **kwargs):

    y0 = [p['L_0'], 0, 0]
    psim = p.copy()
    
    sim = pd.DataFrame()

    for (i,C_W) in enumerate(p['C_W']):
        psim['C_W'] = C_W
        sol = solve_ivp(
            debtox2019, 
            (0,psim['t_max']), 
            y0, 
            args = (psim,), 
            t_eval = t_eval, 
            rtol = rtol, 
            **kwargs
            )
        sim = pd.concat([sim,
                          pd.DataFrame({
                              't':sol.t, 
                              'treatment' : f"T{i}",
                              'C_W' : C_W,
                              'L' : sol.y[0], 
                              'R' : sol.y[1], 
                              'D_j' : sol.y[2]
                              })
        ])

    return sim

defaultparams_debtox2019 = {
    't_max' : 21,
    'C_W' : [0,1,2,4],
    'k_D' : 1,
    'b' : 0.5,
    'z' : 1,
    'pmoa' : 'G',
    'f' : 1.0,
    'r_B' : 0.2,
    'L_0' : 0.1, 
    'L_m' : 1.0,
    'L_p' : 0.7,
    'R_m' : 10,
}

def plot_debtox2019(sim: pd.DataFrame, group = 'treatment'):

    fig, ax = plt.subplots(ncols=3, figsize = (12,5))

    for i,y in enumerate(['D_j', 'L', 'R']):
        sns.lineplot(sim, x = 't', y = y, hue = group, ax=ax[i])

        ax[i].set_xlabel('Time (d)')
        ax[i].set_ylabel(y)

        if i>0:
            ax[i].legend().remove()

    plt.tight_layout()

    return fig, ax