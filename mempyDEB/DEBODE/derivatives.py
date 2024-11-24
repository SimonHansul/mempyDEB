# derivatives.py
# definition of derivative functions which make up the core of the DEB-TKTD model

import numpy as np

def LL2(x, p): 
    """
    Two-parameter log-logistic function
    """
    return 1/(1 + np.power(x/p[0], p[1]))

def LL2h(x, p):
    """
    Cumulative hazard function of the two-parameter log-logistic function, 
    used to model lethal effects under SD.
    """
    return -np.log(LL2(x, p))

def LL2M(x, p):
    """
    Inverse of the log-logistic application, 
    used to model sublethal effects of PMoAs for which the affected state variable 
    increases with increasing damage (maintenance costs).
    """
    return 1+np.power(x/p[0], p[1]) # to ways of doing this - this one has the more interpretable parameter
    #return 1 - np.log(LL2(x, p))

def DEBBase(t, y, glb, spc, LS_max):
    """
    DEBBase(t, y, glb, spc)

    Derivatives of the "DEBBase" model. <br>
    DEBBase is a formulation of DEBkiss with maturity, where structure is expressed as mass (instead of volume). <br>
    The TKTD part assumes log-logistic relationships between scaled damage and the relative response. <br>
    There is no explicit representation of "stress". Instead, we compute the relative response directly by applying the appropriate form of the dose-response function.
    This is the same model formulation as used in the Julia package DEBBase.jl.

    ## Naming conventions:

    - `eta_AB` is the efficiency of converting A to B, or yield
    - `k_x` is a rate constant related to the derivative of x
    - `y_j` is a relative response with respect to PMoA j
    - `X`, `I`, `A`, `M`, `S` are food, ingestion, assimilation, somatic maintenance and structure, all expressed in mass. 

    args:

    - t: current time point
    - y: vector of states
    - glb: global parameters
    - spc: species-specific parameters
    - LS_max: maximum structural length (expressed as cubic root of mass), calculated from parameters in spc.
    """

    S, R, X_emb, X, D_j = y
    
    X_emb = np.maximum(0, X_emb)
    S = np.maximum(0, S)

    #with warnings.catch_warnings():
    #    warnings.simplefilter('ignore')
    LS = S**(1/3) # current structural length

    # relative responses for sublethal effects
    y_G = (int(spc['pmoa'] == 'G') * LL2(D_j, (spc['ED50_j'], spc['beta_j']))) + (int(spc['pmoa'] != 'G') * 1)
    y_M = (int(spc['pmoa'] == 'M') * LL2M(D_j, (spc['ED50_j'], spc['beta_j']))) + (int(spc['pmoa'] != 'M') * 1)
    y_A = (int(spc['pmoa'] == 'A') * LL2(D_j, (spc['ED50_j'], spc['beta_j']))) + (int(spc['pmoa'] != 'A') * 1)
    y_R = (int(spc['pmoa'] == 'R') * LL2(D_j, (spc['ED50_j'], spc['beta_j']))) + (int(spc['pmoa'] != 'R') * 1)

    eta_AS = spc['eta_AS_0'] * y_G
    k_M = spc['k_M_0'] * y_M
    eta_IA = spc['eta_IA_0'] * y_A
    eta_AR = spc['eta_AR_0'] * y_R

    X_emb = np.maximum(X_emb, 0)
    X = np.maximum(X, 0)
    
    if X_emb > 0: # feeeding and assimilation for embryos
        dI = spc["dI_max_emb"] * S**(2/3)
        dA = dI * spc['eta_IA_0'] # this assumes that embryos are not affected by the stressor
        dX_emb = -dI # change in vitellus
        dX = 0 # change in vood abundanceb
        dD_j = 0 # embryo is ingored in the
    else: # feeding, assimilation for all other life stages
        X_V = X/glb['V_patch'] # converts food mass to a concentration 
        f = X_V / (X_V + spc['K_X']) # scaled relative response
        dI = f * spc["dI_max"] * S**(2/3) # ingestion rate
        dA = dI * eta_IA # assimilation rate
        dX = glb['dX_in'] - dI # change in food abundance
        dX_emb = 0 # change in vitellus
        dD_j = (X_emb <= 0) * (spc['kD_j'] * (LS_max / (LS+1e-10)) * (glb['C_W'] - D_j)) - (D_j * (1/(S+1e-10)) * dS) # toxicokinetics with body size feedback

    dM = k_M * S 
    dS = eta_AS * (spc['kappa'] * dA - dM)

    if dS < 0:
        dS = -(dM / spc['eta_SA'] - spc['kappa'] * dA)
    if (S >= spc["S_p"]):
        dR = eta_AR * (1 - spc['kappa']) * dA
    else:
        dR = 0
    
    return dS, dR, dX_emb, dX, dD_j
            