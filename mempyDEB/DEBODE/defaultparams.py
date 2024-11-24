
# defaultparams.py
# definiton of default parameters to easily get started, and as a reference for development and tests

from collections import namedtuple

Params = namedtuple("Params", "glb spc")


# default parameters for the DEBBase model (a variant of DEBkiss with TKTD)
defaultparams_DEBBase = Params(
    { # global parameters
        # Simulation parameters
        'tspan' : (0,23), # simulated time span
        'a_int' : 2, # initial age (including embryonic development); the column `t` in the simulation output starts to count time at this age

        # Environmental parameters
        'V_patch':  0.05, # simulated volume (L)
        'dX_in': 375, # food input rate (µg C / d)
        'C_W' : 0. # chemical exposure concentration
    },
    { # species-specific parameters
        'X_emb_int': 0.675, # mass of an egg (µgC)
        'eta_IA_0': 0.35, # assimilation efficiency  (-)
        'eta_AS_0': 0.9, # growth efficency (-)
        'eta_AR_0': 0.95, # reproduction efficiency (-)
        'dI_max': 4,  # maximum specific ingestion rate (µgC µgC^-2/3 d^-1)
        'dI_max_emb': 4, # maximum specific ingestion rate for embryos (µgC µgC^-2/3 d^-1)
        'K_X': 0.5e3, # half-saturation constant for ingestion (µgC / L)
        'kappa': 0.9, # somatic investment fraction (-)
        'eta_SA': 0.9, # shrinking efficiency (-)
        'k_M_0': 0.4, # somatic maintenance constant (d^-1)
        'S_p': 9, # structural mass at puberty (µgC)

        # sublethal TKTD parameters 
        'kD_j' : .5, # dominant rate constant for sublethal effects (d^-1)
        'ED50_j' : 1., # median effective scaled damage for sublethal effects - this model assumes a log-logistic relationsip between damage and relative response on the DEB level (units chemical exposure)
        'beta_j' : 2., # slope for lethal effects (-)
        'pmoa' : 'G', # physiological mode of action (categorical: G, M, A, R)

        # GUTS-SD parameters
        'kD_h' : .5, #  dominant rate constant for lethal effects (d^-1)
        'ED50_h' : 2., # median effective scaled damage for lethal effects (units match chemical exposure)
        'beta_h' : 1. # slope for lethal effects (-)
    }
)

