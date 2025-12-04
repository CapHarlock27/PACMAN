import numpy as np 
import batman
import matplotlib.pyplot as plt

class TransitModel:

    def __init__(self, input_params):
        
        self.params = batman.TransitParams()
        self.name = input_params.get("name")                                    # name of the exoplanet
        self.params.t0 = input_params.get("t0")                                 # time of inferior conjunction
        self.params.per = input_params.get("per")                               # orbital period
        self.params.rp = input_params.get("rp")                                 # planet radius (in units of stellar radii)
        self.params.a = input_params.get("a")                                   # semi-major axis (in units of stellar radii)
        self.params.inc = input_params.get("inc")                               # orbital inclination (in degrees)
        self.params.ecc = input_params.get("ecc")                               # eccentricity
        self.params.w = input_params.get("w")                                   # longitude of periastron (in degrees)
        self.params.u = input_params.get("u")                                   # limb darkening coefficients [u1, u2]
        self.params.limb_dark = input_params.get("limb_darkening_model")        # limb darkening model

        # self.name = "K2-18_b"                       # Name of the exoplanet
        # self.params = batman.TransitParams()
        # self.params.t0 = 0.                         # time of inferior conjunction
        # self.params.per = 32.939623                 # orbital period
        # self.params.rp = 0.0212                     # planet radius (in units of stellar radii)
        # self.params.a = 30.73                       # semi-major axis (in units of stellar radii)
        # self.params.inc = 89.5785                   # orbital inclination (in degrees)
        # self.params.ecc = 0.2                       # eccentricity
        # self.params.w = 354.3                       # longitude of periastron (in degrees)
        # self.params.u = [0.391617, 0.019183]        # limb darkening coefficients [u1, u2]
        # self.params.limb_dark = "quadratic"         # limb darkening model

    def plot(self):
        t = np.linspace(-0.25, 0.25, 100)
        m = batman.TransitModel(self.params, t)      # initializes model
        flux = m.light_curve(self.params)            # calculates light curve
        plt.plot(t, flux)
        plt.xlabel("Time from central transit")
        plt.ylabel("Relative flux")
        plt.title(f"Light curve of {self.name}")
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"{self.name}_assignment1_taskF.png")
        plt.close()