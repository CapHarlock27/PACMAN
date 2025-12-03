import numpy as np
import batman
import matplotlib.pyplot as plt


class TransitModel:
    """
    Transit model class for exoplanet detection using batman.
    
    This class generates synthetic light curves for exoplanet transits
    and provides visualization capabilities.
    """
    
    def __init__(self, params_dict):
        """
        Initialize the transit model with parameters from configuration file.
        
        Parameters
        ----------
        params_dict : dict
            Dictionary containing transit parameters loaded from YAML file
        """
        self.name = params_dict.get('name', 'Unknown')
        self.params = batman.TransitParams()
        self.params.t0 = params_dict.get('t0', 0.0)
        self.params.per = params_dict.get('per')
        self.params.rp = params_dict.get('rp')
        self.params.a = params_dict.get('a')
        self.params.inc = params_dict.get('inc')
        self.params.ecc = params_dict.get('ecc', 0.0)
        self.params.w = params_dict.get('w', 90.0)
        self.params.u = params_dict.get('u', [0.1, 0.3])
        self.params.limb_dark = params_dict.get('limb_dark', 'quadratic')
        
        # Time array for model evaluation
        self.t = np.linspace(-0.075, 0.075, 1000)
        
        # Initialize batman model
        self.model = batman.TransitModel(self.params, self.t)
        
    def compute_light_curve(self):
        """
        Compute the light curve using the batman transit model.
        
        Returns
        -------
        flux : ndarray
            Array of relative flux values
        """
        self.flux = self.model.light_curve(self.params)
        return self.flux
    
    def compute(self): 
        """Return (t, flux) tuple for convenience.""" 
        return self.t, self.compute_light_curve()
    
    def plot_light_curve(self, output_file='lc.png'):
        """
        Plot and save the transit light curve.
        
        Parameters
        ----------
        output_file : str, optional
            Output filename for the plot (default: 'lc.png')
        """
        t = np.linspace(-0.25, 0.25, 1000)
        m = batman.TransitModel(self.params, t)
        flux = m.light_curve(self.params)

        plt.figure(figsize=(10, 6))
        plt.plot(t, flux)
        plt.xlabel("Time from central transit (days)")
        plt.ylabel("Relative flux")
        plt.title(f"Light curve of {self.name}")
        plt.grid()
        plt.tight_layout()
        plt.savefig(output_file)
        plt.show()

        print(f"Light curve saved to {output_file}")
        
    @staticmethod 
    def plot_multiple_light_curves(model_list, output_file="lcs.png"): 
        """ Overplot multiple transit light curves and save them. 
        
        Parameters 
        ---------- 
        output_file : str, optional 
            Output filename for the plot (default: 'lcs.png') 
        """ 
        plt.figure(figsize=(10, 6))

        for model in model_list:
            t = np.linspace(-0.25, 0.25, 1000)
            m = batman.TransitModel(model.params, t)
            flux = m.light_curve(model.params)
            plt.plot(t, flux, label=model.name)

        plt.xlabel("Time from central transit (days)")
        plt.ylabel("Relative flux")
        plt.title("Overplotted Transit Models")
        plt.grid()
        plt.tight_layout()
        plt.legend()
        plt.savefig(output_file)

        print(f"Light curve saved to {output_file}")

    
    def run(self, output_file='lc.png'):
        """
        Run the complete transit model workflow: compute and plot light curve.
        
        Parameters
        ----------
        output_file : str, optional
            Output filename for the plot (default: 'lc.png')
        """
        self.compute_light_curve()
        self.plot_light_curve(output_file)
