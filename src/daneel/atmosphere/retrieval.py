import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import time
import taurex.log
taurex.log.disableLogging()
from taurex.cache import OpacityCache, CIACache
from taurex.planet import Planet
from taurex.stellar import BlackbodyStar
from taurex.chemistry import TaurexChemistry, ConstantGas
from taurex.model import TransmissionModel
from taurex.contributions import AbsorptionContribution, CIAContribution, RayleighContribution
from taurex.binning import SimpleBinner
from taurex.temperature import Isothermal
from taurex.data.spectrum.observed import ObservedSpectrum
from taurex.optimizer.nestle import NestleOptimizer

class Retrieval:

    def __init__(self, input_params):
        # Star and Planet Parameters
        self.star_name = input_params.get('star_name', 'Unknown Star')
        self.star_teff = input_params.get('star_teff')
        self.star_radius = input_params.get('star_radius')
        self.planet_name = input_params.get('planet_name', 'Unknown Planet')
        self.planet_radius = input_params.get('planet_radius')
        rad_bound = input_params.get('planet_radius_boundary')
        self.planet_radius_boundary = [float(x) for x in rad_bound] if rad_bound else None
        self.planet_mass = input_params.get('planet_mass')
        # Model Configuration
        self.opacity_path = input_params.get('opacity_path')
        self.cia_path = input_params.get('cia_path')
        self.cia_pairs = input_params.get('cia_pairs', ['H2-H2','H2-He'])
        # Observed Spectrum
        self.obs_spectrum = input_params.get('obs_spectrum')
        # Output Filenames
        self.output_params = input_params.get('output_params', self.planet_name + '_retrieved_parameters.txt')
        self.output_spectrum = input_params.get('output_spectrum', self.planet_name + '_retrieved_spectrum.dat')
        self.output_plot = input_params.get('output_plot', self.planet_name + '_retrieved_spectrum.png')
        # Atmospheric Parameters
        atm_params = input_params.get('atmosphere', {})
        self.T_iso = atm_params.get('T_iso')
        self.atm_min_pressure = float(atm_params.get('atm_min_pressure', 1e-0))
        self.atm_max_pressure = float(atm_params.get('atm_max_pressure', 1e6))
        self.nlayers = int(atm_params.get('n_layers', 30))
        # Chemistry & Abundances
        chemistry_params = input_params.get('chemistry', {})
        self.fill_gases = chemistry_params.get('fill_gases', ['H2', 'He'])
        self.H2_He_ratio = chemistry_params.get('H2_He_ratio', 0.172)
        self.molecules = chemistry_params.get('molecules', [])
        abundance_params = chemistry_params.get('abundances', {})
        self.mixing_ratios = {mol: abundance_params.get(mol) for mol in self.molecules}
        # Retrieval settings
        T_bound = atm_params.get('T_iso_boundary')
        self.T_iso_boundary = [float(x) for x in T_bound] if T_bound else None
        abund_bound = abundance_params.get('boundary')
        self.abundances_boundary = [float(x) for x in abund_bound] if abund_bound else None
        self.num_live_points = int(input_params.get('num_live_points', 50))

        if self.opacity_path is None or self.cia_path is None:
            raise ValueError("Opacity path or CIA path is missing in the parameters.")
        
        # Dynamic Opacity Import
        OpacityCache().clear_cache()
        OpacityCache().set_opacity_path(self.opacity_path)
        CIACache().set_cia_path(self.cia_path)

        # Store cross-sections in a dictionary keyed by molecule name
        self.xsecs = {}
        print("Loading Opacities...")
        for mol in self.molecules:
            try:
                # Load the cross section from cache using the molecule name
                self.xsecs[mol] = OpacityCache()[mol]
                print(f" - Loaded xsec for {mol}")
            except KeyError:
                print(f" ! Warning: Could not find opacity data for {mol} in {self.opacity_path}")

    def run(self, plot):
        # Set up the parameters and chemical abundances for the atmospheric model
        star = BlackbodyStar(temperature = self.star_teff, radius = self.star_radius)
        planet = Planet(planet_radius = self.planet_radius, planet_mass = self.planet_mass)
        isothermal = Isothermal(T = self.T_iso)
        isothermal.nlayers = self.nlayers
        chemistry = TaurexChemistry(fill_gases = self.fill_gases, ratio = self.H2_He_ratio)
        for mol, val in self.mixing_ratios.items():
            if val is not None:
                chemistry.addGas(ConstantGas(mol, mix_ratio = val))
        
        # Build the transmission model and add the contributions from physical processes
        tm = TransmissionModel(planet = planet,
                               temperature_profile = isothermal,
                               chemistry = chemistry,
                               star = star,
                               atm_min_pressure = self.atm_min_pressure,
                               atm_max_pressure = self.atm_max_pressure,
                               nlayers = self.nlayers)
        tm.add_contribution(AbsorptionContribution())
        tm.add_contribution(CIAContribution(cia_pairs = self.cia_pairs))
        tm.add_contribution(RayleighContribution())
        tm.build()

        # Initialize chemistry
        tm.chemistry.initialize_chemistry(
            nlayers = tm.nLayers, 
            temperature_profile = tm.temperatureProfile, 
            pressure_profile = tm.pressureProfile, 
            altitude_profile = None
        )
        
        # Load the observed spectrum and bin it to the model resolution
        obs = ObservedSpectrum(self.obs_spectrum)

        # Setting up the nestle optimizer for the retrieval
        opt = NestleOptimizer(num_live_points = self.num_live_points)

        # Setting up the model and observed spectrum for the optimizer
        opt.set_model(tm)
        opt.set_observed(obs)

        # Set up which parameters to fit and their boundaries
        opt.enable_fit('planet_radius')
        opt.set_boundary('planet_radius', self.planet_radius_boundary)
        opt.enable_fit('T')
        opt.set_boundary('T', self.T_iso_boundary)
        for mol in self.molecules:
            opt.enable_fit(mol)
            opt.set_boundary(mol, self.abundances_boundary)
        
        #Fit the model to the observed spectrum
        print(f"Starting Retrieval with {self.num_live_points} live points...")
        time_start = time.time()
        opt.fit()
        taurex.log.disableLogging()
        time_end = time.time()
        print(f"Retrieval completed in {time_end - time_start:.2f} seconds.")

        # Save retrieved parameters, best-fit spectrum and plot
        print(f"Saving retrieved parameters to {self.output_params}...")
        with open(self.output_params, 'w') as f:
            f.write(f"Retrieved Parameters for {self.planet_name}\n")
            f.write("=============================================\n\n")
            f.write(f"{'Parameter':<20} {'Median':<15} {'+1sigma':<15} {'-1sigma':<15} {'Best-Fit (Map)':<15}\n")
            f.write("-" * 85 + "\n")

            # Iterate through the solution
            for sol_idx, map_vals, median, _ in opt.get_solution():
                opt.update_model(map_vals)
                
                samples = opt.get_samples(sol_idx)
                param_names = opt.fit_names

                # Save parameter statistics
                for i, name in enumerate(param_names):
                    s = samples[:, i]
                    q16, q50, q84 = np.percentile(s, [16, 50, 84])
                    sigma_plus = q84 - q50
                    sigma_minus = q50 - q16
                    best_fit_val = map_vals[i] # or median[i] depending on preference
                    f.write(f"{name:<20} {q50:<15.5e} {sigma_plus:<15.5e} {sigma_minus:<15.5e} {best_fit_val:<15.5e}\n")

                # Save Retrieved Spectrum
                print(f"Saving best-fit spectrum to {self.output_spectrum}...")
                obs_binner = obs.create_binner()
                native_grid, native_depth, _, _ = tm.model()
                bin_wn, bin_depth, _, _ = obs_binner.bin_model(tm.model())
                wl_microns = 10000 / bin_wn
                save_data = np.column_stack((
                    wl_microns,          # Col 1: Wavelength
                    obs.spectrum,        # Col 2: Observed Data
                    obs.errorBar,        # Col 3: Obs Error
                    bin_depth            # Col 4: Best Fit Model
                ))
                
                np.savetxt(self.output_spectrum, save_data, header = 'Wavelength(um)  Obs_Depth  Obs_Error  Retrieved_Model_Depth')

                # Plotting
                if plot:
                    print(f"Saving plot to {self.output_plot}...")
                    fig, ax = plt.subplots(figsize = (8, 6))
                    ax.errorbar(obs.wavelengthGrid, obs.spectrum, yerr=obs.errorBar, label = 'Observations', fmt = '.', color = 'black', alpha = 0.5, zorder = 1)
                    # data = np.genfromtxt(self.output_spectrum, skip_header = 1)
                    wngrid = np.sort(10000 / np.logspace(save_data[-1, 0], save_data[0, 0], 1000))
                    bn = SimpleBinner(wngrid = wngrid)
                    bin_wn, bin_rprs, _, _ = bn.bindown(save_data[:, 0], save_data[:, 1], save_data[:, 2])
                    ax.plot(bin_wn, bin_rprs, label = 'Best Fit (Retrieved)', color = 'red', linewidth = 2, zorder = 2)
                    ax.set_xscale('log')
                    ax.set_xlabel(r'Wavelength ($\mu$m)')
                    ax.set_ylabel(r'Transit Depth $(R_p/R_s)^2$')
                    ax.set_title(f'Retrieval Results: {self.planet_name}')
                    w_min = np.min(obs.wavelengthGrid)
                    w_max = np.max(obs.wavelengthGrid)
                    possible_ticks = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]
                    ticks = [t for t in possible_ticks if t >= w_min * 0.9 and t <= w_max * 1.1]
                    ax.set_xticks(ticks)
                    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
                    formatter = mticker.ScalarFormatter(useMathText = True)
                    formatter.set_scientific(True)
                    formatter.set_powerlimits((-3, -3))
                    ax.yaxis.set_major_formatter(formatter)
                    ax.legend()
                    ax.grid(True, alpha = 0.3)
                    fig.tight_layout()
                    fig.savefig(self.output_plot, dpi = 150)
                    plt.close(fig)
