import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from ipywidgets import *
import numpy as np
import taurex.log
taurex.log.disableLogging()
from taurex.cache import OpacityCache, CIACache
from taurex.temperature import Guillot2010
from taurex.planet import Planet
from taurex.stellar import BlackbodyStar, PhoenixStar
from taurex.chemistry import TaurexChemistry, ConstantGas
from taurex.model import TransmissionModel
from taurex.contributions import AbsorptionContribution, CIAContribution, RayleighContribution
from taurex.binning import FluxBinner, SimpleBinner

class ForwardModel:

    def __init__(self, input_params):
        # Star and Planet Parameters
        self.star_name = input_params.get('star_name', 'Unknown Star')
        self.star_teff = input_params.get('star_teff')
        self.star_radius = input_params.get('star_radius')
        self.planet_name = input_params.get('planet_name', 'Unknown Planet')
        self.planet_radius = input_params.get('planet_radius')
        self.planet_mass = input_params.get('planet_mass')
        # Model Configuration
        self.opacity_path = input_params.get('opacity_path')
        self.cia_path = input_params.get('cia_path')
        self.cia_pairs = input_params.get('cia_pairs', ['H2-H2','H2-He'])
        # Output Filenames
        self.output_params = input_params.get('output_params', self.planet_name + '_parameters.txt')
        self.output_spectrum = input_params.get('output_spectrum', self.planet_name + '_spectrum.dat')
        self.output_plot = input_params.get('output_plot', self.planet_name + '_spectrum.png')
        # Atmospheric Parameters
        atm_params = input_params.get('atmosphere', {})
        self.T_irr = atm_params.get('T_irr')
        self.atm_min_pressure = float(atm_params.get('atm_min_pressure'))
        self.atm_max_pressure = float(atm_params.get('atm_max_pressure'))
        self.nlayers = int(atm_params.get('n_layers'))
        # Chemistry & Abundances
        chemistry_params = input_params.get('chemistry', {})
        self.fill_gases = chemistry_params.get('fill_gases', ['H2', 'He'])
        self.H2_He_ratio = chemistry_params.get('H2_He_ratio', 0.172)
        self.molecules = chemistry_params.get('molecules', [])
        
        abundance_params = chemistry_params.get('abundances', {})
        self.abundance_model = abundance_params.get('model', 'fixed')

        if self.abundance_model == 'fixed':
            # Create dict only for molecules present in the 'molecules' list
            self.mixing_ratios = {mol: abundance_params.get(mol) for mol in self.molecules}
            print("Using fixed mixing ratios:")
            for mol, val in self.mixing_ratios.items():
                print(f"{mol}: {val:.3e}")
        else:
            # Random generation logic
            def random_abundance(log_min = -8, log_max = -2):
                return 10**np.random.uniform(log_min, log_max)
            
            self.mixing_ratios = {}
            for mol in self.molecules:
                self.mixing_ratios[mol] = random_abundance()
            
            print("Using randomized mixing ratios:")
            for mol, val in self.mixing_ratios.items():
                print(f"{mol}: {val:.3e}")

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
        guillot = Guillot2010(T_irr = self.T_irr)
        chemistry = TaurexChemistry(fill_gases = self.fill_gases, ratio = self.H2_He_ratio)
        for mol, val in self.mixing_ratios.items():
            if val is not None:
                chemistry.addGas(ConstantGas(mol, mix_ratio = val))

        # Build the transmission model and add the contributions from physical processes
        tm = TransmissionModel(planet = planet,
                               temperature_profile = guillot,
                               chemistry = chemistry,
                               star = star,
                               atm_min_pressure = self.atm_min_pressure,
                               atm_max_pressure = self.atm_max_pressure,
                               nlayers = self.nlayers)
        tm.add_contribution(AbsorptionContribution())
        tm.add_contribution(CIAContribution(cia_pairs = self.cia_pairs))
        tm.add_contribution(RayleighContribution())
        tm.build()

        # Run the model to get the transmission spectrum
        _ = tm.model()

        # Make a logarithmic grid
        wngrid = np.sort(10000 / np.logspace(-0.4, 1.1, 1000))
        bn = SimpleBinner(wngrid = wngrid)
        bin_wn, bin_rprs, _, _ = bn.bin_model(tm.model(wngrid = wngrid))
        errorbars = np.full_like(bin_rprs, 0.00001)

        # Saving the output spectrum to a file
        print(f"Saving transmission spectrum to {self.output_spectrum}...")
        Data = np.zeros((len(bin_wn), 3))
        Data[:,0] = 10000 / bin_wn
        Data[:,1] = bin_rprs
        Data[:,2] = errorbars
        np.savetxt(self.output_spectrum, Data, header = 'Wavelength(micron) (rp/rs)^2 Error_Bars')

        # Saving the input parameters of the model to a text file
        print(f"Saving input parameters to {self.output_params}...")
        with open(self.output_params, 'w') as f:
            f.write(f"Input Parameters for {self.planet_name} Transmission Model\n")
            f.write("=============================================\n\n")
            f.write("Star Parameters:\n")
            f.write(f"  Star Temperature (K): {self.star_teff}\n")
            f.write(f"  Star Radius (Rs): {self.star_radius}\n\n")

            f.write("Planet Parameters:\n")
            f.write(f"  Planet Radius (Rj): {self.planet_radius}\n")
            f.write(f"  Planet Mass (Mj): {self.planet_mass}\n\n")
            
            f.write("Temperature Profile:\n")
            f.write(f"  Irradiation Temperature (T_irr): {self.T_irr}\n\n")
            
            f.write("Chemistry:\n")
            for mol, val in self.mixing_ratios.items():
                if val is not None:
                    f.write(f"  {mol} Mixing Ratio: {val:.3e}\n")
            
            f.write("\nAtmospheric Pressure Range:\n")
            f.write(f"  Min Pressure (bar): {self.atm_min_pressure}\n")
            f.write(f"  Max Pressure (bar): {self.atm_max_pressure}\n")
            f.write(f"  Number of Layers: {self.nlayers}\n")
        
        if plot:
            print(f"Generating plot and saving to {self.output_plot}...")
            fig, ax = plt.subplots(1, 2, figsize = (16, 6))
            # Plot the atmospheric mixing ratio profiles
            pressure_bar = tm.pressureProfile / 1e5
            for x, gasname in enumerate(tm.chemistry.activeGases):
                profile = tm.chemistry.activeGasMixProfile[x]
                ax[0].plot(profile, pressure_bar, label = gasname, linewidth = 2)
            ax[0].invert_yaxis()
            ax[0].set_yscale("log")
            ax[0].set_xscale("log")
            ax[0].set_title(f'Atmospheric Mixing Ratios of {self.planet_name}')
            ax[0].set_xlabel('Mixing Ratio')
            ax[0].set_ylabel('Pressure (bar)')
            ax[0].legend()
            ax[0].grid(True, which = "both", ls = "-", alpha = 0.3)

            # Plot the binned transmission spectrum
            wavelength_microns = 10000 / bin_wn
            ax[1].scatter(wavelength_microns, bin_rprs, label = 'Binned Spectrum', c = 'b', s = 10, zorder = 2)
            ax[1].errorbar(wavelength_microns, bin_rprs, yerr = errorbars, 
                           label = '10ppm Error', c = 'k', fmt = 'none', alpha = 0.3, zorder = 1)
            ax[1].set_xscale('log')
            ax[1].set_title(f'Transmission Spectrum: {self.planet_name}')
            ax[1].set_xlabel(r'Wavelength ($\mu$m)')
            ax[1].set_ylabel(r'$(R_p/R_s)^2$')
            formatter = mticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-3, -3))
            ax[1].yaxis.get_offset_text().set_fontsize(9)
            ax[1].yaxis.set_major_formatter(formatter)
            ax[1].set_xticks([0.5, 1, 2, 5, 10])
            ax[1].get_xaxis().set_major_formatter(mticker.ScalarFormatter())
            ax[1].legend()
            ax[1].grid(True, which = "both", ls = "-", alpha = 0.3)
            plt.tight_layout()
            
            fig.savefig(self.output_plot, dpi = 150)
            plt.close(fig)