import numpy as np
import matplotlib.pyplot as plt
import taurex.log
taurex.log.disableLogging()

from taurex.cache import OpacityCache, CIACache
from taurex.temperature import Isothermal
from taurex.planet import Planet
from taurex.stellar import BlackbodyStar
from taurex.data.spectrum import ObservedSpectrum
from taurex.chemistry import TaurexChemistry, ConstantGas
from taurex.model import TransmissionModel
from taurex.contributions import AbsorptionContribution, CIAContribution, RayleighContribution
from taurex.optimizer.nestle import NestleOptimizer


class RetrievalModel:

    def __init__(self, input_params):

        # === Input observed spectrum ===
        self.input_spectrum = input_params.get("input_spectrum")
        if self.input_spectrum is None:
            raise ValueError("Missing `input_spectrum` in YAML")

        data = np.loadtxt(self.input_spectrum)
        wl = data[:, 0]       # wavelength (micron)
        depth = data[:, 1]    # (rp/rs)^2
        error = data[:, 2] if data.shape[1] > 2 else np.full_like(depth, 1e-5)

        self.wavelength = wl
        self.wavenumber = 10000.0 / wl
        self.depth = depth
        self.error = error
        self.obs = ObservedSpectrum(self.wavenumber, depth, error)

        # === Output filenames ===
        self.output_spectrum = input_params.get(
            "output_retrieved_spectrum",
            "retrieved_spectrum.dat"
        )
        self.output_params = input_params.get(
            "output_retrieved_params",
            "retrieved_parameters.txt"
        )
        self.output_plot = input_params.get(
            "output_retrieved_plot",
            "retrieved_spectrum.png"
        )

        # === Star & Planet ===
        star = input_params.get("star", {})
        planet = input_params.get("planet", {})

        self.star_teff = star.get("Teff")
        self.star_radius = star.get("radius")

        self.planet_radius = planet.get("radius")
        self.planet_mass = planet.get("mass")

        # === Atmosphere parameters ===
        atm = input_params.get("atmosphere", {})

        # NOTE: TaskD notebook uses ISOTHERMAL, not Guillot2010
        self.T_iso = atm.get("T_iso")  # temperature for isothermal profile
        if self.T_iso is None:
            raise ValueError("TaskD retrieval requires `T_iso` in YAML")

        self.atm_min_pressure = float(atm.get("atm_min_pressure"))
        self.atm_max_pressure = float(atm.get("atm_max_pressure"))
        self.nlayers = int(atm.get("n_layers"))

        # === Chemistry ===
        chem = input_params.get("chemistry", {})
        self.fill_gases = chem.get("fill_gases", ["H2", "He"])
        self.H2_He_ratio = chem.get("H2_He_ratio", 0.172)
        self.molecules = chem.get("molecules", [])

        abundance_params = chem.get("abundances", {})
        self.initial_abundances = {
            mol: abundance_params.get(mol, 1e-5)
            for mol in self.molecules
        }

        # === Retrieval parameters ===
        # Format:
        # retrieval_params:
        #   H2O:
        #     bounds: [1e-8, 1e-2]
        #   T:
        #     bounds: [400, 2000]
        self.retrieval_params = input_params.get("retrieval_params", {})

        # === Opacity paths ===
        self.opacity_path = input_params.get("opacity_path")
        self.cia_path = input_params.get("cia_path")
        self.cia_pairs = input_params.get("cia_pairs", ["H2-H2", "H2-He"])

        OpacityCache().clear_cache()
        OpacityCache().set_opacity_path(self.opacity_path)
        CIACache().set_cia_path(self.cia_path)

    # ==================================================================
    # RUN RETRIEVAL
    # ==================================================================
    def run(self, plot=False):

        # === Build Forward Model (as in TaskD) ===
        star = BlackbodyStar(temperature=self.star_teff, radius=self.star_radius)

        planet = Planet(
            planet_radius=self.planet_radius,
            planet_mass=self.planet_mass
        )

        temperature = Isothermal(T=self.T_iso)

        chemistry = TaurexChemistry(
            fill_gases=self.fill_gases,
            ratio=self.H2_He_ratio
        )

        for mol in self.molecules:
            chemistry.addGas(ConstantGas(mol, mix_ratio=self.initial_abundances[mol]))

        tm = TransmissionModel(
            planet=planet,
            temperature_profile=temperature,
            chemistry=chemistry,
            star=star,
            atm_min_pressure=self.atm_min_pressure,
            atm_max_pressure=self.atm_max_pressure,
            nlayers=self.nlayers
        )

        tm.add_contribution(AbsorptionContribution())
        tm.add_contribution(CIAContribution(cia_pairs=self.cia_pairs))
        tm.add_contribution(RayleighContribution())
        tm.build()

        # === Retrieval Optimizer (as in TaskD) ===
        opt = NestleOptimizer(num_live_points=50)
        opt.set_model(tm)
        opt.set_observed(self.obs)

        # Enable fitted parameters
        for p, cfg in self.retrieval_params.items():
            opt.enable_fit(p)
            opt.set_boundary(p, cfg.get("bounds", [1e-8, 1e-2]))

        # === RUN RETRIEVAL ===
        print("Running retrieval...")
        solution = opt.fit()

        # Extract best-fit parameters
        best_params = None

        for sol_idx, map_params, median_params, extra in opt.get_solution():
            best_params = map_params
            opt.update_model(map_params)

        # === Generate best-fit spectrum ===
        wn = self.wavenumber
        _, best_depth, _, _ = self.obs.create_binner().bin_model(tm.model(wn))

        # === Save spectrum ===
        print(f"Saving retrieved spectrum to {self.output_spectrum}")
        err = np.full_like(best_depth, 1e-5)
        data = np.vstack([self.wavelength, best_depth, err]).T
        np.savetxt(
            self.output_spectrum,
            data,
            header="Wavelength(micron) (Rp/Rs)^2 Error"
        )

        # === Save parameters ===
        print(f"Saving retrieved parameters to {self.output_params}")
        with open(self.output_params, 'w') as f:
            f.write("Retrieved Parameters\n")
            f.write("====================\n\n")
            for name, value in best_params.items():
                f.write(f"{name}: {value}\n")

        # === Plotting ===
        if plot:
            print(f"Saving retrieved plot to {self.output_plot}")
            plt.figure(figsize=(8,5))
            plt.errorbar(self.wavelength, self.depth, yerr=self.error,
                         fmt='o', markersize=3, label='Observed')
            plt.plot(self.wavelength, best_depth, label='Retrieved', lw=1.3)
            plt.xscale('log')
            plt.xlabel("Wavelength (micron)")
            plt.ylabel("(Rp/Rs)^2")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_plot, dpi=150)
            plt.close()

