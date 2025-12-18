# PACMAN project

**PACMAN** (exo**P**l**A**netary **C**omputations that **M**ake **A**stronomers **N**eurotic) is a Python package designed to detect, characterize, and simulate exoplanets. It utilizes the loyal and unbiased **daneel** CLI tool to perform tasks ranging from light curve analysis to atmospheric retrieval.

The full documentation is at https://tiziano1590.github.io/comp_astro_25/index.html

## Installation

### Prerequisites

- Python >= 3.10
- PyTorch
- TauREx 3 (atmospheric modeling)

### Install from source

```bash
git clone https://github.com/CapHarlock27/PACMAN.git
cd PACMAN
pip install .
```

### Development installation

```bash
git clone https://github.com/CapHarlock27/PACMAN.git
cd PACMAN
pip install -e .
```

## Usage

After installation, you can run daneel from the command line:

```bash
daneel -i <input_file> [options]
```

### Command-line options
#### Required
- `-i, --input`: Path to input parameter file (YAML) or weights file (for Dream mode)
#### Main Actions
- `-t, --transit`: Plot an exoplanet transit light curve from the input parameters
- `-d, --detect [rf|cnn]`: Initialize detection algorithms
    - `rf`: Use Random Forest Classifier model to detect exoplanets
    - `cnn`: Use Convolutional Neural Network model to detect exoplanets
- `--dream`: Generate synthetic light curves using a GAN trained on TESS data
    - `--n_plots <int>`: Number of samples to generate *(Optional; default behavior if omitted. Default: 1)*
- `-a, --atmosphere [model|retrieve]`: Perform atmospheric characterization
    - `model`: Generate a forward transmission spectrum using fixed or random molecular abundances
    - `retrieve`: Performs an atmospheric retrieval through the usage of nested sampling
    - `--plot <bool>`: Provides the spectrum if True is passed *(Optional; default behavior if omitted)*
#### Output Control
- `-o, --output`: Specify the output directory and filename for saving results *(Optional; default behavior if omitted)*

### Examples

```bash
# 1. Run exoplanet transit light curve plot
daneel -i parameters.yaml -t

# 2. Run exoplanet detection using Convolutional Neural Network
daneel -i parameters.yaml -d cnn 

# 3. Run exoplanet detection using Random Forest
daneel -i parameters.yaml -d rf 

# 4. Run generation of synthetic light curves (Dream mode)   
# generates 5 light curves and saves to specific path
daneel -i generator_weights.pt --dream --n_plots 5 -o ~/synthetic_transit.png    

# 5. Run atmospheric forward model
daneel -i parameters.yaml -a model

# 6. Run atmospheric retrival (with plotting)
daneel -i parameters.yaml -a retrieve --plot True

# 7. Combine both detection and atmospheric retrieval
daneel -i parameters.yaml -d rf -a retrieve
```

## Input File Format

The input file must be a YAML file. Below is a minimal example structure for the different modules:

```bash
# parameters.yaml example

# Section for Light Curve
transit:
  rp: 0.15
  # ... other transit params

# Section for Detection
fr:
  n_bins: 1000
  samples_per_class: 350
  # ... other detection params

# Section for Forward Model
forward_model:
  planet_name: "K2-18b"
  chemistry:
    molecules: ["H2O", "CH4"]
  # ... other model params

# Section for Retrieval
retrieval:
  obs_spectrum: "spectrum.dat"
  num_live_points: 200
  atmosphere:
    atm_min_pressure: 1e-0
    atm_max_pressure: 1e6
    n_layers: 30
  # ... boundaries and other settings
```
*For a complete list of parameters, please refer to the **examples/** folder.*

## Dreaming: Generating Synthetic Transits
Daneel can "dream" (generate) new synthetic exoplanetary transit light curves using a pre-trained Generative Adversarial Network (GAN).

### How to generate the generator_weights.pt
If you are training your own GAN to use with Daneel, you must ensure architecture compatibility. This version of Daneel expects the Generator to have a specific structure to successfully load the weights.\
For this reason consider using this exact class design:
```python
import torch.nn as nn

# Configuration required by Daneel
nz = 100   # Size of z latent vector
ngf = 64   # Size of feature maps
nc = 1     # Number of channels

class Generator(nn.Module):
    def __init__(self, ngpu=1):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias = False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State size: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State size: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State size: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State size: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias = False),
            nn.Tanh()
            # Final state size: (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# To save weights for Daneel:
# torch.save(netG.state_dict(), "generator_weights.pt")
```

## License

This project is licensed under the MIT License.

## Authors

Nicholas Friso (nicholas.friso@studenti.unipd.it)\
Marko Ivanovski (marko.ivanovski@studenti.unipd.it)\
Alessandro Matteo Rossi (alessandromatteo.rossi@studenti.unipd.it)\
Francesco Maria Salion (francescomaria.salion@studenti.unipd.it)\
Tiziano Zingales (tiziano.zingales@unipd.it)