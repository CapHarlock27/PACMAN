# Daneel

A practical example to detect and characterize exoplanets.

The full documentation is at https://tiziano1590.github.io/comp_astro_25/index.html

## Installation

### Prerequisites

- Python >= 3.10

### Install from source

```bash
git clone https://github.com/tiziano1590/comp_astro_25.git
cd comp_astro_25
pip install .
```

### Development installation

```bash
git clone https://github.com/tiziano1590/comp_astro_25.git
cd comp_astro_25
pip install -e .
```

## Usage

After installation, you can run daneel from the command line:

```bash
daneel -i <input_file> [options]
```

### Command-line options
#### Required
- `-i, --input`: Path to input parameter file
#### Main Actions
- `-t, -- transit`: Plot an exoplanet transit light curve from the input parameters
- `-d, --detect`: Initialize detection algorithms for exoplanets
- `--dream`: Generate and plot synthetic exoplanet transit light curves using a GAN trained on TESS data
    - `--n_plots`: Number of GAN-generated light curves to produce when using --dream. *(Optional; default behavior if omitted)*
- `-a, --atmosphere`: Perform atmospheric characterization using the supplied transmission spectrum
#### Output Control
- `-o, --output`: Specify the output directory and filename for saving results *(Optional; default behavior if omitted)*

### Examples

```bash
# Run exoplanet transit light curve plot
daneel -i parameters.yaml -t

# Run exoplanet detection using Convolutional Neural Network
daneel -i parameters.yaml -d cnn 

# Run exoplanet detection using Random Forest
daneel -i parameters.yaml -d rf 

# Run generation of synthetic light curves
# generates 1 light curve and produce GAN_generated_transits.png
daneel -i generator_weights.pt --dream      
# generates 5 light curves and produce syntetic_transit.png in the directory ~/
daneel -i generator_weights.pt --dream --n_plots 5 -o ~/syntetic_transit.png    

# Run atmospheric characterization
daneel -i parameters.yaml -a

# Run both detection and atmospheric analysis
daneel -i parameters.yaml -d -a
```

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
```
At the end, save only the state dictionary:
```python
# Correct way to save for Daneel
torch.save(netG.state_dict(), "generator_weights.pt")
```

## Input File Format

The input file should be a YAML file containing the necessary parameters for the analysis.\
In this dictionary, consider following the *batman* library parameters format while also adding a key with the name of the selected exoplanet.

### Example

```bash
name: "K2-18_b"                       # name of the exoplanet
t0: 0                                 # time of inferior conjunction
per: 32.939623                        # orbital period
rp: 0.0212                            # planet radius (in units of stellar radii)
a: 30.73                              # semi-major axis (in units of stellar radii)
inc: 89.5785                          # orbital inclination (in degrees)
ecc: 0.2                              # eccentricity
w: 354.3                              # longitude of periastron (in degrees)
u: [0.391617, 0.019183]               # limb darkening coefficients [u1, u2]
limb_darkening_model: "quadratic"     # limb darkening model
```


## License

This project is licensed under the MIT License.

## Author

Tiziano Zingales (tiziano.zingales@unipd.it)
