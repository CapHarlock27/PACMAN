import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import math

class Generator(nn.Module):
        def __init__(self, ngpu):
            super().__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias = False),
                nn.BatchNorm2d(64 * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias = False),
                nn.BatchNorm2d(64 * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias = False),
                nn.BatchNorm2d(64 * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 1, 4, 2, 1, bias = False), # nc=1
                nn.Tanh()
            )
        def forward(self, input):
            return self.main(input)

class Dream:
    def __init__(self, weights_path, n_plots):
        self.weights_path = weights_path
        self.n_plots = n_plots
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ngpu = 1
    
    def tensor_to_curve(self, tensor_img):
        # Resize to (45, 45) to match training logic
        t_resized = F.interpolate(tensor_img.unsqueeze(0), size = (45, 45), mode = 'bilinear', align_corners = False)
        arr = t_resized.squeeze().detach().cpu().numpy().flatten()
        return arr[:1000]

    def create_generator(self):
        netG = Generator(self.ngpu).to(self.device)
        map_location = None if torch.cuda.is_available() else torch.device('cpu')
        netG.load_state_dict(torch.load(self.weights_path, map_location = map_location))
        netG.eval()
        return netG

    def dream(self, output_file = "GAN_generated_transits.png"):
        netG = self.create_generator()
        nz = 100
        single_plot_w = 6
        single_plot_h = 4
        max_cols = 3
        total_plots = self.n_plots
        cols = min(total_plots, max_cols)
        rows = math.ceil(total_plots / cols)
        total_width = cols * single_plot_w
        total_height = rows * single_plot_h
        with torch.no_grad():
            noise = torch.randn(self.n_plots, nz, 1, 1, device = self.device)
            generated_images = netG(noise)
            plt.figure(figsize = (total_width, total_height))
            for i in range(self.n_plots):
                light_curve = self.tensor_to_curve(generated_images[i])
                ax = plt.subplot(rows, cols, i + 1)
                ax.plot(light_curve, lw = 1, c = 'C0')
                # x = np.arange(len(light_curve))
                # ax.scatter(x, light_curve, s = 2, c = 'C0', alpha = 0.7)                  # scatter plot does not look good
                ax.set_title(f"Generated Transit {i+1}")
                ax.grid(alpha = 0.3)
                ax.set_ylim(0.4, 1.0)
                ax.set_xlabel("Time")
                ax.set_ylabel("Relative flux")
            plt.tight_layout()
            print(f"Saving generated transits to {output_file}...")
            plt.savefig(output_file)
            plt.close()