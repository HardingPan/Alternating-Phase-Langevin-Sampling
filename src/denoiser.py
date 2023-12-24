'''
    Bias-Free DnCNN
    Architecture from https://github.com/LabForComputationalVision/universal_inverse_problem/blob/943b9ed99e35a4af86c36c413e2ca70ab320e8bc/code/network.py
    BFCNN proposed by https://arxiv.org/abs/1906.05478
    DnCNN proposed by https://arxiv.org/abs/1608.03981
'''

import argparse
import os
from utils import *
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision

class BFCNN(nn.Module):

    def __init__(self, padding=1, num_kernels=64, kernel_size=3, num_layers=20, num_channels=1):
        super(BFCNN, self).__init__()

        self.padding = padding
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.num_channels = num_channels

        self.conv_layers = nn.ModuleList([])
        self.running_sd = nn.ParameterList([])
        self.gammas = nn.ParameterList([])

        self.conv_layers.append(nn.Conv2d(self.num_channels,self.num_kernels, self.kernel_size, padding=self.padding , bias=False))

        for l in range(1,self.num_layers-1):
            self.conv_layers.append(nn.Conv2d(self.num_kernels ,self.num_kernels, self.kernel_size, padding=self.padding , bias=False))
            self.running_sd.append( nn.Parameter(torch.ones(1,self.num_kernels,1,1), requires_grad=False) )
            g = (torch.randn( (1,self.num_kernels,1,1) )*(2./9./64.)).clamp_(-0.025,0.025)
            self.gammas.append(nn.Parameter(g, requires_grad=True) )

        self.conv_layers.append(nn.Conv2d(self.num_kernels,self.num_channels, self.kernel_size, padding=self.padding , bias=False))


    def forward(self, x):

        relu = nn.ReLU(inplace=True)
        x = relu(self.conv_layers[0](x))

        for l in range(1,self.num_layers-1):
            x = self.conv_layers[l](x)
            sd_x = torch.sqrt(x.var(dim=(0,2,3) ,keepdim = True, unbiased=False)+ 1e-05)
            if self.conv_layers[l].training:
                x = x / sd_x.expand_as(x)
                self.running_sd[l-1].data = (1-.1) * self.running_sd[l-1].data + .1 * sd_x
                x = x * self.gammas[l-1].expand_as(x)
            else:
                x = x / self.running_sd[l-1].expand_as(x)
                x = x * self.gammas[l-1].expand_as(x)
            x = relu(x)

        x = self.conv_layers[-1](x)
        return x
    
if __name__ == "__main__":      
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", default="../data/grayscale", help="path to data directory")
    parser.add_argument("--results-path", default="../results/denoiser_results", help="path to results directory")
    parser.add_argument("--noise-std", default=25, type=float, help="standard deviation of additive gaussian noise")
    parser.add_argument('--grayscale', action='store_true', default=True)
    parser.add_argument('--color', dest='grayscale', action='store_false')

    args = parser.parse_args()

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    num_channels = 1 if args.grayscale else 3
    pretrained_path = "../models/BFDnCNN_BSD400_Gray.pt" if args.grayscale else "../models/BFDnCNN_BSD300_Color.pt"

    denoiser = BFCNN(num_channels=num_channels).to(device)
    denoiser.load_state_dict(torch.load(pretrained_path, map_location=device))
    denoiser.eval()

    image_tensors = []
    for image_file in tqdm(os.listdir(args.data_path)):
        if os.path.splitext(image_file)[1] not in [".png", ".jpg", ".jpeg"]:
            continue
        test_image = TestImage(image_path=os.path.join(args.data_path, image_file), grayscale=args.grayscale)
        image = test_image.image
        
        noisy_image = image + torch.randn(image.shape).to(device)*(args.noise_std/255)

        residual = denoiser(noisy_image.unsqueeze(0))
        denoised_image = noisy_image - residual.squeeze()
        
        save_path = os.path.join(args.results_path, os.path.splitext(image_file)[0] + "_denoised_figure" + os.path.splitext(image_file)[1])
        plot_images([image.permute(1, 2, 0), noisy_image.permute(1, 2, 0), denoised_image.permute(1, 2, 0)], ["Ground Truth", "Noisy", "Denoised"],  save_path)

