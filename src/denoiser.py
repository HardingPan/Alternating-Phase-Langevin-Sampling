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
from collections import OrderedDict

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
    
class DnCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=17, act_mode='R'):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(DnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        m_head = DnCNN.conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [DnCNN.conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = DnCNN.conv(nc, out_nc, mode='C', bias=bias)

        self.model = DnCNN.sequential(m_head, *m_body, m_tail)

    def sequential(*args):
        """Advanced nn.Sequential.
        Args:
            nn.Sequential, nn.Module
        Returns:
            nn.Sequential
        """
        if len(args) == 1:
            if isinstance(args[0], OrderedDict):
                raise NotImplementedError('sequential does not support OrderedDict input.')
            return args[0]  # No sequential is needed.
        modules = []
        for module in args:
            if isinstance(module, nn.Sequential):
                for submodule in module.children():
                    modules.append(submodule)
            elif isinstance(module, nn.Module):
                modules.append(module)
        return nn.Sequential(*modules)
    
    def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.2):
        L = []
        for t in mode:
            if t == 'C':
                L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
            elif t == 'T':
                L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
            elif t == 'B':
                L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
            elif t == 'I':
                L.append(nn.InstanceNorm2d(out_channels, affine=True))
            elif t == 'R':
                L.append(nn.ReLU(inplace=True))
            elif t == 'r':
                L.append(nn.ReLU(inplace=False))
            elif t == 'L':
                L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
            elif t == 'l':
                L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
            elif t == '2':
                L.append(nn.PixelShuffle(upscale_factor=2))
            elif t == '3':
                L.append(nn.PixelShuffle(upscale_factor=3))
            elif t == '4':
                L.append(nn.PixelShuffle(upscale_factor=4))
            elif t == 'U':
                L.append(nn.Upsample(scale_factor=2, mode='nearest'))
            elif t == 'u':
                L.append(nn.Upsample(scale_factor=3, mode='nearest'))
            elif t == 'v':
                L.append(nn.Upsample(scale_factor=4, mode='nearest'))
            elif t == 'M':
                L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
            elif t == 'A':
                L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
            else:
                raise NotImplementedError('Undefined type: '.format(t))
        return DnCNN.sequential(*L)

    def forward(self, x):
        n = self.model(x)
        return n

class DnCNN5(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN5, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return out

    
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

    device, _ = get_devices()
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

