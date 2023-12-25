import argparse
import os
from tqdm import tqdm
import torch
import torchvision
from utils import *

class FourierPR:
    def __init__(self, x, alpha):
        super(FourierPR, self).__init__()
        
        _, _, self.length = x.shape

        self.pad_length = int(self.length/2)

        self.device, self.fft_device = get_devices()

        mask = torch.ones((self.length,self.length)).to(self.device)
        mask_pad = (self.pad_length,self.pad_length,self.pad_length,self.pad_length)
        self.mask = torch.nn.functional.pad(mask, mask_pad, 'constant')

        z = torch.fft.fft2(self.pad(x).to(self.fft_device))
        noise = alpha*torch.abs(z)*torch.randn(z.shape)
        self.noisy_measurements = torch.sqrt(torch.clamp(torch.abs(z)**2 + noise, min = 0)).to(self.device)

    def pad(self, x):
        pad = (self.pad_length, self.pad_length, self.pad_length, self.pad_length)
        return torch.nn.functional.pad(x, pad, 'constant')

    def A(self, x, pad=True):
        if pad:     
            return torch.abs(torch.fft.fft2(self.pad(x).to(self.fft_device))).to(self.device)
        return torch.abs(torch.fft.fft2(x.to(self.fft_device))).to(self.device)
    
    def crop(self, x):  
        return torchvision.transforms.functional.crop(x, self.pad_length, self.pad_length, self.length, self.length)

    def AT(self, x, crop=True):
        if crop:
            return self.crop(torch.real(torch.fft.ifft2(x.to(self.fft_device))).to(self.device))
        return torch.real(torch.fft.ifft2(x.to(self.fft_device))).to(self.device)

if __name__ == "__main__":      
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", default="../data/grayscale", help="path to data directory")
    parser.add_argument("--results-path", default="../results/corruption_results", help="path to results directory")
    parser.add_argument('--grayscale', action='store_true', default=True)
    parser.add_argument('--color', dest='grayscale', action='store_false')
    parser.add_argument("--alpha", default=3, type=int)
    
    args = parser.parse_args()

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    image_tensors = []
    for image_file in tqdm(os.listdir(args.data_path)):
        if os.path.splitext(image_file)[1] not in [".png", ".jpg", ".jpeg"]:
            continue
        test_image = TestImage(image_path=os.path.join(args.data_path, image_file), grayscale=args.grayscale)
        image = test_image.image

        fpr = FourierPR(image, args.alpha)

        x_c = fpr.noisy_measurements

        corrupted = torch.log(torch.real(x_c))
        
        inverted_corruption = torch.log(fpr.AT(x_c))
        
        save_path = os.path.join(args.results_path, os.path.splitext(image_file)[0] + "_corruption_figure" + os.path.splitext(image_file)[1])
        plot_images([image.permute(1, 2, 0), corrupted.permute(1, 2, 0), inverted_corruption.permute(1, 2, 0)], ["Ground Truth", "Corrupted", "Inverted Corruption"], save_path)

