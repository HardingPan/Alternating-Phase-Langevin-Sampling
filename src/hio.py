import argparse
import os
import math
import torch
import numpy as np
from utils import *
from fourier_pr import *
from skimage.metrics import peak_signal_noise_ratio as psnr

def hio(task, iters, prev=None, guess=None):
     
    _, fft_device = get_devices()
    x_c = task.noisy_measurements.to(fft_device)
    l = 2*task.length
    beta = 0.8

    if guess is None:
        rand_phase = torch.exp(1j * torch.randn(x_c.shape) * 2 * math.pi).to(fft_device)
        guess = torch.mul(x_c, rand_phase)

    for i in range(iters):
        update = torch.mul(x_c, torch.exp(1j * torch.angle(guess)))
        inv = task.AT(update, crop=False)

        if prev is None:
            prev = inv

        temp = inv
        constraints = ((inv < 0) & (task.mask == 1))  | (task.mask == 0) 
        inv[constraints] = (prev - beta*inv)[constraints]
        
        prev = temp
        guess = torch.fft.fft2(inv.to(fft_device))

    return prev, guess

def hio_init(task, num_starts=100, start_iters=50, full_iters=1000):
    
    best_residual = float('inf')
    best_prev = None
    best_guess = None
    device, _ = get_devices()

    for i in range(num_starts):
        prev, guess = hio(task, start_iters)
        recon = task.crop(prev)
        residual = torch.norm(task.noisy_measurements - torch.abs(task.A(recon)).to(device))

        if residual < best_residual:
            best_residual = residual
            best_guess = guess
            best_prev = prev

    final_recon, _ = hio(task, full_iters, prev=best_prev, guess=best_guess)

    return task.crop(final_recon)

def correct_rotation(gt, recon):
    rotated = torchvision.transforms.functional.rotate(recon, 180)

    if torch.norm(rotated - gt) < torch.norm(recon - gt):
        return rotated
    return recon


if __name__ == "__main__":      
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", default="../data/grayscale", help="path to data directory")
    parser.add_argument("--results-path", default="../results/hio_results", help="path to results directory")
    parser.add_argument("--image-size", default=128, type=int, help="size to resize images to")
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
        test_image = TestImage(image_path=os.path.join(args.data_path, image_file), grayscale=args.grayscale, size=args.image_size)
        image = test_image.image

        fpr = FourierPR(image, args.alpha)
        iters = 100

        hio_recon = hio_init(fpr, num_starts=10)

        hio_recon = correct_rotation(image, hio_recon)
        
        save_path = os.path.join(args.results_path, os.path.splitext(image_file)[0] + "_hio_figure" + os.path.splitext(image_file)[1])

        recon_psnr = np.round(psnr(image.permute(1, 2, 0).detach().cpu().numpy(), hio_recon.permute(1, 2, 0).detach().cpu().numpy(), data_range=1), 2)
        plot_images([image.permute(1, 2, 0), hio_recon.permute(1, 2, 0)], ["Ground Truth", f"HIO Reconstruction PSNR: {recon_psnr}"], save_path)

