import argparse
import os
import math
import torch
import numpy as np
from utils import *
from fourier_pr import *
from denoiser import *
from hio import *
from skimage.metrics import peak_signal_noise_ratio as psnr

def apls(bfcnn, task, y_init, ground_truth, iters=700, eta=0.1, h0=0.01, beta=1e-5, progress_bar=True):

    device, fft_device = get_devices()
    x_c = task.noisy_measurements.to(fft_device)
    denoiser = lambda y: bfcnn(y.unsqueeze(0)).squeeze(0)

    intermediate_ys = torch.zeros(iters, y_init.shape[0], y_init.shape[1], y_init.shape[2]).to(device)

    init_noise = eta*torch.mean(y_init)*torch.randn(y_init.shape).to(device)
    y = y_init + init_noise

    score_estimate = torch.zeros(y.shape).to(device)
    noise = torch.zeros(y.shape).to(device)
    residual = torch.zeros(y.shape).to(device)
    phase = torch.complex(torch.zeros(task.mask.shape), torch.zeros(task.mask.shape)).to(fft_device)
    
    pbar = tqdm(range(iters)) if progress_bar else range(iters)
    recon_psnr = 0

    with torch.no_grad():
        for i in pbar:

            phase_image = y_init if i == 0 else y
            phase[:] = torch.sgn(task.A((phase_image - denoiser(phase_image))))

            residual[:] = denoiser(y)

            step_size = h0*(i+1)/(1+(h0*(i)))

            score_estimate[:] = (residual - task.AT(task.A(residual))) + ((task.AT(task.A(y))) - task.AT(phase * x_c))

            sigma = torch.norm(score_estimate)/task.length

            gamma = sigma*np.sqrt(((1 - (beta*step_size))**2 - (1-step_size)**2))

            noise[:] = torch.randn(y.shape)

            y = y - step_size*score_estimate + gamma*noise

            image = y - denoiser(y)

            if progress_bar:
                recon_psnr = np.round(psnr(ground_truth.permute(1, 2, 0).detach().cpu().numpy(), (image).permute(1, 2, 0).detach().cpu().numpy(), data_range=1), 2)
                pbar.set_description(f"PSNR: {recon_psnr}")

            intermediate_ys[i, :, :, :] = image

        return image, intermediate_ys

def get_apls_eta(bfcnn, task, y_init, ground_truth, iters=50, h0=0.01, beta=1e-5, min_eta=0, max_eta=1, num_grid=5):

    device, _ = get_devices()

    etas = np.arange(min_eta, max_eta, (max_eta - min_eta)/num_grid)

    best_residual = float('inf')
    best_eta = min_eta
    for eta in etas:
        recon, _ = apls(bfcnn, task, y_init, ground_truth, iters=iters,eta=eta, h0=h0, beta=beta, progress_bar=False)
        residual = torch.norm(task.noisy_measurements - torch.abs(task.A(recon)).to(device))

        if residual < best_residual:
            best_residual = residual
            best_eta = eta
    
    return best_eta

if __name__ == "__main__":      
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", default="../data/grayscale", help="path to data directory")
    parser.add_argument("--results-path", default="../results/apls_results", help="path to results directory")
    parser.add_argument("--image-size", default=128, type=int, help="size to resize images to")
    parser.add_argument('--grayscale', action='store_true', default=True)
    parser.add_argument('--color', dest='grayscale', action='store_false')
    parser.add_argument("--alpha", default=3, type=int)
    
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
        test_image = TestImage(image_path=os.path.join(args.data_path, image_file), grayscale=args.grayscale, size=args.image_size)
        image = test_image.image

        fpr = FourierPR(image, args.alpha)
        iters = 100

        hio_recon = hio_init(fpr, num_starts=50)

        hio_recon = correct_rotation(image, hio_recon)

        best_eta = get_apls_eta(denoiser, fpr, hio_recon, image)

        apls_recon, intermediate_images = apls(denoiser, fpr, hio_recon, image, eta=best_eta)
        
        recon_save_path = os.path.join(args.results_path, os.path.splitext(image_file)[0] + "_apls_figure" + os.path.splitext(image_file)[1])
        gif_save_path = os.path.join(args.results_path, os.path.splitext(image_file)[0] + "_apls.gif")

        save_gif(intermediate_images, gif_save_path)
        
        hio_psnr = np.round(psnr(image.permute(1, 2, 0).detach().cpu().numpy(), hio_recon.permute(1, 2, 0).detach().cpu().numpy(), data_range=1), 2)
        apls_psnr = np.round(psnr(image.permute(1, 2, 0).detach().cpu().numpy(), apls_recon.permute(1, 2, 0).detach().cpu().numpy(), data_range=1), 2)

        plot_images([image.permute(1, 2, 0), hio_recon.permute(1, 2, 0), apls_recon.permute(1, 2, 0)], ["Ground Truth", f"HIO Reconstruction PSNR: {hio_psnr}", f"APLS Reconstruction PSNR: {apls_psnr}"], recon_save_path)

