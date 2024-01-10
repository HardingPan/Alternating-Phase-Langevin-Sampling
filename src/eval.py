import argparse
import os
import torch
import numpy as np
from utils import *
from fourier_pr import *
from denoiser import *
from hio import *
from apls import *
from prdeep import *
from skimage.metrics import peak_signal_noise_ratio as psnr

if __name__ == "__main__":      
    parser = argparse.ArgumentParser()

    parser.add_argument("--image-path", default="../data/grayscale/cameraman.png", help="path to image")
    parser.add_argument("--results-path", default="../results/eval_results", help="path to results directory")
    parser.add_argument("--image-size", default=128, type=int, help="size to resize images to")
    parser.add_argument("--alpha", default=3, type=int)
    
    args = parser.parse_args()

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    device, _ = get_devices()
    num_channels = 1
    bfcnn_pretrained_path = "../models/BFDnCNN_BSD400_Gray.pt"

    bfcnn = BFCNN(num_channels=num_channels).to(device)
    bfcnn.load_state_dict(torch.load(bfcnn_pretrained_path, map_location=device))
    bfcnn.eval()

    dncnn_model_paths = ['../models/DnCNN_50_Gray.pth','../models/DnCNN_25_Gray.pth','../models/DnCNN_15_Gray.pth','../models/DnCNN_10_Gray.pth']

    test_image = TestImage(image_path=args.image_path, grayscale=True, size=args.image_size)
    image = test_image.image
    _, image_file = os.path.split(args.image_path)

    fpr = FourierPR(image, args.alpha)
    iters = 100

    hio_recon = hio_init(fpr, num_starts=50)

    hio_recon = correct_rotation(image, hio_recon)

    apls_recon, _ = apls(bfcnn, fpr, hio_recon, image, eta=0.6)
    prdeep_recon = prdeep(dncnn_model_paths, fpr, hio_recon)
    
    save_path = os.path.join(args.results_path, os.path.splitext(image_file)[0] + "_eval_figure" + os.path.splitext(image_file)[1])

    hio_psnr = np.round(psnr(image.permute(1, 2, 0).detach().cpu().numpy(), hio_recon.permute(1, 2, 0).detach().cpu().numpy(), data_range=1), 2)
    apls_psnr = np.round(psnr(image.permute(1, 2, 0).detach().cpu().numpy(), apls_recon.permute(1, 2, 0).detach().cpu().numpy(), data_range=1), 2)
    prdeep_psnr = np.round(psnr(image.permute(1, 2, 0).detach().cpu().numpy(), prdeep_recon.permute(1, 2, 0).detach().cpu().numpy(), data_range=1), 2)

    plot_images([image.permute(1, 2, 0), hio_recon.permute(1, 2, 0), prdeep_recon.permute(1, 2, 0), apls_recon.permute(1, 2, 0)], ["Ground Truth", f"HIO Reconstruction PSNR: {hio_psnr}", f"PrDeep Reconstruction PSNR: {prdeep_psnr}", f"APLS Reconstruction PSNR: {apls_psnr}"], save_path)