import argparse
import os
import torch
import numpy as np
from utils import *
from fourier_pr import *
from denoiser import *
from fasta import Param, fasta
from hio import *
from skimage.metrics import peak_signal_noise_ratio as psnr

def prdeep_step(model, task, fasta_ops, prox_ops, x0):

    _, fft_device = get_devices()
    b = task.noisy_measurements.to(fft_device)

    # components of proximal mapping
    f = lambda z : 1/(2*prox_ops.sigma_w**2)*torch.norm(torch.abs(z)-b,'fro')**2
    subgrad = lambda z : 1/prox_ops.sigma_w**2*(z-(b*z/torch.abs(z)))
    g = lambda x : prox_ops.prox_lambda/2*torch.real(x).t()*(torch.real(x) - denoise(x, model))
    
    # proximal mapping
    def iterative_prox_map(z, t):
      return ((1/(1+t*prox_ops.prox_lambda))*(z+t*prox_ops.prox_lambda*denoise(z, model)))

    # fasta solution
    solution, _, _ = fasta(task.A,task.AT,f,subgrad,g,iterative_prox_map,x0,fasta_ops)

    return solution


def prdeep(models, task, x0, fasta_ops=None, prox_ops=None):

    device, fft_device = get_devices()

    if not prox_ops:
        prox_ops= Param()
        prox_ops.width=task.length
        prox_ops.height=task.length
        prox_ops.prox_iters=1
        prox_ops.sigma_w=torch.std(task.noisy_measurements.to(fft_device) - torch.abs(task.A(task.x)).to(fft_device))
        prox_ops.prox_lambda=0.2

    if not fasta_ops:
        fasta_ops=Param()
        fasta_ops.max_iters=1000
        fasta_ops.tol=1e-50
        fasta_ops.record_objective=False

    for model_path in models:
        try:
            model = DnCNN().to(device)
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
        except:
            
            model = DnCNN5(1).to(device)
            state_dict = torch.load(model_path, map_location=device)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

        x0 = prdeep_step(model=model,task=task,fasta_ops=fasta_ops,prox_ops=prox_ops,x0=x0)
    
    return x0
    
def denoise(input, model):
  device, _ = get_devices()
  input = torch.real(input.unsqueeze(0)).to(device)
  with torch.no_grad():
    output = model(input)
  return input.squeeze() - output.squeeze()

if __name__ == "__main__":      
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", default="../data/grayscale", help="path to data directory")
    parser.add_argument("--results-path", default="../results/prdeep_results", help="path to results directory")
    parser.add_argument("--model-paths", nargs='+', default = ['../models/DnCNN_50_Gray.pth','../models/DnCNN_25_Gray.pth','../models/DnCNN_15_Gray.pth','../models/DnCNN_10_Gray.pth'], help = "list of paths to pretrained DnCNNs")
    parser.add_argument("--image-size", default=128, type=int, help="size to resize images to")
    parser.add_argument("--alpha", default=3, type=int)
    
    args = parser.parse_args()

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    device, _ = get_devices()

    image_tensors = []
    for image_file in tqdm(os.listdir(args.data_path)):
        if os.path.splitext(image_file)[1] not in [".png", ".jpg", ".jpeg"]:
            continue
        test_image = TestImage(image_path=os.path.join(args.data_path, image_file), grayscale=True, size=args.image_size)
        image = test_image.image

        fpr = FourierPR(image, args.alpha)
        
        hio_recon = hio_init(fpr, num_starts=50)

        hio_recon = correct_rotation(image, hio_recon)

        prdeep_recon = prdeep(args.model_paths, fpr, hio_recon)
        
        recon_save_path = os.path.join(args.results_path, os.path.splitext(image_file)[0] + "_prdeep_figure" + os.path.splitext(image_file)[1])
        
        hio_psnr = np.round(psnr(image.permute(1, 2, 0).detach().cpu().numpy(), hio_recon.permute(1, 2, 0).detach().cpu().numpy(), data_range=1), 2)
        prdeep_psnr = np.round(psnr(image.permute(1, 2, 0).detach().cpu().numpy(), prdeep_recon.permute(1, 2, 0).detach().cpu().numpy(), data_range=1), 2)

        plot_images([image.permute(1, 2, 0), hio_recon.permute(1, 2, 0), prdeep_recon.permute(1, 2, 0)], ["Ground Truth", f"HIO Reconstruction PSNR: {hio_psnr}", f"PrDeep Reconstruction PSNR: {prdeep_psnr}"], recon_save_path)

