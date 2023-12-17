import matplotlib.pylab as plt

def plot_denoised_image(ground_truth_image, noisy_image, denoised_image, save_path):

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(ground_truth_image.detach().cpu().numpy(), cmap='gray')
    plt.title('Ground Truth')

    plt.subplot(1, 3, 2)
    plt.imshow(noisy_image.detach().cpu().numpy(), cmap='gray')
    plt.title('Noisy')

    plt.subplot(1, 3, 3)
    plt.imshow(denoised_image.detach().cpu().numpy(), cmap='gray')
    plt.title('Denoised')

    plt.savefig(save_path)