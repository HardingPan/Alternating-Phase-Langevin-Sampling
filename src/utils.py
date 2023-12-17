import matplotlib.pylab as plt
import numpy as np
import torch
import torchvision

class TestImage:
    def __init__(self, image_path, grayscale=True):
        self.image_path = image_path
        self.grayscale = grayscale

        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        self.image = self.load()
    
    def load(self):
        image = plt.imread(self.image_path)
        if len(image.shape) == 3:
            if self.grayscale:
                x = np.expand_dims(np.dot(image[...,:3],[0.299, 0.587, 0.144]), axis=2)
            x = torch.tensor(image).permute(2,0,1)
        else:
            if not self.grayscale:
                x = np.repeat(image.reshape(1, image.shape[0], image.shape[1]), 3, axis=0)
            else:
                x = torch.tensor(image.reshape(1, image.shape[0], image.shape[1]))
        return x.to(self.device)
    

        