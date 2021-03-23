import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
import torchvision.models as models

import os
import cv2
import wget
from tqdm import tqdm, tqdm_notebook
from PIL import Image
import joblib
import numpy as np
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt

from ..base_inference_engine import InferenceEngine


"""
GAN to generate images of embryos 
"""

class Generator(nn.Module):
    def __init__(self, ngf, nz, nc):
        """GAN generator to generate synthetic images of embryos

        Args:
            ngf (int): size of feature maps in generator
            nz (int): size of latent space noise (latent vector)
            nc (int): number of color channels of the output image
        """
        super().__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d( ngf*2, ngf, 4, 2, 1, bias=False),  ## added custom stuff here
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d( ngf, 10, 4, 2, 1, bias=False),  ## added custom stuff here
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            # state size. 10 x 128 x 128
            nn.ConvTranspose2d( 10, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 256 x 256
        )

    def forward(self, input):
        return self.main(input)


class embryo_generator_model(InferenceEngine):   
    def __init__(self, device = "cpu"):
        """Generate synthetic single or multiple images of embryos.
        Do not tweak these unless you're changing the Generator() with a new model with a different architecture. 

        Args:
            device (str, optional): set to "cuda", runs operations on gpu and set to "cpu", runs operations on cpu. Defaults to "cpu".
        """
        self.device = device
        self.ngf = 128 ## generated image size 
        self.nz = 128
        self.nc = 1
        self.generator = Generator(self.ngf, self.nz, self.nc)
        self.model_url = "https://raw.githubusercontent.com/DevoLearn/devolearn/master/devolearn/embryo_generator_model/embryo_generator.pth"
        self.model_name = "embryo_generator.pth"
        self.model_dir = os.path.dirname(__file__)
        # print("at : ", os.path.dirname(__file__))
        #print("Searching here.. ",self.model_dir + "/" + self.model_name)

        try:
            # print("model already downloaded, loading model...")
            self.generator.load_state_dict(torch.load(self.model_dir + "/" + self.model_name, map_location= self.device))
        except:
            print("model not found, downloading from: ", self.model_url)
            if os.path.isdir(self.model_dir) == False:
                os.mkdir(self.model_dir)
            filename = wget.download(self.model_url, out= self.model_dir)
            # print(filename)
            self.generator.load_state_dict(torch.load(self.model_dir + "/" + self.model_name, map_location= self.device))
        
        self.generator.to(self.device)


        
    


    def generate(self, image_size = (700,500)):
        """Generate one synthetic image of embryo.
        The native size of the GAN's output is 256*256, and then it resizes the generated image to the desired size. 
        reference{
            https://github.com/DevoLearn/devolearn#generating-synthetic-images-of-embryos-with-a-pre-trained-gan
        }

        Args:
            image_size (tuple, optional): size of generated image,(width,height). Defaults to (700,500).

        Returns:
            np.array : 1 channel image 
        """
        with torch.no_grad():
            noise = torch.randn([1,128,1,1]).to(self.device)
            im = self.generator(noise)[0][0].cpu().detach().numpy()
        im = cv2.resize(im, image_size)
        im = 255 - cv2.convertScaleAbs(im, alpha=(255.0))   ## temporary fix against inverted images 

        return im


    def generate_n_images(self, n = 3, foldername = "generated_images", image_size = (700,500), notebook_mode = False):
        """This is an extension of the generator.generate() function for generating multiple images at once and saving them into a folder. 
        reference{
            https://github.com/DevoLearn/devolearn#generating-synthetic-images-of-embryos-with-a-pre-trained-gan
        }
        Args:
            n (int, optional): number of images to generate. Defaults to 3.
            foldername (str, optional): name of the folder where the images whould be savedThe function automatically generates a folder if it doesn't exist. Defaults to "generated_images".
            image_size (tuple, optional): size of generated image,(width,height). Defaults to (700,500).
            notebook_mode (bool, optional): toogle between script(False) and notebook(True), for better user interface. Defaults to False.
        """

        if os.path.isdir(foldername) == False:
            os.mkdir(foldername)

        if notebook_mode == True:
            for i in tqdm_notebook(range(n), desc = "generating images :"):
                filename = foldername + "/" + str(i) + ".jpg"
                gen_image = self.generate()  ## 2d numpy array 
                cv2.imwrite(filename, gen_image)
        else:
            for i in tqdm(range(n), desc = "generating images :"):
                filename = foldername + "/" + str(i) + ".jpg"
                gen_image = self.generate()  ## 2d numpy array 
                cv2.imwrite(filename, gen_image)

        print ("Saved ", n, " images in", foldername)