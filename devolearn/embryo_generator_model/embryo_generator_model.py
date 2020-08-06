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
from tqdm import tqdm
from PIL import Image
import joblib
import numpy as np
from collections import deque
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt



"""
GAN to generate images of embryos 
"""

class Generator(nn.Module):
    def __init__(self, ngf, nz, nc):
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
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32

            nn.ConvTranspose2d( ngf, ngf, 4, 2, 1, bias=False),  ## added custom stuff here
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64

            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        return self.main(input)


class embryo_generator_model():   
    def __init__(self, mode = "cpu"):

        """
        ngf = size of output image of the GAN 
        nz = size of latent space noise (latent vector)
        nc = number of color channels of the output image

        Do not tweak these unless you're changing the Generator() with a new model with a different architecture. 
    
        """

        self.ngf = 128 ## generated image size 
        self.nz = 128
        self.nc = 1
        self.generator= Generator(self.ngf, self.nz, self.nc)
        self.model_url = "https://github.com/DevoLearn/devolearn/raw/master/devolearn/embryo_generator_model/embryo_generator.pt"
        self.model_name = "embryo_generator.pt"
        self.model_dir = os.path.dirname(__file__)
        # print("at : ", os.path.dirname(__file__))

        try:
            # print("model already downloaded, loading model...")
            self.generator.load_state_dict(torch.load(self.model_dir + "/" + self.model_name, map_location= "cpu"))
        except:
            print("model not found, downloading from: ", self.model_url)
            if os.path.isdir(self.model_dir) == False:
                os.mkdir(self.model_dir)
            filename = wget.download(self.model_url, out= self.model_dir)
            # print(filename)
            self.generator.load_state_dict(torch.load(self.model_dir + "/" + self.model_name, map_location= "cpu"))



        
           


    def generate(self, image_size = (700,500)):

        """
        inputs{
            image_size <tuple> = (width,height of the generated image)
        }
        outputs{
            1 channel image as an <np.array> 
        }
        The native size of the GAN's output is 128*128, and then it resizes the 
        generated image to the desired size. 

        """
        with torch.no_grad():
            noise = torch.randn([1,128,1,1])
            im = self.generator(noise)[0][0].cpu().detach().numpy()
        im = cv2.resize(im, image_size)
        im = 255 - cv2.convertScaleAbs(im, alpha=(255.0))   ## temporary fix against inverted images 

        return im


    def generate_n_images(self, n = 3, foldername = "generated_images", image_size = (700,500)):
        """
        inputs{
            n <int> = number of images to generate
            foldername <str> = name of the folder where the images whould be saved. 
            The function automatically generates a folder if it doesn't exist 
        }
        outputs{
            None
        }
        
        This is an extension of the generator.generate() function for generating multiple images at once and saving them into a folder. 

        """

        if os.path.isdir(foldername) == False:
            os.mkdir(foldername)

        
        for i in tqdm(range(n), desc = "generating images :"):
            filename = foldername + "/" + str(i) + ".jpg"
            gen_image = self.generate()  ## 2d numpy array 
            cv2.imwrite(filename, gen_image)

        print ("Saved ", n, " images in", foldername)