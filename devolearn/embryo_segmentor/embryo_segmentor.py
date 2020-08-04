import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage

import os
import cv2
import imutils
from tqdm import tqdm
from PIL import Image
import numpy as np
from collections import deque
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import warnings
warnings.filterwarnings("ignore") 

"""
3d segmentation model for C elegans embryo
"""

# Loading the external github package being used as smp

def install(package):
    subprocess.call(['pip', 'install', package])

package = "segmentation_models_pytorch"
install(package)

def generate_centroid_image(thresh):
    thresh = cv2.blur(thresh, (5,5))
    thresh = thresh.astype(np.uint8)
    centroid_image = np.zeros(thresh.shape)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    centroids = []
    for c in cnts:
        try:
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # draw the contour and center of the shape on the image
            cv2.drawContours(centroid_image, [c], -1, (255, 255, 255), 2)
            cv2.circle(centroid_image, (cX, cY), 2, (255, 255, 255), -1)
            centroids.append((cX, cY))
        except:
            pass

    return centroid_image, centroids

class embryo_segmentor(nn.Module):
    def __init__(self):
        super().__init__()
        self.ENCODER = 'resnet18'
        self.ENCODER_WEIGHTS = 'imagenet'
        self.CLASSES = ["nucleus"]
        self.ACTIVATION = 'sigmoid'
        self.DEVICE = 'cpu'
        self.in_channels = 1

        self.model = smp.FPN(
                encoder_name= self.ENCODER, 
                encoder_weights= self.ENCODER_WEIGHTS, 
                classes=len(self.CLASSES), 
                activation= self.ACTIVATION,
                in_channels = self.in_channels 
            )

        self.model = torch.load("devolearn/models/3d_segmentation_model.pth", map_location= "cpu") 
        self.model.eval()

        self.mini_transform = transforms.Compose([
                                     transforms.ToPILImage(),
                                     transforms.Resize((256,256), interpolation = Image.NEAREST),
                                     transforms.ToTensor(),
                                    ])


    def predict(self, image_path, pred_size = (350,250), centroid_mode = False):
        im = cv2.imread(image_path,0)
        tensor = self.mini_transform(im).unsqueeze(0)
        res = self.model(tensor).detach().cpu().numpy()[0][0]
        res = cv2.resize(res,pred_size)
        if centroid_mode == False:
            return res
        else:
            centroid_image, centroids = generate_centroid_image(res)
            return centroid_image, centroids
            

    def predict_from_video(self, video_path, pred_size = (350,250), save_folder = "preds", centroid_mode = False):
        vidObj = cv2.VideoCapture(video_path)   
        success = 1
        images = deque()
        count = 0

        if centroid_mode == True:
            filenames_centroids = []

        while success: 
            success, image = vidObj.read() 
            
            try:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                images.append(image)
                 
            except:
                print("skipped possible corrupt frame number : ", count)
            count += 1 
        
        if os.path.isdir(save_folder) == False:
            os.mkdir(save_folder)
       
        for i in tqdm(range(len(images)), desc = "saving predictions: "):
            save_name = save_folder + "/" + str(i) + ".jpg"
            tensor = self.mini_transform(images[i]).unsqueeze(0)
            res = self.model(tensor).detach().cpu().numpy()[0][0]

            if centroid_mode == True:
                res, centroids = generate_centroid_image(res)
                filenames_centroids.append([save_name, centroids])

            res = cv2.resize(res,pred_size)
            cv2.imwrite(save_name, res*255)

        if centroid_mode == True:
            df = pd.DataFrame(filenames_centroids, columns = ["filenames", "centroids"])
            return df
        else:
            return os.listdir(save_folder)
