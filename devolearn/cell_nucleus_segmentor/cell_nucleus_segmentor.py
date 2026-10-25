import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage

import os
import cv2
import wget
import imutils
from tqdm import tqdm, tqdm_notebook
from PIL import Image
import numpy as np
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import warnings
warnings.filterwarnings("ignore") 

from ..base_inference_engine import InferenceEngine


class cell_nucleus_segmentor(InferenceEngine):
    def __init__(self, device = "cpu"):
        """Segments the c. elegans embryo from images/videos, 
        depends on segmentation-models-pytorch for the model backbone

        Args:
            device (str, optional): set to "cuda", runs operations on gpu and set to "cpu", runs operations on cpu. Defaults to "cpu".
        """
        
        self.device = device
        self.ENCODER = 'resnet18'
        self.ENCODER_WEIGHTS = 'imagenet'
        self.CLASSES = ["nucleus"]
        self.ACTIVATION = 'sigmoid'
        self.in_channels = 1
        self.model_url = "https://github.com/DevoLearn/devolearn/raw/master/devolearn/cell_nucleus_segmentor/cell_nucleus_segmentation_model.pth"
        self.model_name = "cell_nucleus_segmentation_model.pth"
        self.model_dir = os.path.dirname(__file__)
        # print("at : ", os.path.dirname(__file__))

        self.model = smp.FPN(
                encoder_name= self.ENCODER, 
                encoder_weights= self.ENCODER_WEIGHTS, 
                classes=len(self.CLASSES), 
                activation= self.ACTIVATION,
                in_channels = self.in_channels 
            )

        self.download_checkpoint()
        self.model.to(self.device)
        self.model.eval()

        self.mini_transform = transforms.Compose([
                                     transforms.ToPILImage(),
                                     transforms.Resize((256,256), interpolation = Image.NEAREST),
                                     transforms.ToTensor(),
                                    ])


    def download_checkpoint(self):
        if hasattr(smp, "decoders") and hasattr(smp.decoders, "fpn"):
            import sys
            if "segmentation_models_pytorch.fpn" not in sys.modules:
                sys.modules["segmentation_models_pytorch.fpn"] = smp.decoders.fpn
            if "segmentation_models_pytorch.fpn.model" not in sys.modules:
                sys.modules["segmentation_models_pytorch.fpn.model"] = smp.decoders.fpn.model
        try:
            # print("model already downloaded, loading model...")
            self.model = torch.load(self.model_dir + "/" + self.model_name, map_location= self.device) 
        except:
            print("model not found, downloading from:", self.model_url)
            if os.path.isdir(self.model_dir) == False:
                os.mkdir(self.model_dir)
            filename = wget.download(self.model_url, out= self.model_dir)
            # print(filename)
            self.model = torch.load(self.model_dir + "/" + self.model_name, map_location= self.device) 

    def preprocess(self, image_grayscale_numpy):

        tensor = self.mini_transform(image_grayscale_numpy).unsqueeze(0).to(self.device)
        return tensor

    def predict(self, image_path = None, pred_size = (350,250), image = None):
        """
        Loads an image from image_path or in-memory array/PIL image and converts it to grayscale, 
        then passes it through the model and returns centroids of the segmented features.
        reference{
            https://github.com/DevoLearn/devolearn#segmenting-the-c-elegans-embryo
        }

        Args:
            image_path (str, optional): path to image. Defaults to None.
            pred_size (tuple, optional): size of output image,(width,height). Defaults to (350,250).
            image (str, np.ndarray, PIL.Image.Image, optional): path or in-memory image array. Defaults to None.

        Returns:
            np.array : 1 channel image.

        """

        im = self.load_image(image if image is not None else image_path, to_grayscale=True)
        tensor = self.preprocess(im)

        # The model has issues with the latest PyTorch versions, which causes
        # AttributeError: 'Upsample' object has no attribute 'recompute_scale_factor'
        # To avoid this, we manually set the recompute_scale_factor attribute to None
        self.model.segmentation_head[1].recompute_scale_factor = None
        
        res = self.model(tensor).detach().cpu().numpy()[0][0]
        res = cv2.resize(res,pred_size)
        return res
