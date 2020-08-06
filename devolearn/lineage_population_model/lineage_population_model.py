import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
import torchvision.models as models

import os
import cv2
import wget
from PIL import Image
import joblib
import numpy as np
from collections import deque
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt



"""
ResNet18 to determine population of cells in an embryo
"""

class lineage_population_model():   
    def __init__(self, mode = "cpu"):
        self.mode = mode
        self.model = models.resnet18(pretrained = True)
        self.model.fc = nn.Linear(512, 7)  ## resize last layer
        self.model_dir = os.path.dirname(__file__)
        self.scaler = joblib.load(self.model_dir + "/" + 'scaler/scaler.gz')

        self.model_url = "https://github.com/DevoLearn/devolearn/raw/master/devolearn/lineage_population_model/estimate_lineage_population.pt"
        self.model_name = "estimate_lineage_population.pt"
        # print("at : ", os.path.dirname(__file__))


        try:
            # print("model already downloaded, loading model...")
            self.model.load_state_dict(torch.load(self.model_dir + "/" + self.model_name, map_location= "cpu"))
        except:
            print("model not found, downloading from:", self.model_url)
            filename = wget.download(self.model_url, out= self.model_dir)
            # print(filename)
            self.model.load_state_dict(torch.load(self.model_dir + "/" + self.model_name, map_location= "cpu"))


        self.model.eval()

        self.transforms = transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.Resize((256,256), interpolation = Image.NEAREST),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,), (0.5,))
                                            ])

    def predict(self, image_path):

        """
        input{
            image path <str>
        }

        output{
            dictionary containing the cell population values <dict>
        }

        Loads an image from image_path and converts it to grayscale, 
        then passes it though the model and returns a dictionary 
        with the scaled output (see self.scaler)

        """

        image = cv2.imread(image_path, 0)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        tensor = self.transforms(image).unsqueeze(0)
        
        pred = self.model(tensor).detach().cpu().numpy().reshape(1,-1)

        pred_scaled = (self.scaler.inverse_transform(pred).flatten()).astype(np.uint8)

        pred_dict = {
            "A": pred_scaled[0],
            "E": pred_scaled[1],
            "M": pred_scaled[2],
            "P": pred_scaled[3],
            "C": pred_scaled[4],
            "D": pred_scaled[5],
            "Z": pred_scaled[6]
        }

        return pred_dict

    def predict_from_video(self, video_path, csv_name  = "foo.csv", save_csv = False, ignore_first_n_frames = 0, ignore_last_n_frames = 0):

        """
        inputs{
            video path <str> = path to video file 
            csv_name <str> = filename to be used to save the predictions 
            save_csv <bool> = set to True if you want to save the predictions into a CSV files
            ignore_first_n_frames <int> = number of frames to drop in the start of the video 
            ignore_last_n_frames <int> = number of frames to drop in the end of the video 
        }


        output{
            DataFrame containing all the preds with the corresponding column name <pandas.DataFrame>
        }
        
        Splits a video from video_path into frames and passes the 
        frames through the model for predictions. Saves all the predictions
        into a pandas.DataFrame which can be optionally saved as a CSV file.

        The model was trained to make predictions upto the 
        stage where the population of "A" lineage is 250

        """
        A_population_upper_limit = 250

        vidObj = cv2.VideoCapture(video_path)   
        success = 1
        images = deque()
        count = 0

        preds = deque()

        while success: 
            success, image = vidObj.read() 
            
            try:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                images.append(image)
                
            except:
                print("skipped possible corrupt frame number : ", count)
            count += 1 

        for i in tqdm(range(len(images)), desc='Predicting from video file:  :'):
            tensor = self.transforms(images[i]).unsqueeze(0)
            pred = self.model(tensor).detach().cpu().numpy().reshape(1,-1)
            pred_scaled = (self.scaler.inverse_transform(pred).flatten()).astype(np.uint8)
            preds.append(pred_scaled)

       
        df = pd.DataFrame(preds, columns = ["A", "E", "M", "P", "C", "D", "Z"]) 

        if ignore_first_n_frames != 0:
            df= df.tail(df.shape[0] - ignore_first_n_frames)


        if ignore_last_n_frames != 0:
            df= df.head(df.shape[0] - ignore_last_n_frames)

        a_values = df["A"].values

        for limit in range(len(a_values)):  ## show preds upto 250 A cells 
            if a_values[limit]>=250:
                break
        
        df = df.head(limit)

        if save_csv == True:
            df.to_csv(csv_name, index = False)

        return  df


        
    def create_population_plot_from_video(self, video_path, save_plot = False, plot_name = "plot.png", ignore_first_n_frames = 0, ignore_last_n_frames = 0 ):

        """
        inputs{
            video_path <str> = path to video file 
            save_plot <bool> = set to True to save the plot as an image file 
            plot_name <str> = filename of the plot image to be saved 
            ignore_first_n_frames <int> = number of frames to drop in the start of the video 
            ignore_last_n_frames <int> = number of frames to drop in the end of the video 
        }

        outputs{
            plot object which can be customized further <matplotlib.pyplot>
        }

        plots all the predictions from a video into a matplotlib.pyplot 
        
        """
        df = self.predict_from_video(video_path, ignore_first_n_frames = ignore_first_n_frames, ignore_last_n_frames = ignore_last_n_frames )  
        
        labels = ["A", "E", "M", "P", "C", "D", "Z"]

        for label in labels:
            plt.plot(df[label].values, label = label)

        plt.xlabel("frames")
        plt.ylabel("population")

        if save_plot == True:
            plt.legend()
            
            plt.savefig(plot_name)

        return plt