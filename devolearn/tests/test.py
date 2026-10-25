import unittest
from unittest import TestCase
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
import pytest

import warnings
warnings.filterwarnings("ignore") 

from devolearn import lineage_population_model
from devolearn import Generator, embryo_generator_model
from devolearn import cell_membrane_segmentor
from devolearn import cell_nucleus_segmentor


import os
import cv2


class test(unittest.TestCase):

    @pytest.mark.lineage_population_model
    def test_lineage_population_model(self):
        test_dir = os.path.dirname(__file__)

        model = lineage_population_model(device = "cpu")
        pred = model.predict(image_path = test_dir + "/" + "sample_data/images/embryo_sample.png")
        self.assertTrue(isinstance(pred, dict), "should be dict")

        # Test prediction with numpy array (local variable)
        img_np = cv2.imread(test_dir + "/" + "sample_data/images/embryo_sample.png")
        pred_np = model.predict(img_np)
        self.assertTrue(isinstance(pred_np, dict), "should be dict")
        pred_np_kw = model.predict(image = img_np)
        self.assertTrue(isinstance(pred_np_kw, dict), "should be dict")

        video_pred = model.predict_from_video(video_path = test_dir + "/" + "sample_data/videos/embryo_timelapse.mov", save_csv = False, csv_name = "sample_preds/video_preds.csv", ignore_first_n_frames= 10, ignore_last_n_frames= 10 )
        self.assertTrue(isinstance(video_pred, pd.DataFrame), "should be dict")

        plot = model.create_population_plot_from_video(video_path = test_dir + "/" + "sample_data/videos/embryo_timelapse.mov", save_plot= False, plot_name= "sample_preds/plot.png", ignore_last_n_frames= 0 )
        self.assertTrue(isinstance(plot, type(plt)), "should be matplotlib.pyplot")

    @pytest.mark.embryo_generator_model   
    def test_embryo_generator_model(self):
        test_dir = os.path.dirname(__file__)

        generator = embryo_generator_model()
        gen_image = generator.generate()  ## 2d numpy array 
        self.assertTrue(isinstance(gen_image, np.ndarray), "should be dict")
        self.assertTrue(isinstance(generator.generate_n_images(n = 1, foldername= test_dir + "/" + "generated_images", image_size= (700,500)), type(None)), "should return None without errors")

    @pytest.mark.cell_membrane_segmentor
    def test_cell_membrane_segmentor(self):
        test_dir = os.path.dirname(__file__)

        segmentor = cell_membrane_segmentor()
        seg_pred = segmentor.predict(image_path = test_dir + "/" + "sample_data/images/seg_sample.jpg", centroid_mode =False )
        self.assertTrue(isinstance(seg_pred, np.ndarray), "should be numpy.ndarray")

        seg_pred, centroids = segmentor.predict(image_path = test_dir + "/" + "sample_data/images/seg_sample.jpg", centroid_mode = True )
        self.assertTrue(isinstance(seg_pred, np.ndarray), "should be numpy.ndarray")
        self.assertTrue(isinstance(centroids, list), "should be a list")

        # Test prediction with numpy array (local variable)
        img_np = cv2.imread(test_dir + "/" + "sample_data/images/seg_sample.jpg")
        seg_pred_np = segmentor.predict(img_np, centroid_mode = False)
        self.assertTrue(isinstance(seg_pred_np, np.ndarray), "should be numpy.ndarray")
        seg_pred_np_kw = segmentor.predict(image = img_np, centroid_mode = False)
        self.assertTrue(isinstance(seg_pred_np_kw, np.ndarray), "should be numpy.ndarray")

        filenames = segmentor.predict_from_video(video_path = test_dir + "/" + "sample_data/videos/seg_sample.mov", centroid_mode = False, save_folder = test_dir + "/" + "preds")
        self.assertTrue(isinstance(filenames, list), "should be a list")

        centroid_df = segmentor.predict_from_video(video_path = test_dir + "/" +  "sample_data/videos/seg_sample.mov", centroid_mode = True, save_folder = test_dir + "/" + "preds")
        self.assertTrue(isinstance(centroid_df, pd.DataFrame), "should be a pandas.DataFrame")

    @pytest.mark.cell_nucleus_segmentor
    def test_cell_nucleus_segmentor(self):
        test_dir = os.path.dirname(__file__)

        segmentor = cell_nucleus_segmentor()
        seg_pred = segmentor.predict(image_path = test_dir + "/" + "sample_data/images/nucleus_seg_sample.png")

        #cv2.imwrite(test_dir+"/generated_images/nucleus_seg_inference.png", seg_pred*255)
        self.assertTrue(isinstance(seg_pred, np.ndarray), "should be numpy.ndarray")

        # Test prediction with numpy array (local variable)
        img_np = cv2.imread(test_dir + "/" + "sample_data/images/nucleus_seg_sample.png")
        seg_pred_np = segmentor.predict(img_np)
        self.assertTrue(isinstance(seg_pred_np, np.ndarray), "should be numpy.ndarray")
        seg_pred_np_kw = segmentor.predict(image = img_np)
        self.assertTrue(isinstance(seg_pred_np_kw, np.ndarray), "should be numpy.ndarray")

    def test_predict_numpy_and_pil_inputs(self):
        """Test predicting on in-memory numpy arrays (2D, 3D), float arrays, and PIL Images."""
        test_dir = os.path.dirname(__file__)
        img_path = os.path.join(test_dir, "sample_data", "images", "seg_sample.jpg")
        img_bgr = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_float = img_gray.astype(np.float32) / 255.0
        from PIL import Image
        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        segmentor = cell_membrane_segmentor()

        # 1. Test string representation fix
        self.assertTrue(isinstance(repr(segmentor), str), "repr should return str")

        # 2. Test 2D grayscale array
        pred_2d = segmentor.predict(img_gray)
        self.assertTrue(isinstance(pred_2d, np.ndarray))

        # 3. Test 3D BGR array
        pred_3d = segmentor.predict(img_bgr)
        self.assertTrue(isinstance(pred_3d, np.ndarray))

        # 4. Test float array in [0, 1]
        pred_float = segmentor.predict(img_float)
        self.assertTrue(isinstance(pred_float, np.ndarray))

        # 5. Test PIL Image
        pred_pil = segmentor.predict(img_pil)
        self.assertTrue(isinstance(pred_pil, np.ndarray))

        # 6. Test centroid mode with array input
        pred_c, centroids = segmentor.predict(img_bgr, centroid_mode=True)
        self.assertTrue(isinstance(pred_c, np.ndarray))
        self.assertTrue(isinstance(centroids, list))

        # 7. Test invalid inputs
        with self.assertRaises(ValueError):
            segmentor.predict()

        with self.assertRaises(TypeError):
            segmentor.predict(12345)

if __name__ == '__main__':

    unittest.main()
