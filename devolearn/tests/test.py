import unittest
from unittest import TestCase
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
import sys
import pytest

import warnings
warnings.filterwarnings("ignore") 

# Compatibility with newer segmentation_models_pytorch package layout
try:
    import segmentation_models_pytorch as smp
    if hasattr(smp, "decoders") and hasattr(smp.decoders, "fpn"):
        if "segmentation_models_pytorch.fpn" not in sys.modules:
            sys.modules["segmentation_models_pytorch.fpn"] = smp.decoders.fpn
        if "segmentation_models_pytorch.fpn.model" not in sys.modules:
            sys.modules["segmentation_models_pytorch.fpn.model"] = smp.decoders.fpn.model
except Exception:
    pass

from devolearn import lineage_population_model
from devolearn import Generator, embryo_generator_model
from devolearn import cell_membrane_segmentor
from devolearn import cell_nucleus_segmentor


class test(unittest.TestCase):

    @pytest.mark.lineage_population_model
    def test_lineage_population_model(self):
        test_dir = os.path.dirname(__file__)
        expected_lineages = ["A", "E", "M", "P", "C", "D", "Z"]

        model = lineage_population_model(device = "cpu")
        pred = model.predict(image_path = test_dir + "/" + "sample_data/images/embryo_sample.png")
        
        # Check output type and schema
        self.assertTrue(isinstance(pred, dict), "should be dict")
        self.assertEqual(sorted(list(pred.keys())), sorted(expected_lineages), "all lineage keys should be present")

        # Check output dtypes and value validity
        for k, v in pred.items():
            self.assertTrue(isinstance(v, (int, np.integer)), f"lineage count {k} should be integer dtype")
            self.assertGreaterEqual(v, 0, f"lineage population for {k} must be non-negative")

        # Reference population checks on sample image
        self.assertGreater(pred["A"], 0, "A lineage count should be non-zero on sample image")
        self.assertLessEqual(pred["A"], 250, "A lineage count should not exceed model upper limit")

        # Video prediction tests
        video_pred = model.predict_from_video(
            video_path = test_dir + "/" + "sample_data/videos/embryo_timelapse.mov",
            save_csv = False,
            csv_name = "sample_preds/video_preds.csv",
            ignore_first_n_frames = 10,
            ignore_last_n_frames = 10
        )
        self.assertTrue(isinstance(video_pred, pd.DataFrame), "should be pandas.DataFrame")
        self.assertEqual(list(video_pred.columns), expected_lineages, "DataFrame columns should match lineage labels")
        self.assertGreater(len(video_pred), 0, "video predictions DataFrame should not be empty")
        self.assertFalse(video_pred.isnull().values.any(), "predictions should not contain NaN/null values")
        self.assertTrue((video_pred.values >= 0).all(), "all predicted lineage populations should be non-negative")
        self.assertTrue((video_pred["A"].values <= 250).all(), "A lineage population should remain within upper bound")

        plot = model.create_population_plot_from_video(
            video_path = test_dir + "/" + "sample_data/videos/embryo_timelapse.mov",
            save_plot = False,
            plot_name = "sample_preds/plot.png",
            ignore_last_n_frames = 0
        )
        self.assertTrue(isinstance(plot, type(plt)), "should be matplotlib.pyplot")

    @pytest.mark.embryo_generator_model   
    def test_embryo_generator_model(self):
        test_dir = os.path.dirname(__file__)

        generator = embryo_generator_model()
        gen_image = generator.generate()  ## 2d numpy array 

        # Check type, dimensionality, shape, and dtype
        self.assertTrue(isinstance(gen_image, np.ndarray), "should be numpy.ndarray")
        self.assertEqual(gen_image.ndim, 2, "generated image should be 2-dimensional")
        self.assertEqual(gen_image.shape, (500, 700), "generated image default shape should match (500, 700)")
        self.assertTrue(
            np.issubdtype(gen_image.dtype, np.floating) or np.issubdtype(gen_image.dtype, np.integer),
            "generated image should have float or int dtype"
        )
        self.assertFalse(np.isnan(gen_image).any(), "generated image should not contain NaN values")
        self.assertFalse(np.isinf(gen_image).any(), "generated image should not contain Infinite values")

        self.assertTrue(
            isinstance(generator.generate_n_images(n = 1, foldername= test_dir + "/" + "generated_images", image_size= (700,500)), type(None)),
            "should return None without errors"
        )

    @pytest.mark.cell_membrane_segmentor
    def test_cell_membrane_segmentor(self):
        test_dir = os.path.dirname(__file__)
        pred_width, pred_height = 350, 250

        segmentor = cell_membrane_segmentor()

        # 1. Prediction without centroid_mode
        seg_pred = segmentor.predict(
            image_path = test_dir + "/" + "sample_data/images/seg_sample.jpg",
            pred_size = (pred_width, pred_height),
            centroid_mode = False
        )
        self.assertTrue(isinstance(seg_pred, np.ndarray), "should be numpy.ndarray")
        self.assertEqual(seg_pred.shape, (pred_height, pred_width), "output shape should match pred_size (H, W)")
        self.assertTrue(np.issubdtype(seg_pred.dtype, np.floating), "segmentation output should have floating dtype")
        self.assertGreaterEqual(float(seg_pred.min()), 0.0, "probabilities should be >= 0.0")
        self.assertLessEqual(float(seg_pred.max()), 1.0, "probabilities should be <= 1.0")
        self.assertFalse(np.isnan(seg_pred).any(), "output must not contain NaN")
        self.assertGreater(float(seg_pred.max()), 0.5, "model should detect segmented membrane with confidence > 0.5")

        # 2. Prediction with centroid_mode
        seg_pred, centroids = segmentor.predict(
            image_path = test_dir + "/" + "sample_data/images/seg_sample.jpg",
            pred_size = (pred_width, pred_height),
            centroid_mode = True
        )
        self.assertTrue(isinstance(seg_pred, np.ndarray), "should be numpy.ndarray")
        self.assertTrue(isinstance(centroids, list), "centroids should be a list")
        self.assertGreater(len(centroids), 0, "centroids list should not be empty for sample image")

        for c in centroids:
            self.assertEqual(len(c), 2, "each centroid must be a coordinate pair (x, y)")
            cX, cY = c
            self.assertTrue(0 <= cX <= pred_width, f"centroid X coordinate {cX} out of bounds")
            self.assertTrue(0 <= cY <= pred_height, f"centroid Y coordinate {cY} out of bounds")

        # 3. Video prediction
        filenames = segmentor.predict_from_video(
            video_path = test_dir + "/" + "sample_data/videos/seg_sample.mov",
            pred_size = (pred_width, pred_height),
            centroid_mode = False,
            save_folder = test_dir + "/" + "preds"
        )
        self.assertTrue(isinstance(filenames, list), "should be a list")
        self.assertGreater(len(filenames), 0, "should save predictions for video frames")

        centroid_df = segmentor.predict_from_video(
            video_path = test_dir + "/" + "sample_data/videos/seg_sample.mov",
            pred_size = (pred_width, pred_height),
            centroid_mode = True,
            save_folder = test_dir + "/" + "preds"
        )
        self.assertTrue(isinstance(centroid_df, pd.DataFrame), "should be a pandas.DataFrame")
        self.assertEqual(list(centroid_df.columns), ["filenames", "centroids"], "DataFrame columns must match")
        self.assertGreater(len(centroid_df), 0, "DataFrame should contain frame entries")

    @pytest.mark.cell_nucleus_segmentor
    def test_cell_nucleus_segmentor(self):
        test_dir = os.path.dirname(__file__)
        pred_width, pred_height = 350, 250

        segmentor = cell_nucleus_segmentor()
        seg_pred = segmentor.predict(
            image_path = test_dir + "/" + "sample_data/images/nucleus_seg_sample.png",
            pred_size = (pred_width, pred_height)
        )

        self.assertTrue(isinstance(seg_pred, np.ndarray), "should be numpy.ndarray")
        self.assertEqual(seg_pred.shape, (pred_height, pred_width), "output shape should match pred_size (H, W)")
        self.assertTrue(np.issubdtype(seg_pred.dtype, np.floating), "output should have floating dtype")
        self.assertGreaterEqual(float(seg_pred.min()), 0.0, "segmentation probabilities should be >= 0.0")
        self.assertLessEqual(float(seg_pred.max()), 1.0, "segmentation probabilities should be <= 1.0")
        self.assertFalse(np.isnan(seg_pred).any(), "output must not contain NaN")
        self.assertGreater(float(seg_pred.max()), 0.5, "model should detect cell nuclei with confidence > 0.5")


if __name__ == '__main__':
    unittest.main()
