import os
import cv2
import numpy as np
from PIL import Image

class InferenceEngine():
    def __init__(self, model=None):
        self.model = model
        
    def download_checkpoint(self):
        raise NotImplementedError

    def load_checkpoint(self):
        raise NotImplementedError
    
    def preprocess(self, x):
        raise NotImplementedError

    def deprocess(self, x):
        raise NotImplementedError

    def inference(self, x):
        raise NotImplementedError

    @staticmethod
    def load_image(image_input, to_grayscale=True):
        """Loads and standardizes an image from a file path, numpy array, or PIL Image.

        Args:
            image_input (str, os.PathLike, np.ndarray, PIL.Image.Image): Input image or path to image.
            to_grayscale (bool, optional): Whether to convert the image to 1-channel grayscale. Defaults to True.

        Returns:
            np.ndarray: Processed image array (2D uint8 if to_grayscale=True, or 3D uint8 BGR if False).
        """
        if image_input is None:
            raise ValueError("No image input provided. Expected a file path, numpy array, or PIL Image.")

        if isinstance(image_input, (str, os.PathLike)):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            if to_grayscale:
                im = cv2.imread(str(image_input), 0)
            else:
                im = cv2.imread(str(image_input))
            if im is None:
                raise ValueError(f"Failed to read image from path: {image_input}")
            return im
        elif isinstance(image_input, np.ndarray):
            im = image_input.copy()
            if np.issubdtype(im.dtype, np.floating) and im.max() <= 1.0:
                im = (im * 255).astype(np.uint8)
            elif im.dtype != np.uint8:
                im = im.astype(np.uint8)

            if to_grayscale:
                if im.ndim == 2:
                    return im
                elif im.ndim == 3:
                    if im.shape[2] == 1:
                        return im.squeeze(axis=-1)
                    elif im.shape[2] == 3:
                        return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    elif im.shape[2] == 4:
                        return cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY)
                    else:
                        raise ValueError(f"Unsupported number of channels in image array: {im.shape[2]}")
                else:
                    raise ValueError(f"Unsupported array dimensions: {im.ndim}")
            else:
                if im.ndim == 2:
                    return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                elif im.ndim == 3:
                    if im.shape[2] == 1:
                        return cv2.cvtColor(im.squeeze(axis=-1), cv2.COLOR_GRAY2BGR)
                    elif im.shape[2] in (3, 4):
                        return im
                    else:
                        raise ValueError(f"Unsupported number of channels in image array: {im.shape[2]}")
                else:
                    raise ValueError(f"Unsupported array dimensions: {im.ndim}")
        elif isinstance(image_input, Image.Image):
            if to_grayscale:
                return np.array(image_input.convert('L'))
            else:
                rgb_arr = np.array(image_input.convert('RGB'))
                return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        else:
            raise TypeError(
                f"Expected image file path (str), numpy array (np.ndarray), or PIL Image, got {type(image_input)}"
            )

    def __repr__(self):
        return str(self.model) if self.model is not None else super().__repr__()
