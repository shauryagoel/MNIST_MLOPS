# Script to do basic inference given an image using onnxruntime
import logging
from pathlib import Path

import cv2
import numpy as np

import utils

# Set logging level to info
logging.basicConfig(level=logging.INFO)

# Load config
config = utils.load_yaml_config(Path("inference.yaml"))
onnx_model_path = config["onnx_model_path"]
image_path = config["image_path"]

# Load the onnx model
ort_session = utils.load_onnx_model(Path(onnx_model_path))

# Load the image and preprocess it such that it can be passed to the model
image = cv2.imread(image_path)
pre_processed_image = utils.pre_process_image(image)

# Do actual onnxruntime inference
ort_outs = utils.ort_inference(ort_session, pre_processed_image)

# Take the max index of predictions
predicted_digit = np.argmax(ort_outs[0])
logging.info("Predicted digit is: %s", predicted_digit)
