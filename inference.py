# Script to do basic inference given an image using onnxruntime
import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

from utils import load_yaml_config

# Set logging level to info
logging.basicConfig(level=logging.INFO)

# Load config
config = load_yaml_config(Path("inference.yaml"))
onnx_model_path = config["onnx_model_path"]
image_path = config["image_path"]

# Load the onnx model
ort_session = onnxruntime.InferenceSession(
    onnx_model_path, providers=["CPUExecutionProvider"]
)

# Load the image and preprocess it such that it can be passed to the model
img = cv2.imread(image_path)
img = cv2.resize(img, (28, 28))
# convert from RGB to grayscale as model takes grayscale images
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
x = img.astype(np.float32)
x = x[None]
x = x.reshape(1, 28 * 28)
x = np.ascontiguousarray(x)


# Do actual onnxruntime inference
ort_inputs = {ort_session.get_inputs()[0].name: x}
ort_outs = ort_session.run(None, ort_inputs)


# Take the max index of predictions
predicted_digit = np.argmax(ort_outs[0])
logging.info("Predicted digit is: %s", predicted_digit)
