# Hold some basic utility functions
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
import yaml


def load_yaml_config(file_path: Path):
    """Load the yaml config file."""
    with open(file_path, "r") as f:
        config = yaml.load(f, yaml.Loader)
    return config


def load_onnx_model(onnx_model_file_path: Path):
    """Load the onnx model from the file path on CPU."""
    ort_session = onnxruntime.InferenceSession(
        str(onnx_model_file_path), providers=["CPUExecutionProvider"]
    )
    return ort_session


def pre_process_image(image: np.ndarray):
    """Preprocess the image according to the trained model.
    Args:
        image: numpy array which represents image in BGR format
    """
    x = cv2.resize(image, (28, 28))
    # Convert from RGB to grayscale as model takes grayscale images
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x = x.astype(np.float32)
    x = x[None]
    x = x.reshape(1, 28 * 28)
    x = np.ascontiguousarray(x)
    return x


def ort_inference(onnx_model, image: np.ndarray):
    """Do onnx inference on image using onnx_model.
    Args:
        onnx_model: onnxruntime session denoting the model
        image: numpy array which has shape (1, 784) and data type float32
    """
    ort_inputs = {onnx_model.get_inputs()[0].name: image}
    ort_outs = onnx_model.run(None, ort_inputs)
    return ort_outs
