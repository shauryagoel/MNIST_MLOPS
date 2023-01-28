# FastAPI server to identify the digit present in an image
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile

import utils

config = utils.load_yaml_config(Path("../configs/serve.yaml"))
onnx_model_path = config["onnx_model_path"]

ort_session = utils.load_onnx_model(Path(onnx_model_path))

app = FastAPI()


@app.post("/infer/")
async def process_video(image: UploadFile = File(...)):
    """POST endpoint which accepts an image of a digit as input and predicts the digit.
    Response:
        {"digit" : <integer representing predicted digit>}
    """
    # Load the image and preprocess it such that it can be passed to the model
    img = await image.read()
    image = np.fromstring(img, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    pre_processed_image = utils.pre_process_image(image)

    # Do actual onnxruntime inference
    ort_outs = utils.ort_inference(ort_session, pre_processed_image)
    predicted_digit = np.argmax(ort_outs[0]).item()
    return {"digit": predicted_digit}
