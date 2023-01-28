# Script to do basic inference given an image using onnxruntime
import cv2
import numpy as np
import onnxruntime

onnx_model_path = "mnist_classifier.onnx"
img_path = "sample_digit_7.png"

# Load the onnx model
ort_session = onnxruntime.InferenceSession(
    onnx_model_path, providers=["CPUExecutionProvider"]
)

# Load the image and preprocess it such that it can be passed to the model
img = cv2.imread(img_path)
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
print("Predicted digit is:", predicted_digit)
