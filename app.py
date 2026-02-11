import streamlit as st
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas

st.title("Handwritten Digit Recognition (Scratch DNN)")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

def load_model():
    data = np.load("model_weights.npz")
    return data["W1"], data["b1"], data["W2"], data["b2"], data["W3"], data["b3"]

W1, b1, W2, b2, W3, b3 = load_model()

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def predict(image):
    image = image.reshape(1, 784)

    Z1 = image @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)

    Z3 = A2 @ W3 + b3
    A3 = softmax(Z3)

    return np.argmax(A3), np.max(A3)

if canvas_result.image_data is not None:

    img = canvas_result.image_data
    img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGBA2GRAY)
    img = cv2.resize(img, (28, 28))
    img = 255 - img
    img = img / 255.0

    digit, confidence = predict(img)

    st.image(img, width=150)
    st.write(f"### Predicted Digit: {digit}")
    st.write(f"Confidence: {confidence:.4f}")
