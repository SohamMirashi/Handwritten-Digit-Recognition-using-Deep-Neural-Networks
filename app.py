import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import cv2

st.title("Handwritten Digit Recognition")

# Canvas settings
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

if canvas_result.image_data is not None:

    img = canvas_result.image_data

    # Convert RGBA → Grayscale
    img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGBA2GRAY)

    # Resize to 28x28
    img = cv2.resize(img, (28, 28))

    # Invert colors (white background → black background like MNIST)
    img = 255 - img

    # Normalize
    img = img / 255.0

    st.image(img, caption="Processed 28x28 Image", width=150)
