
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from image_processing import *

st.set_page_config(page_title="Photo Editor", layout="wide")
st.title("📸 Photo Editor using OpenCV and Streamlit")

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    image = np.array(img)

    st.sidebar.header("Adjustments")
    scale = st.sidebar.slider("Resize (%)",10,200,100)
    brightness = st.sidebar.slider("Brightness",-100,100,0)
    contrast = st.sidebar.slider("Contrast",0.5,3.0,1.0,0.1)

    edited = resize_image(image, scale)
    edited = adjust_brightness(edited, brightness)
    edited = adjust_contrast(edited, contrast)

    if st.sidebar.checkbox("Grayscale"):
        edited = grayscale(edited)
    if st.sidebar.checkbox("Blur"):
        edited = blur(edited)
    if st.sidebar.checkbox("Warm Filter"):
        edited = warm_filter(edited)
    if st.sidebar.checkbox("Portrait Blur"):
        edited = portrait_blur(edited)
    if st.sidebar.checkbox("Sharpen"):
        edited = sharpen(edited)
    if st.sidebar.checkbox("Edge Detection"):
        edited = edge_detection(edited)
    if st.sidebar.checkbox("Sketch"):
        edited = sketch(edited)
    if st.sidebar.checkbox("Cartoon"):
        edited = cartoon(edited)
    angle = st.sidebar.slider("Rotation",0,360,0)
    edited = rotate_image(edited, angle)

    c1,c2=st.columns(2)
    c1.image(image, caption="Original", use_container_width=True)
    c2.image(edited, caption="Edited", use_container_width=True)

    out = Image.fromarray(edited)
    import io
    buf=io.BytesIO()
    out.save(buf, format="PNG")
    st.download_button("Download Image", buf.getvalue(), "edited.png","image/png")
