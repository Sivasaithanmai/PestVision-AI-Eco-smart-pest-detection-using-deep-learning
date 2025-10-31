import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2
import tempfile
import json

# Page config
st.set_page_config(page_title="PestVision AI", page_icon="🪲", layout="centered")

# Custom CSS for clean layout
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #e8f9ee 0%, #ffffff 100%);
    }
    [data-testid="stHeader"] {background: rgba(0,0,0,0);}
    .title {
        font-size: 50px;
        font-weight: 800;
        color: #05652d;
        text-align: center;
        text-shadow: 1px 1px 2px #cce8d3;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 20px;
        color: #2b2b2b;
        text-align: center;
        margin-bottom: 20px;
    }
    .bug-icon {
        font-size: 60px;
        text-align: center;
        margin: 20px 0;
    }
    .upload-section {
        background-color: #ffffff;
        border: 2px solid #bfe8c6;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        max-width: 500px;
        margin: auto;
        box-shadow: 0px 4px 18px rgba(0,0,0,0.08);
    }
    .upload-label {
        font-size: 22px;
        font-weight: 700;
        color: #05652d;
        margin-bottom: 10px;
    }
    .footer {
        text-align: center;
        font-size: 15px;
        color: #1f4628;
        margin-top: 30px;
        opacity: 0.8;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<div class='title'>PestVision AI </div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Eco-smart Pest Detection powered by Deep Learning</div>", unsafe_allow_html=True)

# Bug icon
st.markdown("<div class='bug-icon'>🪲</div>", unsafe_allow_html=True)

# Upload file section
st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
st.markdown("<div class='upload-label'>Upload a pest image for detection</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>This project demonstrates AI supporting sustainable agriculture by detecting and classifying crop pests efficiently.</div>", unsafe_allow_html=True)
