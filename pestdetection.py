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

# Custom CSS for the CareerLens-like clean layout
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #e8f9ee 0%, #ffffff 100%);
        padding-top: 40px;
    }
    [data-testid="stHeader"] {background: rgba(0,0,0,0);}
    .main-card {
        background: #ffffff;
        border-radius: 25px;
        box-shadow: 0 4px 25px rgba(0,0,0,0.1);
        padding: 50px;
        max-width: 700px;
        margin: auto;
        text-align: center;
    }
    .title {
        font-size: 42px;
        font-weight: 800;
        color: #05652d;
        text-align: center;
        text-shadow: 1px 1px 2px #cce8d3;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        color: #2b2b2b;
        text-align: center;
        margin-bottom: 30px;
    }
    .bug-icon {
        font-size: 65px;
        text-align: center;
        margin: 15px 0 25px 0;
    }
    .upload-label {
        font-size: 22px;
        font-weight: 700;
        color: #05652d;
        margin-bottom: 15px;
    }
    .note {
        font-size: 15px;
        color: #3e3e3e;
        margin-bottom: 15px;
        opacity: 0.8;
    }
    .footer {
        text-align: center;
        font-size: 15px;
        color: #1f4628;
        margin-top: 40px;
        line-height: 1.6;
        opacity: 0.85;
        max-width: 750px;
        margin-left: auto;
        margin-right: auto;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Main content
st.markdown("<div class='main-card'>", unsafe_allow_html=True)

# Title and tagline
st.markdown("<div class='title'>PestVision AI </div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Eco-smart Pest Detection powered by Deep Learning</div>", unsafe_allow_html=True)

# Bug icon below subtitle
st.markdown("<div class='bug-icon'>🪲</div>", unsafe_allow_html=True)

# Upload section
st.markdown("<div class='upload-label'>Upload a Pest Image for Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='note'>Choose an image file (JPG, JPEG, or PNG) of the pest</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# Closing card
st.markdown("</div>", unsafe_allow_html=True)

# ✨ Footer with impressive description
st.markdown("""
<div class='footer'>
PestVision AI combines the power of deep learning with sustainable farming principles to protect crops intelligently. 
By automating pest identification, it empowers farmers and researchers to make faster, eco-friendly decisions — 
reducing pesticide waste, preserving biodiversity, and ensuring a greener, healthier planet for the generations ahead.
</div>
""", unsafe_allow_html=True)
