import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# ---------------- SETTINGS ----------------
MODEL_PATH = "model.h5"
CLASS_NAMES = ["Aphid", "Armyworm", "Bollworm", "Grasshopper", "Mites"]  # <-- CHANGE to your classes
IMG_SIZE = (224, 224)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()


st.set_page_config(page_title="PestVision AI", page_icon="🪲", layout="centered")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #e8f9ee 0%, #ffffff 100%);
        padding-top: 40px;
    }
    [data-testid="stHeader"] {background: rgba(0,0,0,0);}
    .title {font-size: 42px; font-weight: 800; color: #05652d; text-align:center;}
    .subtitle {font-size: 18px; text-align:center; color:#2b2b2b;}
    .bug-icon {font-size: 65px; text-align:center;}
    .upload-label {font-size: 22px; font-weight: 700; color:#05652d;}
    .note {font-size: 15px; opacity:0.8;}
    .footer {text-align:center; margin-top:40px; opacity:0.85;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>PestVision AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Eco-smart Pest Detection powered by Deep Learning</div>", unsafe_allow_html=True)
st.markdown("<div class='bug-icon'>🪲</div>", unsafe_allow_html=True)

st.markdown("<div class='upload-label'>Upload a Pest Image for Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='note'>Choose JPG, JPEG, or PNG</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def process_and_predict(img_file):
    img = image.load_img(img_file, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)  # <--- IMPORTANT
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_id = np.argmax(preds)
    confidence = preds[0][class_id]

    return CLASS_NAMES[class_id], float(confidence)



if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=300)

    with st.spinner("Analyzing image..."):
        pest_name, confidence = process_and_predict(uploaded_file)

    st.success(f"### 🐛 Detected Pest: **{pest_name}**")
    st.info(f"Confidence: **{confidence:.2f}**")

st.markdown("""
<div class='footer'>
PestVision AI combines smart deep learning models with sustainable agriculture
to help farmers detect pest threats early and protect crops efficiently.
</div>
""", unsafe_allow_html=True)
