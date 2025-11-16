import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

MODEL_PATH = "pest_model.keras"
CLASS_NAMES = ["Aphid", "Armyworm", "Bollworm", "Grasshopper", "Mites"]
IMG_SIZE = (224, 224)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.set_page_config(page_title="PestVision AI", page_icon="🪲", layout="centered")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {background: linear-gradient(180deg, #e8f9ee 0%, #ffffff 100%); padding-top: 40px;}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}
.main-card {background: #ffffff; border-radius: 25px; box-shadow: 0 4px 25px rgba(0,0,0,0.1); padding: 50px; max-width: 700px; margin: auto; text-align: center;}
.title {font-size: 42px; font-weight: 800; color: #05652d; text-align: center; text-shadow: 1px 1px 2px #cce8d3; margin-bottom: 10px;}
.subtitle {font-size: 18px; color: #2b2b2b; text-align: center; margin-bottom: 30px;}
.bug-icon {font-size: 65px; text-align: center; margin: 15px 0 25px 0;}
.upload-label {font-size: 22px; font-weight: 700; color: #05652d; margin-bottom: 15px;}
.note {font-size: 15px; color: #3e3e3e; margin-bottom: 15px; opacity: 0.8;}
.footer {text-align: center; font-size: 15px; color: #1f4628; margin-top: 40px; line-height: 1.6; opacity: 0.85; max-width: 750px; margin-left: auto; margin-right: auto;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>PestVision AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Eco-smart Pest Detection powered by Deep Learning</div>", unsafe_allow_html=True)
st.markdown("<div class='bug-icon'>🪲</div>", unsafe_allow_html=True)
st.markdown("<div class='upload-label'>Upload a Pest Image</div>", unsafe_allow_html=True)
st.markdown("<div class='note'>Choose JPG, JPEG, or PNG</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

def process_and_predict(img_file):
    img = image.load_img(img_file, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    class_id = np.argmax(preds)
    confidence = preds[0][class_id]
    return CLASS_NAMES[class_id], float(confidence)

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=300)
    with st.spinner("Analyzing image..."):
        pest_name, confidence = process_and_predict(uploaded_file)
    st.success(f"### 🐛 Detected Pest: {pest_name}")
    st.info(f"Confidence: {confidence:.2f}")

st.markdown("""
<div class='footer'>
PestVision AI combines smart deep learning with sustainable farming to help protect crops intelligently.
</div>
""", unsafe_allow_html=True)
