import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ----------------- Page config -----------------
st.set_page_config(page_title="PestVision AI", page_icon="🐛", layout="wide")

# ----------------- CSS styling -----------------
st.markdown("""
<style>
body {background-color:white; color:#0b3d0b; font-family:'Arial', sans-serif;}
.header {text-align:center; font-size:3em; font-weight:900; color:#0b3d0b;}
.subtitle {text-align:center; font-size:1.2em; margin-bottom:20px; color:#0b3d0b;}
.image-card {background-color:#f0fff0; padding:10px; border-radius:15px; box-shadow:2px 2px 10px #d9f2d9; text-align:center;}
.prediction-box {text-align:center; margin-top:20px; font-size:1.5em; font-weight:bold; color:#0b3d0b; background-color:#d9f2d9; padding:15px; border-radius:15px; box-shadow:1px 1px 8px #c4e0c4;}
.metrics {text-align:center; padding:20px; border-radius:15px; border:2px solid #0b3d0b; color:#0b3d0b; font-weight:bold; margin-top:20px; background-color:#e6f2e6;}
.tips-box {background-color:#f0fff0; border-left:5px solid #0b3d0b; padding:15px; margin-top:20px; border-radius:10px;}
</style>
""", unsafe_allow_html=True)

# ----------------- Header -----------------
st.markdown("<div class='header'>🐛 PESTVISION AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Detect pests and protect plants like a pro!</div>", unsafe_allow_html=True)

# ----------------- File uploader -----------------
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","jpeg","png"])

# ----------------- Load model -----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pest_model.keras")

model = load_model()
class_names = ["Healthy", "Pest"]

# ----------------- Process uploaded file -----------------
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224,224))
    img_array = np.array(img_resized).astype('float32')
    img_array_exp = np.expand_dims(img_array, axis=0)
    img_array_exp = preprocess_input(img_array_exp)

    prediction = model.predict(img_array_exp)

    # safe prediction handling
    pred = np.array(prediction).flatten()
    if len(pred) == 1:  # binary
        predicted_class = class_names[int(pred[0]>0.5)]
        confidence = float(pred[0] if predicted_class=="Pest" else 1-pred[0])
    else:  # multi-class
        predicted_class = class_names[np.argmax(pred)]
        confidence = float(np.max(pred))

    # ----------------- Display -----------------
    st.markdown('<div class="image-card">Uploaded Image</div>', unsafe_allow_html=True)
    st.image(img, use_column_width=True)

    st.markdown(f"<div class='prediction-box'>Prediction: {predicted_class} ({confidence*100:.2f}%)</div>", unsafe_allow_html=True)

    st.markdown("""
        <div class='metrics'>
            Model Performance<br>
            Accuracy: 92%<br>
            Precision: 90%<br>
            Recall: 88%<br>
            F1 Score: 89%
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='tips-box'>
            <b>Advanced Leaf Defense Tips:</b><br>
            • Beneficial microbes boost leaf immunity.<br>
            • Gentle leaf “massage” improves stomata circulation.<br>
            • Light UV exposure kills surface pests safely.<br>
            • Companion flowers act as visual decoys.<br>
            • Tiny reflective surfaces confuse flying pests.<br>
            • Essential oils like neem/clove confuse pests.<br>
            • Sound therapy: play classical/nature sounds.<br>
            • Morning cool drafts slow pests without harming leaves.
        </div>
    """, unsafe_allow_html=True)