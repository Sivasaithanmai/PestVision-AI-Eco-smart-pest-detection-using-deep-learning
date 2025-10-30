import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2
import tempfile
import json

st.set_page_config(page_title="PestVision AI", page_icon="🪲", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #e8f9ee 0%, #ffffff 100%);
        }
        [data-testid="stHeader"] {background: rgba(0,0,0,0);}
        .title-container {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            margin-top: -10px;
        }
        .bug-icon {
            font-size: 46px;
            margin-right: 8px;
        }
        .title {
            font-size: 46px;
            font-weight: 800;
            color: #05652d;
            text-shadow: 1px 1px 2px #cce8d3;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #2b2b2b;
            margin-bottom: 25px;
        }
        .upload-box {
            background-color: #ffffff;
            padding: 35px;
            border-radius: 20px;
            border: 2px solid #bfe8c6;
            box-shadow: 0px 4px 18px rgba(0,0,0,0.08);
            transition: 0.3s;
            text-align: center;
        }
        .upload-box:hover {
            box-shadow: 0px 6px 25px rgba(0, 100, 0, 0.2);
            transform: scale(1.02);
        }
        .upload-label {
            font-weight: 700;
            color: #05652d;
            font-size: 20px;
            margin-bottom: 15px;
        }
        .result-box {
            background-color: #f0fbf2;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            box-shadow: 0px 3px 12px rgba(0,0,0,0.1);
            margin-top: 25px;
        }
        .prediction {
            font-size: 28px;
            font-weight: 700;
            color: #05652d;
        }
        .confidence {
            font-size: 18px;
            color: #087f39;
        }
        .gradcam-title {
            font-size: 22px;
            font-weight: 600;
            color: #05652d;
            text-align: center;
            margin-top: 30px;
        }
        .footer {
            text-align: center;
            color: #1f4628;
            font-size: 15px;
            margin-top: 40px;
            opacity: 0.8;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class='title-container'>
        <div class='bug-icon'>🪲</div>
        <div class='title'>PestVision AI</div>
    </div>
""", unsafe_allow_html=True)

st.markdown("<p class='subtitle'>Eco-smart Pest Detection powered by Deep Learning</p>", unsafe_allow_html=True)

MODEL_PATH = "pest_model.keras"
MAPPING_PATH = "pest_class_mapping.json"

@st.cache_resource
def load_model_and_mapping():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(MAPPING_PATH, 'r') as f:
        idx_to_label = json.load(f)
    idx_to_label = {int(k): v for k, v in idx_to_label.items()}
    return model, idx_to_label

model, idx_to_label = load_model_and_mapping()

st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
st.markdown("<div class='upload-label'>📤 Upload a pest image for detection</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    img = image.load_img(tmp_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    x = np.expand_dims(img_array, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    pred_class_idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))
    pred_label = idx_to_label.get(pred_class_idx, str(pred_class_idx))

    st.markdown(
        f"<div class='result-box'><div class='prediction'>Prediction: {pred_label}</div>"
        f"<div class='confidence'>Confidence: {confidence*100:.2f}%</div></div>",
        unsafe_allow_html=True
    )

    def grad_cam(img_array, model, layer_name='block_16_project'):
        grad_model = tf.keras.models.Model([model.input], [model.get_layer(layer_name).output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index[None]]
        grads = tape.gradient(loss, conv_outputs)[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-9
        return heatmap

    heatmap = grad_cam(x, model)
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_SUMMER)
    orig = cv2.imread(tmp_path)
    orig = cv2.resize(orig, (224, 224))
    superimposed = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

    st.markdown("<p class='gradcam-title'>🌱 Model Focus Visualization (Grad-CAM)</p>", unsafe_allow_html=True)
    st.image(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB), use_column_width=True)

    st.markdown("""
        <p class='footer'>
        This project demonstrates how AI can support sustainable agriculture by detecting and classifying crop pests efficiently.
        Powered by TensorFlow & Streamlit — designed for farmers, researchers, and innovators.
        </p>
    """, unsafe_allow_html=True)

else:
    st.info("👆 Upload a pest image above to analyze it using the trained MobileNetV2 model.")
