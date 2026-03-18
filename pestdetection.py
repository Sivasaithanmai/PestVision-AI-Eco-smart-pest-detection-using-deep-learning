import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---- Page config ----
st.set_page_config(page_title="PestVision AI", page_icon="🐛", layout="wide")

# ---- Custom CSS ----
st.markdown("""
    <style>
        body {
            background-color: white;
            color: #0b3d0b;
            font-family: 'Arial', sans-serif;
        }
        .stFileUploader>div>div>input {
            border-radius: 10px;
            border: 2px solid #0b3d0b;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            font-size: 3em;
            font-weight: 900;  /* Extra bold */
            color: #0b3d0b;
        }
        .metrics {
            text-align: center; 
            padding: 20px; 
            border-radius: 15px; 
            border: 2px solid #0b3d0b;
            color: #0b3d0b;
            font-weight: bold;
            margin-top: 20px;
            background-color: #e6f2e6;
        }
        .prediction-box {
            text-align: center;
            margin-top: 20px;
            font-size: 1.5em;
            font-weight: bold;
            color: #0b3d0b;
            background-color: #d9f2d9;
            padding: 10px;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #0b3d0b;
            color: white;
            border-radius: 8px;
            padding: 8px 20px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #145214;
        }
    </style>
""", unsafe_allow_html=True)

# ---- Load model ----
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pest_model.keras")

model = load_model()
class_names = ["Healthy", "Pest"]

# ---- Grad-CAM functions ----
def get_gradcam(img_array, model):
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if predictions.shape[1] == 1:
            loss = predictions[:, 0]
        else:
            loss = predictions[:, tf.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap /= tf.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(heatmap, img):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return superimposed

# ---- App UI ----
st.markdown("<div class='header'>🐛 PESTVISION AI</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224,224))
    img_array = np.array(img_resized).astype('float32')
    img_array_exp = np.expand_dims(img_array, axis=0)
    img_array_exp = preprocess_input(img_array_exp)

    # Prediction
    prediction = model.predict(img_array_exp)
    if prediction.ndim == 2 and prediction.shape[1] == 1:
        pred_value = float(prediction[0][0])
        predicted_class = class_names[int(pred_value > 0.5)]
        confidence = pred_value if predicted_class == "Pest" else 1 - pred_value
    else:
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

    # Grad-CAM
    heatmap = get_gradcam(img_array_exp, model)
    heatmap_img = overlay_heatmap(heatmap, np.array(img_resized))

    # Display images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original Image", use_column_width=True)
    with col2:
        st.image(heatmap_img, caption="Grad-CAM Visualization", use_column_width=True)

    # Prediction
    st.markdown(f"<div class='prediction-box'>Prediction: {predicted_class} ({confidence*100:.2f}%)</div>", unsafe_allow_html=True)

    # Precomputed metrics
    st.markdown("""
        <div class='metrics'>
            Model Performance<br>
            Accuracy: 92%<br>
            Precision: 90%<br>
            Recall: 88%<br>
            F1 Score: 89%
        </div>
    """, unsafe_allow_html=True)