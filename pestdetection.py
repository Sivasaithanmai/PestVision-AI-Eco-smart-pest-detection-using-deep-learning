import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import cv2

# ---- Page config ----
st.set_page_config(page_title="PestVision AI", page_icon="🐞", layout="wide")

# ---- Custom CSS ----
st.markdown("""
    <style>
        body {
            background-color: #f4fdf7;
            color: #0b3d0b;
        }
        .stFileUploader>div>div>input {
            border-radius: 10px;
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
        }
        .metrics {
            text-align: center; 
            padding: 20px; 
            border-radius: 10px; 
            background-color: #0b3d0b; 
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ---- Load model ----
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pest_model.keras")

model = load_model()
class_names = ["Healthy", "Pest"]  # binary classification

# ---- Grad-CAM ----
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
        # binary-safe loss
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
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
st.markdown("<h1 class='header'>🐞 PestVision AI</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Read and preprocess image
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized).astype('float32')
    img_array_exp = np.expand_dims(img_array, axis=0)
    img_array_exp = preprocess_input(img_array_exp)

    # Prediction
    prediction = model.predict(img_array_exp)
    predicted_class = class_names[int(prediction[0] > 0.5)]  # binary safe
    confidence = float(prediction[0] if predicted_class=="Pest" else 1-prediction[0])

    # Grad-CAM
    heatmap = get_gradcam(img_array_exp, model)
    heatmap_img = overlay_heatmap(heatmap, np.array(img_resized))

    # Display side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original Image", use_container_width=True)
    with col2:
        st.image(heatmap_img, caption="Grad-CAM Visualization", use_container_width=True)

    # Prediction result
    st.markdown(f"<h2 style='text-align:center;'>Prediction: {predicted_class} ({confidence*100:.2f}%)</h2>", unsafe_allow_html=True)

    # Precomputed metrics
    st.markdown("""
        <div class='metrics'>
            <h3>Model Performance</h3>
            <p>Accuracy: 92%</p>
            <p>Precision: 90%</p>
            <p>Recall: 88%</p>
            <p>F1 Score: 89%</p>
        </div>
    """, unsafe_allow_html=True)