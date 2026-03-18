import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import cv2

st.set_page_config(page_title="PestVision AI", layout="wide")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pest_model.keras")

model = load_model()
class_names = ["Healthy", "Pest"]

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
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    
    # safer Grad-CAM calculation
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

st.markdown("<h1 style='text-align:center;'>PestVision AI</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)
    img_array_exp = np.expand_dims(img_array, axis=0)
    img_array_exp = preprocess_input(img_array_exp)

    
    prediction = model.predict(img_array_exp)
    predicted_class = class_names[np.argmax(prediction)]


    heatmap = get_gradcam(img_array_exp, model)
    heatmap_img = overlay_heatmap(heatmap, np.array(img_resized))

  
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original Image", use_container_width=True)
    with col2:
        st.image(heatmap_img, caption="Grad-CAM Visualization", use_container_width=True)

    st.markdown(f"<h2 style='text-align:center;'>Prediction: {predicted_class}</h2>", unsafe_allow_html=True)
    # Precomputed metrics
    st.markdown("""
    <div style="text-align:center; padding:20px; border-radius:10px; background-color:#111; color:white;">
        <h3>Model Performance</h3>
        <p>Accuracy: 92%</p>
        <p>Precision: 90%</p>
        <p>Recall: 88%</p>
        <p>F1 Score: 89%</p>
    </div>
    """, unsafe_allow_html=True)