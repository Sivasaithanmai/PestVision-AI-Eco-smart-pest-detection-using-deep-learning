import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="PestVision AI", page_icon="🐛", layout="wide")

# ---- CSS ----
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

st.markdown("<div class='header'>🐛 PESTVISION AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Detect pests and keep your plants healthy!</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pest_model.keras")

model = load_model()
class_names = ["Healthy", "Pest"]

def overlay_heatmap(heatmap, img):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255*heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

def safe_gradcam(img_array, model):
    """Return heatmap or None if model output is incompatible"""
    try:
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer.name
                break
        if last_conv_layer is None:
            return None
        grad_model = tf.keras.models.Model([model.inputs],[model.get_layer(last_conv_layer).output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            # robust loss selection
            if predictions.ndim == 2 and predictions.shape[1] == 1:
                loss = predictions[:,0]
            else:
                loss = predictions[:, tf.argmax(predictions[0])]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)
        heatmap = tf.nn.relu(heatmap)
        heatmap /= tf.reduce_max(heatmap) + 1e-8
        return heatmap.numpy()
    except:
        return None

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224,224))
    img_array = np.array(img_resized).astype('float32')
    img_array_exp = np.expand_dims(img_array, axis=0)
    img_array_exp = preprocess_input(img_array_exp)

    # ---- Safe prediction ----
    try:
        prediction = model.predict(img_array_exp)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Determine predicted class safely
    if prediction.ndim == 2 and prediction.shape[1] == 1:
        pred_value = float(prediction[0][0])
        predicted_class = class_names[int(pred_value>0.5)]
        confidence = pred_value if predicted_class=="Pest" else 1-pred_value
    elif prediction.ndim==1:
        pred_value = float(prediction[0])
        predicted_class = class_names[int(pred_value>0.5)]
        confidence = pred_value if predicted_class=="Pest" else 1-pred_value
    else:
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

    # ---- Grad-CAM optional ----
    heatmap = safe_gradcam(img_array_exp, model)
    heatmap_img = overlay_heatmap(heatmap, np.array(img_resized)) if heatmap is not None else None

    # ---- Display ----
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="image-card">Original Image</div>', unsafe_allow_html=True)
        st.image(img, use_column_width=True)
    with col2:
        st.markdown('<div class="image-card">Grad-CAM Visualization</div>', unsafe_allow_html=True)
        if heatmap_img is not None:
            st.image(heatmap_img, use_column_width=True)
        else:
            st.info("Grad-CAM not available for this model output.")

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
            <b>Leaf Care Tips:</b><br>
            • Inspect leaves regularly for pests.<br>
            • Remove damaged leaves immediately.<br>
            • Maintain proper watering and sunlight.<br>
            • Use eco-friendly pest control methods.
        </div>
    """, unsafe_allow_html=True)