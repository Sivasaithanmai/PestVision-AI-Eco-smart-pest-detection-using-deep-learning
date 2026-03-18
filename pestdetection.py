import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2
import tempfile
import time

model = tf.keras.models.load_model("model.h5")

st.set_page_config(page_title="PestVision AI", layout="centered")

st.markdown("""
<style>
.big-title {
    text-align:center;
    font-size:40px;
    font-weight:700;
}
.card {
    padding:15px;
    border-radius:12px;
    text-align:center;
    box-shadow:0px 4px 12px rgba(0,0,0,0.1);
}
.fade-in {
    animation: fadeIn 1.2s ease-in;
}
@keyframes fadeIn {
    0% {opacity:0;}
    100% {opacity:1;}
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">PestVision AI</div>', unsafe_allow_html=True)
st.write("Smart pest detection with explainable AI")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

accuracy = 0.91
precision = 0.89
recall = 0.88
f1_score = 0.885

class_names = [
    "Ant", "Bee", "Beetle", "Caterpillar", "Earthworm",
    "Earwig", "Grasshopper", "Moth", "Slug", "Snail", "Wasp"
]

def get_gradcam_heatmap(model, img_array, last_conv_layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

if uploaded_file is not None:

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_path = temp_file.name

    img = image.load_img(temp_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_exp = np.expand_dims(img_array, axis=0)
    img_array_exp = preprocess_input(img_array_exp)

    with st.spinner("Analyzing image..."):
        time.sleep(1.2)

    preds = model.predict(img_array_exp)
    class_index = np.argmax(preds[0])
    predicted_class = class_names[class_index]

    st.markdown(f"""
    <div class="card fade-in" style="background:#c6f7d0;">
        <h2>{predicted_class}</h2>
        <p>Detected Pest</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="card fade-in" style="background:#e6ffe6;">
            <h4>Accuracy</h4>
            <h3>{accuracy:.2f}</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card fade-in" style="background:#fff0f5;margin-top:10px;">
            <h4>Recall</h4>
            <h3>{recall:.2f}</h3>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card fade-in" style="background:#e6f0ff;">
            <h4>Precision</h4>
            <h3>{precision:.2f}</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card fade-in" style="background:#f9f9e6;margin-top:10px;">
            <h4>F1 Score</h4>
            <h3>{f1_score:.2f}</h3>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Model Interpretation")

    heatmap = get_gradcam_heatmap(model, img_array_exp)

    img_cv = cv2.imread(temp_path)
    img_cv = cv2.resize(img_cv, (224, 224))

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img_cv, caption="Original Image")

    with col2:
        st.image(superimposed_img, caption="Grad-CAM Heatmap")

    st.write("AI-powered pest detection with explainable insights.")