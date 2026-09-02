import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
#py -3.11 -m streamlit run app.py use this to run
#PAGE CONFIG
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered"
)

#LOAD MODEL (CACHED)
@st.cache_resource
def load_my_model():
    return load_model("../models/brain_tumor_final_model.h5")

model = load_my_model()

#CLASS LABELS
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
img_size = 224

#HEADER
st.title("🧠 Brain Tumor MRI Classification System")
st.write("AI-powered tumor detection using Deep Learning (CNN)")
st.markdown("---")

#FILE UPLOADER
uploaded_file = st.file_uploader(
    "📁 Upload Brain MRI Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    # Convert image to RGB (important fix)
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    #PREPROCESSING
    img = np.array(image)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = np.reshape(img, (1, img_size, img_size, 3))

    #PREDICTION
    with st.spinner("🔍 Analyzing MRI image... Please wait"):
        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

    st.markdown("---")
    st.subheader("🧾 Prediction Result")

    predicted_class = class_names[class_index]

    # Smart result display
    if predicted_class == "notumor":
        st.success("✅ No Tumor Detected")
    else:
        st.error(f"⚠️ Tumor Detected: {predicted_class.upper()}")

    st.info(f"📊 Confidence Score: {confidence * 100:.2f}%")

    #PROBABILITY CHART
    st.subheader("📈 Class Probabilities")

    fig, ax = plt.subplots()
    ax.bar(class_names, prediction[0])
    ax.set_ylabel("Probability")
    ax.set_ylim([0, 1])
    st.pyplot(fig)

#MODEL DETAILS
with st.expander("Model Details"):
    st.write("Architecture: MOBILENETV2")
    st.write("Dataset Size: 7,200 MRI Images")
    st.write("Classes: Glioma, Meningioma, Pituitary, No Tumor")
    st.write("Training Accuracy: ~95%")
    st.write("Validation Accuracy: ~93%")

st.markdown("---")
st.caption("⚠️ This system is for research and educational purposes only.")
