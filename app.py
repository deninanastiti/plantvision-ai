import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="PlantVision AI",
    layout="wide"
)

IMG_SIZE = (150, 150)
MODEL_PATH = "plant_mobilenet_model.keras"
CLASS_NAMES_PATH = "class_names.json"


# ---------------- LOAD FUNCTIONS ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as f:
        return json.load(f)


# ---------------- PREPROCESS ----------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ---------------- PREDICTION ----------------
def predict_image(model, class_names, image: Image.Image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array, verbose=0)[0]

    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[predicted_index]
    confidence = float(prediction[predicted_index]) * 100

    top_3_idx = np.argsort(prediction)[-3:][::-1]
    top_3_predictions = [
        (class_names[i], float(prediction[i]) * 100)
        for i in top_3_idx
    ]

    return predicted_class, confidence, top_3_predictions


# ---------------- HEADER ----------------
st.title("PlantVision AI")

st.markdown(
    "<p style='font-size:15px; color:#222222;'><b>Developed by Denina Nastiti Putri Amani</b></p>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='font-size:14px; color:#555555; margin-top:-10px;'>"
    "Plant Image Classification using MobileNetV2 with Transfer Learning and Data Augmentation"
    "</p>",
    unsafe_allow_html=True
)

st.markdown(
    """
    This application predicts plant types from uploaded images using a deep learning model.
    Upload an image in JPG, JPEG, or PNG format to obtain prediction results.
    """
)

st.divider()


# ---------------- MAIN ----------------
try:
    model = load_model()
    class_names = load_class_names()

    left_col, right_col = st.columns([1, 1], gap="large")

    # -------- LEFT: UPLOAD --------
    with left_col:
        st.subheader("Upload Image")

        uploaded_file = st.file_uploader(
            "Upload plant image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
               st.image(image, caption="Uploaded Image") 
        

    # -------- RIGHT: RESULT --------
    with right_col:
        st.subheader("Prediction Output")

        if uploaded_file is None:
            st.info("Please upload an image to begin.")
        else:
            image = Image.open(uploaded_file)

            if st.button("Run Prediction"):
                with st.spinner("Processing image..."):
                    predicted_class, confidence, top_3_predictions = predict_image(
                        model,
                        class_names,
                        image
                    )

                st.success("Prediction completed")

                col1, col2 = st.columns(2)
                col1.metric("Predicted Class", predicted_class)
                col2.metric("Confidence", f"{confidence:.2f}%")

                st.markdown("### Top 3 Predictions")

                for label, score in top_3_predictions:
                    st.write(f"{label}")
                    st.progress(min(int(score), 100))
                    st.caption(f"{score:.2f}%")

    st.divider()

    # ---------------- FOOTER ----------------
    st.markdown(
        """
        **Model Information**

        - Architecture: MobileNetV2  
        - Task: Image Classification  
        - Input Size: 150 × 150  
        """
    )

except Exception as e:
    st.error(f"Application error: {e}")
